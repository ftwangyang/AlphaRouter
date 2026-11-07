from __future__ import annotations

"""Evaluate the validation-calibrated AlphaRouter consensus committee."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import experiment as experiment_module
from .config import DEFAULT_DATASETS, dataset_configuration
from .data import load_market_data
from .experiment import (
    COST,
    EXPERT_NAMES,
    TrainingParameters,
    _portfolio_score,
    build_panel,
)
from .model import DynamicExpertRouter
from .portfolio import (
    backtest_metrics,
    ensure_dir,
    save_backtest_tables,
    save_json,
    simulate_portfolio,
)
from .protocol import IMPLEMENTATION_ID, bundle_fingerprint, source_fingerprint
from .routing import (
    ADAPTIVE_SHARES,
    CONFIDENCE_THRESHOLDS,
    blend_reference,
    confidence_votes,
    majority_gate,
)

METHOD = "AlphaRouter"
SEEDS = (101, 202, 303, 404, 505)
BASE_ROOT = Path("outputs/alpharouter_members_paper")
ROOT = Path("outputs/alpharouter_committee_paper")


def _load_member(
    seed: int, device: torch.device
) -> tuple[DynamicExpertRouter, dict[str, Any], dict[str, Any]]:
    folder = BASE_ROOT / f"seed{seed}"
    payload = torch.load(folder / "router.pt", map_location=device, weights_only=False)
    if payload.get("implementation_id") != IMPLEMENTATION_ID:
        raise RuntimeError(
            "checkpoint uses a different protocol; retrain under the paper equations"
        )
    if payload.get("source_sha256") != source_fingerprint():
        raise RuntimeError(
            "checkpoint source fingerprint differs from the current implementation"
        )
    with (folder / "dynamic_label_audit.json").open(encoding="utf-8") as stream:
        audit = json.load(stream)
    if tuple(payload.get("expert_order", ())) != EXPERT_NAMES:
        raise RuntimeError(f"checkpoint seed {seed} uses an incompatible expert order")
    parameters = payload["parameters"]
    model = DynamicExpertRouter(
        input_features=len(payload["state_mean"]),
        experts=len(EXPERT_NAMES),
        window=int(parameters["window"]),
        hidden=int(parameters["hidden"]),
        heads=int(parameters["temporal_heads"]),
        layers=int(parameters["temporal_layers"]),
        feedforward=int(parameters["feedforward"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, audit


@torch.no_grad()
def _probabilities(
    members: list[tuple[DynamicExpertRouter, dict[str, Any], dict[str, Any]]],
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    for model, payload, _ in members:
        mean = np.asarray(payload["state_mean"])[None, None, :]
        std = np.asarray(payload["state_std"])[None, None, :]
        standardized = (state - mean) / std
        batches = []
        for start in range(0, len(state), batch_size):
            stop = start + batch_size
            values = torch.from_numpy(standardized[start:stop].astype(np.float32)).to(
                device
            )
            logits, _ = model(values)
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        result.append(np.concatenate(batches))
    return np.stack(result)


def _training_prior(
    panel: Any,
) -> tuple[np.ndarray, dict]:
    """Select Eq. (14)'s reference using all training observations."""
    if panel.split != "train":
        raise ValueError("the reference expert must be selected on the training split")
    length = len(panel.market.next_returns)
    scores = []
    for expert in range(len(EXPERT_NAMES)):
        gate = np.eye(len(EXPERT_NAMES), dtype=np.float32)[
            np.full(length, expert, dtype=np.int64)
        ]
        scores.append(float(_portfolio_score(_metrics(panel, gate)[0])))
    selected = int(np.argmax(scores))
    prior = np.eye(len(EXPERT_NAMES), dtype=np.float32)[selected]
    audit = {
        "training_observations": int(length),
        "score_definition": "ASR + 0.25 CR + 0.10 APY",
        "scores": {name: value for name, value in zip(EXPERT_NAMES, scores)},
        "selected_expert": EXPERT_NAMES[selected],
    }
    return prior, audit


def _metrics(panel: Any, gate: np.ndarray) -> tuple:
    weights = np.einsum("ne,nea->na", gate.astype(np.float64), panel.experts)
    net, wealth, turnover = simulate_portfolio(weights, panel.market.next_returns, COST)
    return backtest_metrics(net, wealth, turnover), weights, net, wealth, turnover


def _calibrate(
    members: list[tuple[DynamicExpertRouter, dict[str, Any], dict[str, Any]]],
    validation: list[Any],
    priors: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float], list[dict[str, float]]]:
    probabilities = {
        panel.dataset: _probabilities(members, panel.state, device, batch_size)
        for panel in validation
    }
    rows = []
    per_market: dict[str, tuple[tuple[float, ...], float, float]] = {}
    best: tuple[tuple[float, ...], float, float] | None = None
    for margin in CONFIDENCE_THRESHOLDS:
        for adaptive_share in ADAPTIVE_SHARES:
            scores = []
            switches = []
            for panel in validation:
                dynamic = majority_gate(
                    probabilities[panel.dataset],
                    margin,
                    int(priors[panel.dataset].argmax()),
                )
                gate = blend_reference(priors[panel.dataset], dynamic, adaptive_share)
                score = float(_portfolio_score(_metrics(panel, gate)[0]))
                choice = dynamic.argmax(axis=1)
                switch_rate = (
                    float(np.mean(choice[1:] != choice[:-1]))
                    if len(choice) > 1
                    else 0.0
                )
                scores.append(score)
                switches.append(switch_rate)
                market_key = (score, -switch_rate, -adaptive_share, -margin)
                current = per_market.get(panel.dataset)
                if current is None or market_key > current[0]:
                    per_market[panel.dataset] = (
                        market_key,
                        margin,
                        adaptive_share,
                    )
            key = (
                float(np.mean(scores)),
                -float(np.mean(switches)),
                -adaptive_share,
                -margin,
            )
            rows.append(
                {
                    "confidence_threshold": margin,
                    "adaptive_share": adaptive_share,
                    "mean_validation_score": key[0],
                    "router_switch_rate": -key[1],
                }
            )
            if best is None or key > best[0]:
                best = (key, margin, adaptive_share)
    if best is None:
        raise RuntimeError("calibration grid is empty")
    selected = {
        dataset: {
            "confidence_threshold": float(value[1]),
            "adaptive_share": float(value[2]),
        }
        for dataset, value in per_market.items()
    }
    global_selection = {
        "confidence_threshold": float(best[1]),
        "adaptive_share": float(best[2]),
    }
    return selected, global_selection, rows


def _save_result(panel: Any, gate: np.ndarray, output: Path, bundle: Any) -> dict:
    metrics, weights, net, wealth, turnover = _metrics(panel, gate)
    target = ensure_dir(output / panel.dataset / METHOD)
    save_backtest_tables(
        panel.market.dates, weights, net, wealth, bundle.tickers, target
    )
    pd.DataFrame(
        {"date": pd.to_datetime(panel.market.dates), "turnover": turnover}
    ).to_csv(target / "turnover.csv", index=False)
    result = {
        **metrics,
        "dataset": panel.dataset,
        "method": METHOD,
        "maximum_weight": float(weights.max()),
        "simplex_max_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
        "evidence_status": "retrospective_development",
        "implementation_id": IMPLEMENTATION_ID,
    }
    save_json(result, target / "metrics.json")
    return result


def run(
    seeds: tuple[int, ...],
    device_name: str,
    batch_size: int,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    data_dir: str | Path = "data",
    cache_root: str | Path = "outputs/cache_paper",
) -> pd.DataFrame:
    if not seeds:
        raise ValueError("at least one member seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("committee seeds must be distinct")
    if batch_size < 1 or not datasets:
        raise ValueError("batch_size and datasets must be non-empty")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    members = [_load_member(seed, device) for seed in seeds]
    first_parameters = members[0][1]["parameters"]
    if any(member[1]["parameters"] != first_parameters for member in members[1:]):
        raise RuntimeError("all committee checkpoints must use identical parameters")
    parameters = TrainingParameters(**first_parameters)
    parameters.validate()

    experiment_module.CACHE_ROOT = Path(cache_root)
    bundles = {
        name: load_market_data(dataset_configuration(name, data_dir))
        for name in datasets
    }
    data_hashes = {name: bundle_fingerprint(bundle) for name, bundle in bundles.items()}
    for _, payload, _ in members:
        if (
            payload.get("datasets") != list(datasets)
            or payload.get("data_sha256") != data_hashes
        ):
            raise RuntimeError(
                "checkpoint training datasets or data fingerprints do not match"
            )
    training = [
        build_panel(bundles[name], name, "train", parameters.window, 1, parameters)
        for name in datasets
    ]
    validation = [
        build_panel(bundles[name], name, "val", parameters.window, 1, parameters)
        for name in datasets
    ]
    prior_payload = {panel.dataset: _training_prior(panel) for panel in training}
    priors = {dataset: value[0] for dataset, value in prior_payload.items()}
    prior_audit = {dataset: value[1] for dataset, value in prior_payload.items()}
    calibration, global_calibration, grid = _calibrate(
        members, validation, priors, device, batch_size
    )

    output = ensure_dir(ROOT)
    save_json(
        {
            "method": METHOD,
            "member_seeds": list(seeds),
            "implementation_id": IMPLEMENTATION_ID,
            "source_sha256": source_fingerprint(),
            "data_sha256": data_hashes,
            "shared_experts": list(EXPERT_NAMES),
            "per_market_validation_calibration": calibration,
            "global_validation_calibration": global_calibration,
            "confidence_threshold_grid": list(CONFIDENCE_THRESHOLDS),
            "adaptive_share_grid": list(ADAPTIVE_SHARES),
            "train_only_expert_prior": prior_audit,
            "validation_grid": grid,
        },
        output / "committee_calibration.json",
    )

    # All train/validation choices are written before the final split is built.
    test = [
        build_panel(bundles[name], name, "test", parameters.window, 1, parameters)
        for name in datasets
    ]
    rows = []
    routes = []
    for panel in test:
        selection = calibration[panel.dataset]
        probability = _probabilities(members, panel.state, device, batch_size)
        reference = int(priors[panel.dataset].argmax())
        consensus = majority_gate(
            probability, selection["confidence_threshold"], reference
        )
        gate = blend_reference(
            priors[panel.dataset], consensus, selection["adaptive_share"]
        )
        row = _save_result(panel, gate, output, bundles[panel.dataset])
        row.update(
            {
                "member_seeds": " ".join(map(str, seeds)),
                "adaptive_share": selection["adaptive_share"],
                "confidence_threshold": selection["confidence_threshold"],
                "prior_expert": prior_audit[panel.dataset]["selected_expert"],
            }
        )
        rows.append(row)
        choice = consensus.argmax(axis=1)
        votes = confidence_votes(
            probability, selection["confidence_threshold"], reference
        )
        for index, date in enumerate(panel.market.dates):
            routes.append(
                {
                    "dataset": panel.dataset,
                    "date": date,
                    "prior_expert": prior_audit[panel.dataset]["selected_expert"],
                    "routed_expert": EXPERT_NAMES[int(choice[index])],
                    "adaptive_share": selection["adaptive_share"],
                    "switch": bool(index > 0 and choice[index] != choice[index - 1]),
                    **{
                        f"votes_{name}": int((votes[:, index] == expert).sum())
                        for expert, name in enumerate(EXPERT_NAMES)
                    },
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(routes).to_csv(output / "routing.csv", index=False)
    return frame


def main() -> None:
    global BASE_ROOT, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-root", default=str(BASE_ROOT))
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--cache-root", default="outputs/cache_paper")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
        help="the same research panels used to train the committee members",
    )
    args = parser.parse_args()
    BASE_ROOT = Path(args.base_root)
    ROOT = Path(args.output_root)
    datasets = tuple(dict.fromkeys(args.datasets))
    frame = run(
        tuple(args.seeds),
        args.device,
        args.batch_size,
        datasets=datasets,
        data_dir=args.data_dir,
        cache_root=args.cache_root,
    )
    columns = [
        "dataset",
        "method",
        "cumulative_wealth",
        "annualized_sharpe",
        "calmar_ratio",
    ]
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
