from __future__ import annotations

"""Train dynamic AlphaRouter members on four shared portfolio experts.

Labels are recomputed for every training and validation date from subsequent
returns inside the corresponding split.  Test returns are never used to form a
router input, a checkpoint decision, or a calibration parameter.
"""

import argparse
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import DEFAULT_DATASETS, dataset_configuration
from .data import MarketDataBundle, MarketWindowDataset, load_market_data
from .experts import expert_parameters, profile_weights
from .model import DynamicExpertRouter
from .portfolio import (backtest_metrics, ensure_dir, save_backtest_tables,
                        save_json, seed_everything, simulate_portfolio)

METHOD = "AlphaRouter-S"
DATASETS = DEFAULT_DATASETS
EXPERT_NAMES = (
    "slow_sparse",
    "defensive_dense",
    "fast_sparse",
    "adaptive_dense",
)
ROOT = Path("outputs/alpharouter_members")
CACHE_ROOT = Path("outputs/cache")
COST = 0.0025


@dataclass(frozen=True)
class TrainingParameters:
    window: int = 30
    label_horizon: int = 1
    hidden: int = 128
    temporal_heads: int = 8
    temporal_layers: int = 6
    feedforward: int = 256
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    utility_weight: float = 0.25
    soft_target_weight: float = 1.00
    utility_temperature: float = 0.75
    downside_weight: float = 0.25
    drawdown_weight: float = 0.10
    concentration_weight: float = 0.002
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.window < 2 or self.label_horizon < 1:
            raise ValueError(
                "window must be at least 2 and label_horizon must be positive"
            )
        if self.hidden < 1 or self.temporal_heads < 1 or self.temporal_layers < 1:
            raise ValueError(
                "hidden, temporal_heads, and temporal_layers must be positive"
            )
        if self.hidden % self.temporal_heads:
            raise ValueError("hidden must be divisible by temporal_heads")
        if self.feedforward < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("feedforward, batch_size, and epochs must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if not 0.0 <= self.soft_target_weight <= 1.0:
            raise ValueError("soft_target_weight must lie in [0, 1]")
        if self.utility_temperature <= 0.0:
            raise ValueError("utility_temperature must be positive")


@dataclass
class RouterPanel:
    dataset: str
    split: str
    market: MarketWindowDataset
    state: np.ndarray
    experts: np.ndarray
    future_path: np.ndarray
    utility: np.ndarray
    label: np.ndarray


class JointRouterDataset(Dataset):
    def __init__(self, panels: list[RouterPanel]) -> None:
        self.state = np.concatenate([panel.state for panel in panels]).astype(
            np.float32
        )
        utility = np.concatenate([panel.utility for panel in panels]).astype(np.float32)
        center = utility.mean(axis=1, keepdims=True)
        scale = utility.std(axis=1, keepdims=True)
        self.utility = np.clip((utility - center) / np.maximum(scale, 1e-5), -5.0, 5.0)
        self.label = np.concatenate([panel.label for panel in panels]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self.state[index]),
            "utility": torch.from_numpy(self.utility[index]),
            "label": torch.tensor(self.label[index], dtype=torch.long),
        }


def _market_state(
    bundle: MarketDataBundle, indices: np.ndarray, window: int
) -> np.ndarray:
    """Return 21 causal return descriptors without a market identity channel."""
    output = np.empty((len(indices), window, 21), dtype=np.float32)
    for row, target in enumerate(indices):
        values = bundle.returns[int(target) - window : int(target)].astype(np.float64)
        output[row, :, 0] = values.mean(axis=1)
        output[row, :, 1] = values.std(axis=1)
        output[row, :, 2] = np.quantile(values, 0.10, axis=1)
        output[row, :, 3] = np.quantile(values, 0.50, axis=1)
        output[row, :, 4] = np.quantile(values, 0.90, axis=1)
        output[row, :, 5] = (values > 0.0).mean(axis=1)
        descriptors: list[float] = []
        for horizon in (20, 60, 220):
            history = bundle.returns[
                max(0, int(target) - horizon) : int(target)
            ].astype(np.float64)
            market_return = history.mean(axis=1)
            asset_variance = history.var(axis=0).mean()
            correlation_proxy = market_return.var() / max(asset_variance, 1e-12)
            descriptors.extend(
                [
                    float(market_return.mean()),
                    float(market_return.std()),
                    float(history.std(axis=1).mean()),
                    float((history > 0.0).mean()),
                    float(np.clip(correlation_proxy, 0.0, 1.0)),
                ]
            )
        output[row, :, 6:21] = np.asarray(descriptors, dtype=np.float32)[None, :]
    return output


def _state_with_expert_evidence(
    bundle: MarketDataBundle,
    indices: np.ndarray,
    window: int,
    experts: np.ndarray,
) -> np.ndarray:
    market = _market_state(bundle, indices, window)
    expert_history = np.empty(
        (len(indices), window, len(EXPERT_NAMES)), dtype=np.float32
    )
    for row, target in enumerate(indices):
        history = bundle.returns[int(target) - window : int(target)].astype(np.float64)
        # How today's four feasible portfolios would have behaved over the
        # observable trailing window.  No return at or after target is included.
        expert_history[row] = np.einsum("wa,ka->wk", history, experts[row])
    return np.concatenate([market, expert_history], axis=2).astype(np.float32)


def _profile_cache(dataset: str, split: str, window: int, horizon: int) -> Path:
    reset = "_annualreset" if split == "test" else ""
    return CACHE_ROOT / f"{dataset}_{split}_w{window}_h{horizon}{reset}_v3.npz"


def _expert_bank(
    bundle: MarketDataBundle,
    dataset: str,
    split: str,
    market: MarketWindowDataset,
) -> np.ndarray:
    path = _profile_cache(dataset, split, market.window, market.horizon)
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        if np.array_equal(payload["indices"], market.indices):
            return payload["experts"].astype(np.float32)
    values = [
        profile_weights(bundle, market, split, expert_parameters(name))
        for name in EXPERT_NAMES
    ]
    experts = np.stack(values, axis=1).astype(np.float32)
    ensure_dir(path.parent)
    np.savez_compressed(path, indices=market.indices, experts=experts)
    return experts


def _candidate_turnover(
    bundle: MarketDataBundle, market: MarketWindowDataset, experts: np.ndarray
) -> np.ndarray:
    n, k, assets = experts.shape
    output = np.empty((n, k), dtype=np.float64)
    previous = np.full((k, assets), 1.0 / assets, dtype=np.float64)
    previous_target: int | None = None
    for row, target_value in enumerate(market.indices):
        target = int(target_value)
        if previous_target is not None and target == previous_target + 1:
            realized = bundle.returns[target - 1].astype(np.float64)
            drifted = previous * (1.0 + realized[None, :])
            previous = drifted / np.maximum(drifted.sum(axis=1, keepdims=True), 1e-12)
        elif previous_target is not None:
            previous[:] = 1.0 / assets
        output[row] = np.abs(experts[row] - previous).sum(axis=1)
        previous = experts[row].astype(np.float64).copy()
        previous_target = target
    return output


def _dynamic_utility(
    future_path: np.ndarray,
    experts: np.ndarray,
    turnover: np.ndarray,
    parameters: TrainingParameters,
) -> np.ndarray:
    gross_path = future_path.astype(np.float64)
    turnover_path = turnover.astype(np.float64)
    if turnover_path.ndim == 2:
        turnover_path = turnover_path[:, None, :]
    if turnover_path.shape[1] not in (1, gross_path.shape[1]):
        raise ValueError("turnover must contain one value or one value per horizon day")
    path = np.clip((1.0 + gross_path) * (1.0 - COST * turnover_path) - 1.0, -0.99, None)
    log_growth = np.log1p(path).sum(axis=1)
    downside = np.sqrt(np.square(np.minimum(path, 0.0)).mean(axis=1)) * np.sqrt(
        path.shape[1]
    )
    wealth = np.cumprod(1.0 + path, axis=1)
    peak = np.maximum.accumulate(
        np.concatenate([np.ones_like(wealth[:, :1]), wealth], axis=1), axis=1
    )[:, 1:]
    drawdown = np.max(1.0 - wealth / np.maximum(peak, 1e-12), axis=1)
    assets = experts.shape[-1]
    hhi = np.square(experts.astype(np.float64)).sum(axis=-1)
    if hhi.ndim == 3:
        hhi = hhi.mean(axis=1)
    concentration = np.clip((assets * hhi - 1.0) / max(assets - 1.0, 1.0), 0.0, 1.0)
    return (
        log_growth
        - parameters.downside_weight * downside
        - parameters.drawdown_weight * drawdown
        - parameters.concentration_weight * concentration
    ).astype(np.float32)


def build_panel(
    bundle: MarketDataBundle,
    dataset: str,
    split: str,
    window: int,
    horizon: int,
    parameters: TrainingParameters | None = None,
) -> RouterPanel:
    parameters = TrainingParameters() if parameters is None else parameters
    market = MarketWindowDataset(bundle, split, window, horizon)
    # Candidate rules are recomputed on every future decision day.  This makes
    # the supervision match deployment, where the selected expert is a trading
    # rule whose feasible portfolio changes with the causal information set.
    daily_market = MarketWindowDataset(bundle, split, window, 1)
    daily_experts = _expert_bank(bundle, dataset, split, daily_market)
    positions = np.searchsorted(daily_market.indices, market.indices)
    if not np.array_equal(daily_market.indices[positions], market.indices):
        raise RuntimeError("label dates are not aligned with the daily expert path")
    experts = daily_experts[positions]
    daily_gross = np.einsum(
        "na,nka->nk", bundle.returns[daily_market.indices], daily_experts
    ).astype(np.float32)
    gross_windows = np.lib.stride_tricks.sliding_window_view(
        daily_gross, window_shape=horizon, axis=0
    ).transpose(0, 2, 1)
    future_path = gross_windows[positions].astype(np.float32)
    if split == "test":
        # Test outcomes are carried only for the final portfolio simulator.
        # They are deliberately not converted into supervision or labels.
        utility = np.zeros((len(market), len(EXPERT_NAMES)), dtype=np.float32)
        label = np.full(len(market), -1, dtype=np.int64)
    else:
        daily_turnover = _candidate_turnover(bundle, daily_market, daily_experts)
        turnover_windows = np.lib.stride_tricks.sliding_window_view(
            daily_turnover, window_shape=horizon, axis=0
        ).transpose(0, 2, 1)
        expert_windows = np.lib.stride_tricks.sliding_window_view(
            daily_experts, window_shape=horizon, axis=0
        ).transpose(0, 3, 1, 2)
        utility = _dynamic_utility(
            future_path,
            expert_windows[positions],
            turnover_windows[positions],
            parameters,
        )
        label = utility.argmax(axis=1).astype(np.int64)
    return RouterPanel(
        dataset=dataset,
        split=split,
        market=market,
        state=_state_with_expert_evidence(bundle, market.indices, window, experts),
        experts=experts,
        future_path=future_path,
        utility=utility,
        label=label,
    )


def _standardize(*groups: list[RouterPanel]) -> tuple[np.ndarray, np.ndarray]:
    train = groups[0]
    values = np.concatenate([panel.state for panel in train])
    mean = values.mean(axis=(0, 1), keepdims=True)
    std = values.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    for group in groups:
        for panel in group:
            panel.state = np.nan_to_num((panel.state - mean) / std).astype(np.float32)
    return mean.reshape(-1), std.reshape(-1)


def _loss(
    model: DynamicExpertRouter,
    batch: dict[str, torch.Tensor],
    parameters: TrainingParameters,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, gate = model(batch["state"].to(device))
    target = batch["label"].to(device)
    utility = batch["utility"].to(device)
    classification = nn.functional.cross_entropy(logits, target, label_smoothing=0.02)
    soft_target = torch.softmax(utility / parameters.utility_temperature, dim=1)
    distribution = nn.functional.kl_div(
        nn.functional.log_softmax(logits, dim=1), soft_target, reduction="batchmean"
    )
    regret = (utility.max(dim=1).values - (gate * utility).sum(dim=1)).mean()
    total = (
        (1.0 - parameters.soft_target_weight) * classification
        + parameters.soft_target_weight * distribution
        + parameters.utility_weight * regret
    )
    accuracy = (logits.argmax(dim=1) == target).float().mean()
    return total, {
        "loss": float(total.detach()),
        "classification": float(classification.detach()),
        "distribution": float(distribution.detach()),
        "utility_regret": float(regret.detach()),
        "accuracy": float(accuracy.detach()),
    }


@torch.no_grad()
def predict_probability(
    model: DynamicExpertRouter,
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(state), batch_size):
        logits, _ = model(
            torch.from_numpy(state[start : start + batch_size]).to(device)
        )
        values.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values)


def hard_gate(probability: np.ndarray, switching_margin: float = 0.0) -> np.ndarray:
    log_probability = np.log(np.clip(probability, 1e-12, 1.0))
    choice = np.empty(len(probability), dtype=np.int64)
    choice[0] = int(log_probability[0].argmax())
    for row in range(1, len(probability)):
        candidate = int(log_probability[row].argmax())
        previous = int(choice[row - 1])
        if (
            candidate != previous
            and (log_probability[row, candidate] - log_probability[row, previous])
            < switching_margin
        ):
            candidate = previous
        choice[row] = candidate
    return np.eye(probability.shape[1], dtype=np.float32)[choice]


def dynamic_label_gate(panel: RouterPanel) -> np.ndarray:
    """Return the auditable per-date best-expert labels inside labelled splits."""
    if np.any(panel.label < 0):
        raise ValueError("dynamic labels are unavailable for the test split")
    return np.eye(len(EXPERT_NAMES), dtype=np.float32)[panel.label]


def _metrics(panel: RouterPanel, gate: np.ndarray):
    weights = np.einsum("ne,nea->na", gate.astype(np.float64), panel.experts)
    net, wealth, turnover = simulate_portfolio(weights, panel.market.next_returns, COST)
    return backtest_metrics(net, wealth, turnover), weights, net, wealth, turnover


def _portfolio_score(metrics: dict[str, float]) -> float:
    return (
        metrics["annualized_sharpe"]
        + 0.25 * metrics["calmar_ratio"]
        + 0.10 * metrics["annualized_return"]
    )


def _calibrate_switching_margin(
    model: DynamicExpertRouter,
    validation: list[RouterPanel],
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict]:
    probabilities = [
        predict_probability(model, panel.state, device, batch_size)
        for panel in validation
    ]
    best = None
    rows = []
    for margin in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50):
        scores = []
        switch_rates = []
        for panel, probability in zip(validation, probabilities):
            gate = hard_gate(probability, margin)
            scores.append(_portfolio_score(_metrics(panel, gate)[0]))
            choice = gate.argmax(axis=1)
            switch_rates.append(float(np.mean(choice[1:] != choice[:-1])))
        candidate = (float(np.mean(scores)), -float(np.mean(switch_rates)), -margin)
        rows.append(
            {
                "margin": margin,
                "mean_portfolio_score": candidate[0],
                "mean_switch_rate": -candidate[1],
            }
        )
        if best is None or candidate > best[0]:
            best = (candidate, margin)
    assert best is not None
    return float(best[1]), {"grid": rows, "selected_margin": float(best[1])}


def _save_result(
    panel: RouterPanel,
    gate: np.ndarray,
    method: str,
    output: Path,
    bundle: MarketDataBundle,
) -> dict:
    metrics, weights, net, wealth, turnover = _metrics(panel, gate)
    target = ensure_dir(
        output / panel.dataset / method.replace(" ", "_").replace("/", "_")
    )
    save_backtest_tables(
        panel.market.dates, weights, net, wealth, bundle.tickers, target
    )
    pd.DataFrame(
        {"date": pd.to_datetime(panel.market.dates), "turnover": turnover}
    ).to_csv(target / "turnover.csv", index=False)
    result = {
        **metrics,
        "dataset": panel.dataset,
        "method": method,
        "maximum_weight": float(weights.max()),
        "simplex_max_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
        "evidence_status": "retrospective_current_panel_development",
    }
    save_json(result, target / "metrics.json")
    return result


def _label_audit(panel: RouterPanel) -> dict:
    counts = np.bincount(panel.label, minlength=len(EXPERT_NAMES))
    return {
        "dataset": panel.dataset,
        "split": panel.split,
        "observations": int(len(panel.label)),
        "label_changes": int(np.sum(panel.label[1:] != panel.label[:-1])),
        "label_change_rate": float(np.mean(panel.label[1:] != panel.label[:-1])),
        "expert_counts": {name: int(counts[i]) for i, name in enumerate(EXPERT_NAMES)},
        "expert_fractions": {
            name: float(counts[i] / len(panel.label))
            for i, name in enumerate(EXPERT_NAMES)
        },
    }


def run(
    seed: int,
    parameters: TrainingParameters,
    device_name: str,
    evaluate_test: bool = True,
    datasets: tuple[str, ...] = DATASETS,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    started_at = time.perf_counter()
    parameters.validate()
    if not datasets:
        raise ValueError("at least one dataset is required")
    seed_everything(seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    bundles = {
        name: load_market_data(dataset_configuration(name, data_dir))
        for name in datasets
    }
    train = [
        build_panel(
            bundles[name],
            name,
            "train",
            parameters.window,
            parameters.label_horizon,
            parameters,
        )
        for name in datasets
    ]
    validation_labels = [
        build_panel(
            bundles[name],
            name,
            "val",
            parameters.window,
            parameters.label_horizon,
            parameters,
        )
        for name in datasets
    ]
    validation = [
        build_panel(bundles[name], name, "val", parameters.window, 1, parameters)
        for name in datasets
    ]
    test = (
        [
            build_panel(bundles[name], name, "test", parameters.window, 1, parameters)
            for name in datasets
        ]
        if evaluate_test
        else []
    )
    mean, std = _standardize(train, validation_labels, validation, test)
    dataset = JointRouterDataset(train)
    loader = DataLoader(
        dataset,
        batch_size=parameters.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1907),
        num_workers=0,
    )
    model = DynamicExpertRouter(
        input_features=train[0].state.shape[-1],
        experts=len(EXPERT_NAMES),
        window=parameters.window,
        hidden=parameters.hidden,
        heads=parameters.temporal_heads,
        layers=parameters.temporal_layers,
        feedforward=parameters.feedforward,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    history = []
    for epoch in range(1, parameters.epochs + 1):
        model.train()
        batch_rows = []
        for batch in loader:
            loss, stats = _loss(model, batch, parameters, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), parameters.max_grad_norm)
            optimizer.step()
            batch_rows.append(stats)
        validation_scores = []
        for panel in validation:
            probability = predict_probability(
                model, panel.state, device, parameters.batch_size
            )
            validation_scores.append(
                _portfolio_score(_metrics(panel, hard_gate(probability))[0])
            )
        correct = total = 0
        for panel in validation_labels:
            probability = predict_probability(
                model, panel.state, device, parameters.batch_size
            )
            correct += int((probability.argmax(axis=1) == panel.label).sum())
            total += len(panel.label)
        accuracy = correct / total
        portfolio_score = float(np.mean(validation_scores))
        row = {
            "epoch": epoch,
            **{
                key: float(np.mean([item[key] for item in batch_rows]))
                for key in batch_rows[0]
            },
            "validation_dynamic_label_accuracy": accuracy,
            "validation_portfolio_score": portfolio_score,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{METHOD}/seed{seed}] epoch={epoch} dynamic_acc={accuracy:.4f} "
                f"val_score={portfolio_score:.3f}",
                flush=True,
            )
        # Actual validation portfolio quality is the primary checkpoint rule.
        key = (portfolio_score, accuracy)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    output = ensure_dir(ROOT / f"seed{seed}")
    switching_margin, calibration = _calibrate_switching_margin(
        model, validation, device, parameters.batch_size
    )
    label_audit = [_label_audit(panel) for panel in train + validation_labels]
    save_json(
        {
            "dynamic_label_definition": (
                "argmax risk-adjusted utility of the daily recomputed expert rule "
                f"over the next {parameters.label_horizon} split-contained observations"
            ),
            "label_deployment_alignment": "daily expert-rule path with daily transaction costs",
            "fixed_market_labels": False,
            "expert_order": list(EXPERT_NAMES),
            "label_audit": label_audit,
        },
        output / "dynamic_label_audit.json",
    )
    save_json(calibration, output / "validation_calibration.json")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "best_epoch": best_epoch,
            "parameters": asdict(parameters),
            "expert_order": EXPERT_NAMES,
            "state_mean": mean,
            "state_std": std,
            "switching_margin": switching_margin,
        },
        output / "router.pt",
    )
    save_json({"history": history}, output / "training_history.json")
    if not evaluate_test:
        elapsed = float(time.perf_counter() - started_at)
        save_json(
            {
                "seed": seed,
                "device": str(device),
                "elapsed_seconds": elapsed,
                "epochs_requested": parameters.epochs,
                "best_epoch": best_epoch,
                "best_validation_portfolio_score": float(best_key[0]),
                "best_validation_dynamic_label_accuracy": float(best_key[1]),
                "test_evaluated": False,
            },
            output / "runtime.json",
        )
        return pd.DataFrame(
            [
                {
                    "dataset": "validation",
                    "method": METHOD,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_portfolio_score": float(best_key[0]),
                    "validation_dynamic_label_accuracy": float(best_key[1]),
                }
            ]
        )
    rows = []
    routes = []
    for panel in test:
        probability = predict_probability(
            model, panel.state, device, parameters.batch_size
        )
        dynamic_gate = hard_gate(probability, switching_margin)
        result = _save_result(
            panel, dynamic_gate, METHOD, output, bundles[panel.dataset]
        )
        result.update({"seed": seed, "best_epoch": best_epoch})
        rows.append(result)
        choice = dynamic_gate.argmax(axis=1)
        for index, date in enumerate(panel.market.dates):
            routes.append(
                {
                    "dataset": panel.dataset,
                    "date": date,
                    "selected_expert": EXPERT_NAMES[int(choice[index])],
                    "switch": bool(index > 0 and choice[index] != choice[index - 1]),
                    **{
                        f"p_{name}": float(probability[index, k])
                        for k, name in enumerate(EXPERT_NAMES)
                    },
                }
            )
    frame = pd.DataFrame(rows)
    elapsed = float(time.perf_counter() - started_at)
    save_json(
        {
            "seed": seed,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "epochs_requested": parameters.epochs,
            "best_epoch": best_epoch,
        },
        output / "runtime.json",
    )
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(routes).to_csv(output / "routing_probabilities.csv", index=False)
    return frame


def main() -> None:
    global CACHE_ROOT, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--epochs", type=int, default=TrainingParameters.epochs)
    parser.add_argument(
        "--label-horizon", type=int, default=TrainingParameters.label_horizon
    )
    parser.add_argument("--hidden", type=int, default=TrainingParameters.hidden)
    parser.add_argument(
        "--layers", type=int, default=TrainingParameters.temporal_layers
    )
    parser.add_argument("--heads", type=int, default=TrainingParameters.temporal_heads)
    parser.add_argument(
        "--feedforward", type=int, default=TrainingParameters.feedforward
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=TrainingParameters.batch_size)
    parser.add_argument(
        "--utility-weight", type=float, default=TrainingParameters.utility_weight
    )
    parser.add_argument(
        "--soft-target-weight",
        type=float,
        default=TrainingParameters.soft_target_weight,
    )
    parser.add_argument(
        "--utility-temperature",
        type=float,
        default=TrainingParameters.utility_temperature,
    )
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="research panel names, or 'synthetic' for an offline smoke run",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="train and select checkpoints without constructing or evaluating the test split",
    )
    args = parser.parse_args()
    ROOT = Path(args.output_root)
    CACHE_ROOT = Path(args.cache_root)
    datasets = tuple(dict.fromkeys(args.datasets))
    frames = [
        run(
            seed,
            TrainingParameters(
                epochs=args.epochs,
                label_horizon=args.label_horizon,
                hidden=args.hidden,
                temporal_layers=args.layers,
                temporal_heads=args.heads,
                feedforward=args.feedforward,
                batch_size=args.batch_size,
                utility_weight=args.utility_weight,
                soft_target_weight=args.soft_target_weight,
                utility_temperature=args.utility_temperature,
            ),
            args.device,
            evaluate_test=not args.validation_only,
            datasets=datasets,
            data_dir=args.data_dir,
        )
        for seed in args.seeds
    ]
    combined = pd.concat(frames, ignore_index=True)
    ensure_dir(ROOT)
    destination = ROOT / "all_seed_metrics.csv"
    if destination.exists():
        previous = pd.read_csv(destination)
        previous = previous[~previous.seed.isin(args.seeds)]
        combined = pd.concat([previous, combined], ignore_index=True)
    combined = combined.sort_values(["dataset", "method", "seed"]).reset_index(
        drop=True
    )
    combined.to_csv(destination, index=False)
    display_columns = [
        column
        for column in (
            "dataset",
            "method",
            "seed",
            "cumulative_wealth",
            "annualized_sharpe",
            "calmar_ratio",
            "validation_portfolio_score",
            "validation_dynamic_label_accuracy",
        )
        if column in combined.columns
    ]
    print(combined[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
