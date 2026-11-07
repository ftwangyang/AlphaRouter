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
from .experts import expert_parameters, generate_expert_weights
from .model import DynamicExpertRouter
from .portfolio import (
    backtest_metrics,
    ensure_dir,
    save_json,
    seed_everything,
    simulate_portfolio,
)
from .protocol import IMPLEMENTATION_ID, bundle_fingerprint, source_fingerprint
from .routing import member_gate

METHOD = "AlphaRouter-S"
DATASETS = DEFAULT_DATASETS
EXPERT_NAMES = (
    "slow_sparse",
    "defensive_dense",
    "fast_sparse",
    "adaptive_dense",
)
ROOT = Path("outputs/alpharouter_members_paper")
CACHE_ROOT = Path("outputs/cache_paper")
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
        # Eqs. (12)-(13) use raw Eq. (11) scores, without a date-wise z-score.
        self.utility = np.concatenate([panel.utility for panel in panels]).astype(
            np.float32
        )
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
    """Return Eq. (10)'s past daily states, each causal at its own date."""
    last = int(indices[-1])
    daily = np.empty((last, 21), dtype=np.float32)
    values = bundle.returns[:last].astype(np.float64)
    daily[:, 0] = values.mean(axis=1)
    daily[:, 1] = values.std(axis=1)
    daily[:, 2] = np.quantile(values, 0.10, axis=1)
    daily[:, 3] = np.quantile(values, 0.50, axis=1)
    daily[:, 4] = np.quantile(values, 0.90, axis=1)
    daily[:, 5] = (values > 0.0).mean(axis=1)
    for date in range(last):
        descriptors: list[float] = []
        for horizon in (20, 60, 220):
            history = values[max(0, date + 1 - horizon) : date + 1]
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
        daily[date, 6:21] = np.asarray(descriptors, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        daily, window_shape=window, axis=0
    ).transpose(0, 2, 1)
    return windows[indices - window].copy()


def _state_with_expert_evidence(
    bundle: MarketDataBundle,
    indices: np.ndarray,
    window: int,
    expert_path: np.ndarray,
) -> np.ndarray:
    market = _market_state(bundle, indices, window)
    expert_history = np.empty(
        (len(indices), window, len(EXPERT_NAMES)), dtype=np.float32
    )
    for row, target in enumerate(indices):
        history = bundle.returns[int(target) - window : int(target)].astype(np.float64)
        past_weights = expert_path[int(target) - window : int(target)]
        # Realized returns of each causal expert rule at the corresponding
        # historical date, rather than a hindsight portfolio held constant.
        expert_history[row] = np.einsum("wa,wka->wk", history, past_weights)
    return np.concatenate([market, expert_history], axis=2).astype(np.float32)


def _profile_cache(dataset: str, split: str, window: int, horizon: int) -> Path:
    return CACHE_ROOT / f"{dataset}_{split}_w{window}_h{horizon}_paper.npz"


def _expert_bank(
    bundle: MarketDataBundle,
    dataset: str,
    split: str,
    market: MarketWindowDataset,
) -> np.ndarray:
    path = _profile_cache(dataset, split, market.window, market.horizon)
    data_hash = bundle_fingerprint(bundle)
    code_hash = source_fingerprint()
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            if (
                str(payload.get("data_sha256", "")) == data_hash
                and str(payload.get("source_sha256", "")) == code_hash
                and len(payload["experts"]) > int(market.indices[-1])
            ):
                return payload["experts"].astype(np.float32)
    values = [
        generate_expert_weights(
            bundle.returns,
            np.arange(int(market.indices[-1]) + 1, dtype=np.int64),
            expert_parameters(name),
        )
        for name in EXPERT_NAMES
    ]
    experts = np.stack(values, axis=1).astype(np.float32)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        indices=market.indices,
        experts=experts,
        data_sha256=data_hash,
        source_sha256=code_hash,
    )
    return experts


def _candidate_turnover(
    bundle: MarketDataBundle,
    market: MarketWindowDataset,
    experts: np.ndarray,
    expert_path: np.ndarray | None = None,
) -> np.ndarray:
    n, k, assets = experts.shape
    output = np.empty((n, k), dtype=np.float64)
    previous = np.full((k, assets), 1.0 / assets, dtype=np.float64)
    previous_target: int | None = None
    first = int(market.indices[0])
    if expert_path is not None and first > 0:
        previous = expert_path[first - 1].astype(np.float64)
        drifted = previous * (1.0 + bundle.returns[first - 1][None, :])
        previous = drifted / drifted.sum(axis=1, keepdims=True)
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
    """Compute Eq. (11) using the exact component definitions in B.3."""
    gross_path = future_path.astype(np.float64)
    turnover_path = turnover.astype(np.float64)
    if turnover_path.ndim == 2:
        turnover_path = turnover_path[:, None, :]
    if turnover_path.shape[1] not in (1, gross_path.shape[1]):
        raise ValueError("turnover must contain one value or one value per horizon day")
    if np.any(gross_path <= -1.0) or not np.isfinite(gross_path).all():
        raise ValueError("gross expert returns must be finite and greater than -1")
    # Transaction costs enter as c*T in Eq. (11), not inside G, D, or M.
    wealth = np.cumprod(1.0 + gross_path, axis=1)
    growth = wealth[:, -1] - 1.0
    downside = np.sqrt(np.square(np.minimum(gross_path, 0.0)).mean(axis=1))
    peak = np.maximum.accumulate(
        np.concatenate([np.ones_like(wealth[:, :1]), wealth], axis=1), axis=1
    )[:, 1:]
    drawdown = np.max(1.0 - wealth / np.maximum(peak, 1e-12), axis=1)
    hhi = np.square(experts.astype(np.float64)).sum(axis=-1)
    if hhi.ndim == 3:
        hhi = hhi.mean(axis=1)
    total_turnover = np.broadcast_to(turnover_path, gross_path.shape).sum(axis=1)
    return (
        growth
        - parameters.downside_weight * downside
        - parameters.drawdown_weight * drawdown
        - COST * total_turnover
        - parameters.concentration_weight * hhi
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
    if len(market) == 0:
        raise ValueError(f"no split-contained windows available for {dataset}/{split}")
    # Candidate rules are recomputed on every future decision day.  This makes
    # the supervision match deployment, where the selected expert is a trading
    # rule whose feasible portfolio changes with the causal information set.
    daily_market = MarketWindowDataset(bundle, split, window, 1)
    expert_path = _expert_bank(bundle, dataset, split, daily_market)
    daily_experts = expert_path[daily_market.indices]
    positions = np.searchsorted(daily_market.indices, market.indices)
    if not np.array_equal(daily_market.indices[positions], market.indices):
        raise RuntimeError("label dates are not aligned with the daily expert path")
    experts = daily_experts[positions]
    if split == "test":
        # Test outcomes are read only by the portfolio simulator, never by targets.
        future_path = np.empty((len(market), 0, len(EXPERT_NAMES)), dtype=np.float32)
        utility = np.zeros((len(market), len(EXPERT_NAMES)), dtype=np.float32)
        label = np.full(len(market), -1, dtype=np.int64)
    else:
        daily_gross = np.einsum(
            "na,nka->nk", bundle.returns[daily_market.indices], daily_experts
        ).astype(np.float32)
        gross_windows = np.lib.stride_tricks.sliding_window_view(
            daily_gross, window_shape=horizon, axis=0
        ).transpose(0, 2, 1)
        future_path = gross_windows[positions].astype(np.float32)
        daily_turnover = _candidate_turnover(
            bundle, daily_market, daily_experts, expert_path
        )
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
        state=_state_with_expert_evidence(bundle, market.indices, window, expert_path),
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
    soft_target = torch.softmax(utility / parameters.utility_temperature, dim=1)
    cross_entropy = (
        -(soft_target * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
    )
    regret = (utility.max(dim=1).values - (gate * utility).sum(dim=1)).mean()
    total = cross_entropy + parameters.utility_weight * regret
    accuracy = (logits.argmax(dim=1) == target).float().mean()
    return total, {
        "loss": float(total.detach()),
        "cross_entropy": float(cross_entropy.detach()),
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


def _reference_expert(panel: RouterPanel) -> int:
    """Select Eq. (14)'s reference over the complete training period."""
    if panel.split != "train":
        raise ValueError("the reference expert must be selected on training data")
    scores = []
    for expert in range(len(EXPERT_NAMES)):
        gate = np.eye(len(EXPERT_NAMES), dtype=np.float32)[
            np.full(len(panel.market), expert, dtype=np.int64)
        ]
        scores.append(_portfolio_score(_metrics(panel, gate)[0]))
    return int(np.argmax(scores))


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
    mean, std = _standardize(train, validation_labels, validation)
    references = {panel.dataset: _reference_expert(panel) for panel in train}
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
                _portfolio_score(
                    _metrics(
                        panel, member_gate(probability, 0.0, references[panel.dataset])
                    )[0]
                )
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
    label_audit = [_label_audit(panel) for panel in train + validation_labels]
    save_json(
        {
            "dynamic_label_definition": (
                "argmax risk-adjusted utility of the daily recomputed expert rule "
                f"over the next {parameters.label_horizon} split-contained observations"
            ),
            "implementation_id": IMPLEMENTATION_ID,
            "score_definition": "Eq11; components Eqs20-26; raw scores without date-wise standardization",
            "expert_order": list(EXPERT_NAMES),
            "label_audit": label_audit,
        },
        output / "dynamic_label_audit.json",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "best_epoch": best_epoch,
            "parameters": asdict(parameters),
            "implementation_id": IMPLEMENTATION_ID,
            "source_sha256": source_fingerprint(),
            "datasets": list(datasets),
            "data_sha256": {
                name: bundle_fingerprint(bundle) for name, bundle in bundles.items()
            },
            "expert_order": EXPERT_NAMES,
            "state_mean": mean,
            "state_std": std,
            "reference_experts": references,
        },
        output / "router.pt",
    )
    save_json({"history": history}, output / "training_history.json")
    elapsed = float(time.perf_counter() - started_at)
    save_json(
        {
            "seed": seed,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "epochs_requested": parameters.epochs,
            "best_epoch": best_epoch,
            "implementation_id": IMPLEMENTATION_ID,
            "test_evaluated": False,
            "best_validation_portfolio_score": float(best_key[0]),
            "best_validation_dynamic_label_accuracy": float(best_key[1]),
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
                "implementation_id": IMPLEMENTATION_ID,
            }
        ]
    )


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
        choices=DATASETS,
        help="research panels used jointly for training",
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
                utility_temperature=args.utility_temperature,
            ),
            args.device,
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
