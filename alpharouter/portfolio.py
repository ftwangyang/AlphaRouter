from __future__ import annotations

"""Portfolio simulation, metrics, output, and reproducibility helpers."""

import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def simulate_portfolio(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    transaction_cost: float = 0.0025,
    initial_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate daily rebalancing with multiplicative proportional costs."""
    weights = np.asarray(weights, dtype=np.float64)
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    if weights.shape != asset_returns.shape:
        raise ValueError(
            f"weights {weights.shape} and returns {asset_returns.shape} must match"
        )
    if initial_weights is None:
        initial_weights = np.full(weights.shape[1], 1.0 / weights.shape[1])
    previous = np.asarray(initial_weights, dtype=np.float64)
    net_returns = np.empty(weights.shape[0], dtype=np.float64)
    turnovers = np.empty_like(net_returns)
    for index, (weight, next_return) in enumerate(zip(weights, asset_returns)):
        turnover = np.abs(weight - previous).sum()
        gross = float(np.dot(weight, next_return))
        net_returns[index] = (1.0 + gross) * (1.0 - transaction_cost * turnover) - 1.0
        turnovers[index] = turnover
        drifted = weight * (1.0 + next_return)
        total = drifted.sum()
        previous = drifted / total if total > 1e-12 else weight
    wealth = np.cumprod(1.0 + np.clip(net_returns, -0.999, None))
    return net_returns, wealth, turnovers


def backtest_metrics(
    net_returns: np.ndarray,
    wealth: np.ndarray | None = None,
    turnovers: np.ndarray | None = None,
    periods_per_year: int = 252,
    cvar_alpha: float = 0.05,
) -> Dict[str, float]:
    returns = np.asarray(net_returns, dtype=np.float64)
    if len(returns) == 0:
        raise ValueError("cannot calculate metrics for an empty backtest")
    if wealth is None:
        wealth = np.cumprod(1.0 + np.clip(returns, -0.999, None))
    wealth = np.asarray(wealth, dtype=np.float64)
    cumulative = float(wealth[-1])
    annualized_return = float(cumulative ** (periods_per_year / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * returns.mean() / (volatility + 1e-12))
    running_peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    maximum_drawdown = float(-(wealth / running_peak - 1.0).min())
    count = max(1, int(np.ceil(cvar_alpha * len(returns))))
    result = {
        "cumulative_wealth": cumulative,
        "annualized_return": annualized_return,
        "annualized_sharpe": sharpe,
        "calmar_ratio": float(annualized_return / (maximum_drawdown + 1e-12)),
        "max_drawdown": maximum_drawdown,
        "daily_cvar_05": float(-np.sort(returns)[:count].mean()),
        "daily_volatility": volatility,
    }
    if turnovers is not None:
        result["mean_turnover"] = float(np.asarray(turnovers).mean())
    return result


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def save_backtest_tables(
    dates: np.ndarray,
    weights: np.ndarray,
    returns: np.ndarray,
    wealth: np.ndarray,
    tickers: list[str],
    output_dir: str | Path,
) -> None:
    output = ensure_dir(output_dir)
    pd.DataFrame(weights, index=pd.to_datetime(dates), columns=tickers).to_csv(
        output / "weights.csv", index_label="date"
    )
    pd.DataFrame(
        {"net_return": returns, "wealth": wealth}, index=pd.to_datetime(dates)
    ).to_csv(output / "backtest.csv", index_label="date")
