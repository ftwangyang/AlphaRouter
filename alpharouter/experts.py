from __future__ import annotations

"""Four causal, capped-simplex portfolio experts used by AlphaRouter."""

from dataclasses import dataclass

import numpy as np

from .data import MarketDataBundle, MarketWindowDataset


@dataclass(frozen=True)
class SparseExpert:
    lookback: int = 60
    skip: int = 20
    top_fraction: float = 0.10
    rebalance: int = 20
    risk_normalize: bool = True

    def validate(self) -> None:
        if self.lookback < 20 or self.skip < 0 or self.rebalance < 1:
            raise ValueError("invalid sparse lookback, skip, or rebalance interval")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class DenseExpert:
    lookback: int = 60
    rebalance: int = 5
    risk_aversion: float = 8.0
    diagonal_shrinkage: float = 0.50
    mean_scale: float = 1.0
    iterations: int = 15

    def validate(self) -> None:
        if self.lookback < 20 or self.rebalance < 1 or self.iterations < 1:
            raise ValueError(
                "invalid dense lookback, rebalance interval, or iterations"
            )
        if self.risk_aversion <= 0.0 or self.mean_scale < 0.0:
            raise ValueError(
                "risk_aversion must be positive and mean_scale non-negative"
            )
        if not 0.0 <= self.diagonal_shrinkage <= 1.0:
            raise ValueError("diagonal_shrinkage must lie in [0, 1]")


@dataclass(frozen=True)
class ExpertParameters:
    sparse: SparseExpert = SparseExpert()
    dense: DenseExpert = DenseExpert()
    sparse_weight: float = 0.50
    cap_multiple: float = 4.0

    def validate(self, assets: int) -> None:
        self.sparse.validate()
        self.dense.validate()
        if not 0.0 <= self.sparse_weight <= 1.0:
            raise ValueError("sparse_weight must lie in [0, 1]")
        if self.cap_multiple < 1.0:
            raise ValueError("cap_multiple must be at least one")
        if min(1.0, self.cap_multiple / assets) * assets < 1.0 - 1e-10:
            raise ValueError("infeasible per-asset cap")


def expert_parameters(name: str) -> ExpertParameters:
    """Return one of the four strategy-named expert profiles."""
    profiles = {
        "slow_sparse": ExpertParameters(
            sparse=SparseExpert(lookback=60, skip=20, top_fraction=0.10, rebalance=20),
            sparse_weight=1.0,
        ),
        "defensive_dense": ExpertParameters(
            dense=DenseExpert(
                lookback=220,
                rebalance=10,
                risk_aversion=32.0,
                diagonal_shrinkage=0.75,
                mean_scale=0.50,
            ),
            sparse_weight=0.0,
        ),
        "fast_sparse": ExpertParameters(
            sparse=SparseExpert(lookback=20, skip=5, top_fraction=0.25, rebalance=20),
            sparse_weight=1.0,
        ),
        "adaptive_dense": ExpertParameters(
            dense=DenseExpert(
                lookback=60,
                rebalance=5,
                risk_aversion=8.0,
                diagonal_shrinkage=0.0,
                mean_scale=1.0,
            ),
            sparse_weight=0.0,
        ),
    }
    try:
        return profiles[name]
    except KeyError as error:
        raise KeyError(f"unknown expert profile {name!r}") from error


def capped_simplex_projection(
    values: np.ndarray, cap: float, iterations: int = 50
) -> np.ndarray:
    """Project values onto ``w >= 0, sum(w) = 1, w_i <= cap``."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim < 1 or vector.shape[-1] < 1:
        raise ValueError("values must have a non-empty asset axis")
    assets = vector.shape[-1]
    if not np.isfinite(vector).all() or not np.isfinite(cap):
        raise ValueError("values and cap must be finite")
    if cap <= 0.0 or cap * assets < 1.0 - 1e-12:
        raise ValueError("cap is infeasible for the number of assets")
    cap = min(float(cap), 1.0)
    flat = vector.reshape(-1, assets)
    low = np.min(flat - cap, axis=1)
    high = np.max(flat, axis=1)
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        projected = np.clip(flat - middle[:, None], 0.0, cap)
        too_large = projected.sum(axis=1) > 1.0
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    result = np.clip(flat - high[:, None], 0.0, cap)
    residual = 1.0 - result.sum(axis=1)
    for row in range(len(result)):
        if residual[row] > 1e-13:
            for index in np.flatnonzero(result[row] < cap - 1e-14):
                addition = min(residual[row], cap - result[row, index])
                result[row, index] += addition
                residual[row] -= addition
                if residual[row] <= 1e-13:
                    break
        elif residual[row] < -1e-13:
            for index in np.flatnonzero(result[row] > 1e-14):
                removal = min(-residual[row], result[row, index])
                result[row, index] -= removal
                residual[row] += removal
                if residual[row] >= -1e-13:
                    break
    return result.reshape(vector.shape)


def _drift(weight: np.ndarray, realized: np.ndarray) -> np.ndarray:
    value = weight * (1.0 + realized)
    total = float(value.sum())
    return value / total if total > 1e-12 else weight.copy()


def _repair(weight: np.ndarray, cap: float) -> np.ndarray:
    value = np.nan_to_num(
        np.asarray(weight, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    value = np.maximum(value, 0.0)
    if float(value.sum()) <= 1e-12:
        value = np.ones_like(value)
    value /= value.sum()
    return capped_simplex_projection(value, cap, iterations=25)


def _sparse_target(
    history: np.ndarray, parameters: SparseExpert, cap: float
) -> np.ndarray:
    cumulative = np.prod(1.0 + history, axis=0) - 1.0
    volatility = np.clip(history.std(axis=0, ddof=1), 1e-4, None)
    score = cumulative / volatility if parameters.risk_normalize else cumulative
    count = max(1, int(np.ceil(parameters.top_fraction * history.shape[1])))
    selected = np.argsort(score, kind="mergesort")[-count:]
    raw = np.zeros(history.shape[1], dtype=np.float64)
    raw[selected] = 1.0 / volatility[selected]
    return _repair(raw, cap)


def _dense_target(
    history: np.ndarray,
    previous: np.ndarray,
    parameters: DenseExpert,
    cap: float,
) -> np.ndarray:
    mean = history.mean(axis=0)
    covariance = np.cov(history, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    shrinkage = parameters.diagonal_shrinkage
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    scale = max(float(np.linalg.norm(covariance, ord=np.inf)), 1e-12)
    step = min(1.0 / (2.0 * parameters.risk_aversion * scale), 100.0)
    weight = previous.copy()
    for _ in range(parameters.iterations):
        gradient = (
            -parameters.mean_scale * mean
            + 2.0 * parameters.risk_aversion * covariance @ weight
        )
        weight = capped_simplex_projection(weight - step * gradient, cap, iterations=15)
    return _repair(weight, cap)


def generate_expert_weights(
    historical_returns: np.ndarray,
    target_indices: np.ndarray,
    parameters: ExpertParameters,
) -> np.ndarray:
    """Generate causal weights for strictly increasing target indices."""
    returns = np.asarray(historical_returns, dtype=np.float64)
    indices = np.asarray(target_indices, dtype=np.int64)
    if returns.ndim != 2 or indices.ndim != 1 or len(indices) == 0:
        raise ValueError("returns must be [time, assets] and indices non-empty")
    if (
        np.any(indices < 0)
        or np.any(indices >= len(returns))
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("target indices must be strictly increasing and in range")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")
    assets = returns.shape[1]
    parameters.validate(assets)
    cap = min(1.0, parameters.cap_multiple / float(assets))
    sparse = np.full(assets, 1.0 / assets)
    dense = sparse.copy()
    output = np.empty((len(indices), assets), dtype=np.float64)
    for row, target in enumerate(indices):
        if row % parameters.sparse.rebalance == 0:
            end = int(target) - parameters.sparse.skip
            history = returns[max(0, end - parameters.sparse.lookback) : end]
            if len(history) >= 20:
                sparse = _sparse_target(history, parameters.sparse, cap)
        if row % parameters.dense.rebalance == 0:
            end = int(target)
            history = returns[max(0, end - parameters.dense.lookback) : end]
            if len(history) >= 20:
                dense = _dense_target(history, dense, parameters.dense, cap)
        sparse = _repair(sparse, cap)
        dense = _repair(dense, cap)
        output[row] = _repair(
            parameters.sparse_weight * sparse
            + (1.0 - parameters.sparse_weight) * dense,
            cap,
        )
        realized = returns[int(target)]
        sparse = _drift(sparse, realized)
        dense = _drift(dense, realized)
    return output


def profile_weights(
    bundle: MarketDataBundle,
    market: MarketWindowDataset,
    split: str,
    parameters: ExpertParameters,
) -> np.ndarray:
    """Generate an expert path, resetting each test calendar year."""
    if split != "test":
        return generate_expert_weights(bundle.returns, market.indices, parameters)
    dates = bundle.dates[market.indices]
    years = dates.astype("datetime64[Y]")
    output = np.empty((len(market), bundle.num_assets), dtype=np.float64)
    for year in np.unique(years):
        keep = years == year
        output[keep] = generate_expert_weights(
            bundle.returns, market.indices[keep], parameters
        )
    return output
