from __future__ import annotations

"""Four causal, capped-simplex portfolio experts used by AlphaRouter."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

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
    iterations: int = 200

    def validate(self) -> None:
        if self.lookback < 20 or self.rebalance < 1 or self.iterations < 1:
            raise ValueError(
                "invalid dense lookback, rebalance interval, or iterations"
            )
        if self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive")
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
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("portfolio value must remain positive during drift")
    return value / total


def _repair(weight: np.ndarray, cap: float) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float64)
    if value.ndim != 1 or len(value) == 0:
        raise ValueError("portfolio weights must be a non-empty vector")
    if not np.isfinite(value).all():
        raise ValueError("portfolio weights must be finite")
    if float(value.min()) < -1e-10:
        raise ValueError("portfolio weights must be non-negative")
    value = np.maximum(value, 0.0)
    total = float(value.sum())
    if total <= 1e-12:
        raise ValueError("portfolio weights must have a positive sum")
    value /= total
    return capped_simplex_projection(value, cap, iterations=25)


def _sparse_target(
    history: np.ndarray, parameters: SparseExpert, cap: float
) -> np.ndarray:
    """Implement the Eq. (6) preference followed by the Eq. (7) projection."""
    cumulative = np.prod(1.0 + history, axis=0) - 1.0
    inverse_risk = 1.0 / (history.std(axis=0, ddof=1) + 1e-8)
    score = cumulative * inverse_risk if parameters.risk_normalize else cumulative
    count = max(1, int(np.ceil(parameters.top_fraction * history.shape[1])))
    selected = np.argsort(score, kind="mergesort")[-count:]
    preference = np.zeros(history.shape[1], dtype=np.float64)
    preference[selected] = inverse_risk[selected]
    return capped_simplex_projection(preference, cap)


def _dense_target(
    history: np.ndarray,
    previous: np.ndarray,
    parameters: DenseExpert,
    cap: float,
) -> np.ndarray:
    """Numerically solve the convex capped-simplex program in Eq. (8).

    SLSQP is supplied with the analytic gradient and the exact equality/bound
    constraints.  Failure is reported explicitly rather than replaced by a
    different portfolio construction rule.
    """
    mean = history.mean(axis=0)
    covariance = np.atleast_2d(np.cov(history, rowvar=False))
    diagonal = np.diag(np.diag(covariance))
    shrinkage = parameters.diagonal_shrinkage
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance = 0.5 * (covariance + covariance.T)
    initial = _repair(previous, cap)

    def objective(weight: np.ndarray) -> float:
        return float(
            -mean @ weight + parameters.risk_aversion * weight @ covariance @ weight
        )

    def gradient(weight: np.ndarray) -> np.ndarray:
        return -mean + 2.0 * parameters.risk_aversion * covariance @ weight

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, cap)] * len(initial),
        constraints={
            "type": "eq",
            "fun": lambda weight: float(weight.sum() - 1.0),
            "jac": lambda weight: np.ones_like(weight),
        },
        options={
            "maxiter": parameters.iterations,
            "ftol": 1e-12,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(
            "dense expert Eq. (8) optimization failed: "
            f"status={result.status}, message={result.message}"
        )
    weight = np.asarray(result.x, dtype=np.float64)
    feasible = (
        np.isfinite(weight).all()
        and abs(float(weight.sum()) - 1.0) <= 1e-8
        and float(weight.min()) >= -1e-10
        and float(weight.max()) <= cap + 1e-10
    )
    if not feasible:
        raise RuntimeError("dense expert Eq. (8) solver returned an infeasible point")
    return weight


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
    if not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError("returns must be finite and greater than -1")
    assets = returns.shape[1]
    parameters.validate(assets)
    cap = min(1.0, parameters.cap_multiple / float(assets))
    sparse = np.full(assets, 1.0 / assets)
    dense = sparse.copy()
    output = np.empty((len(indices), assets), dtype=np.float64)
    for row, target in enumerate(indices):
        if row % parameters.sparse.rebalance == 0:
            end = max(0, int(target) - parameters.sparse.skip)
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
    """Return a split slice from one causal expert path rooted at data start.

    Train, validation, and test therefore share identical state and rebalance
    phase at every overlapping date.  ``split`` remains in the public API for
    callers and cache keys, but does not alter expert dynamics.
    """
    del split
    last_target = int(market.indices[-1])
    full_indices = np.arange(last_target + 1, dtype=np.int64)
    full_path = generate_expert_weights(bundle.returns, full_indices, parameters)
    return full_path[market.indices]
