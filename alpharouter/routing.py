"""Confidence-gated voting and reference blending, Eqs. (15)-(18)."""

from __future__ import annotations

import numpy as np

CONFIDENCE_THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
ADAPTIVE_SHARES = (0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200)


def confidence_votes(
    probability: np.ndarray, threshold: float, reference: int
) -> np.ndarray:
    """Return the per-member expert indices from Eqs. (15) and (16).

    Input shape is [members, dates, experts]. Every decision depends only on
    the current probability vector, with no previous-day switching state.
    """
    values = np.asarray(probability)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] < 2:
        raise ValueError("probability must have shape [members, dates, experts >= 2]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence threshold must lie in [0, 1]")
    if not 0 <= reference < values.shape[2]:
        raise ValueError("reference must be a valid expert index")
    top_two = np.sort(values, axis=-1)[..., -2:]
    margin = top_two[..., 1] - top_two[..., 0]
    return np.where(margin >= threshold, values.argmax(axis=-1), reference)


def majority_gate(
    probability: np.ndarray, threshold: float, reference: int
) -> np.ndarray:
    """Aggregate calibrated votes by Eq. (17), including its tie rule.

    Eq. (17) selects the largest vote count (plurality), without requiring an
    absolute majority. Only tied vote winners are compared by mean raw
    probability. An exact remaining tie uses the lowest expert index.
    """
    values = np.asarray(probability)
    decisions = confidence_votes(values, threshold, reference)
    counts = np.stack(
        [(decisions == expert).sum(axis=0) for expert in range(values.shape[2])],
        axis=1,
    )
    tied = counts == counts.max(axis=1, keepdims=True)
    scores = np.where(tied, values.mean(axis=0), -np.inf)
    choice = scores.argmax(axis=1)
    return np.eye(values.shape[2], dtype=np.float32)[choice]


def member_gate(
    probability: np.ndarray, threshold: float, reference: int
) -> np.ndarray:
    """Apply the same confidence rule to a single router."""
    return majority_gate(np.asarray(probability)[None, ...], threshold, reference)


def blend_reference(
    reference: np.ndarray, routed_gate: np.ndarray, adaptive_share: float
) -> np.ndarray:
    """Construct the convex expert gate in Eq. (18)."""
    if not np.isfinite(adaptive_share) or not 0.0 <= adaptive_share <= 1.0:
        raise ValueError("adaptive_share must lie in [0, 1]")
    return (1.0 - adaptive_share) * np.asarray(
        reference, dtype=np.float32
    ) + adaptive_share * np.asarray(routed_gate, dtype=np.float32)
