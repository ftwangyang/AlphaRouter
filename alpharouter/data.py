from __future__ import annotations

"""Chronological market data loading with train-only normalization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_NAMES = ["open", "close", "high", "low", "volume", "vwap", "turn", "chg"]


@dataclass
class MarketDataBundle:
    features: np.ndarray
    returns: np.ndarray
    dates: np.ndarray
    tickers: list[str]
    feature_names: list[str]
    split_indices: Dict[str, tuple[int, int]]
    mean: np.ndarray
    std: np.ndarray

    @property
    def num_assets(self) -> int:
        return self.features.shape[1]

    @property
    def num_features(self) -> int:
        return self.features.shape[2]


def _split_bounds(
    length: int, train_ratio: float, val_ratio: float
) -> Dict[str, tuple[int, int]]:
    train_end = max(2, int(length * train_ratio))
    val_end = max(train_end + 1, int(length * (train_ratio + val_ratio)))
    val_end = min(val_end, length - 1)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def _date_split_bounds(
    dates: np.ndarray,
    train_end: str,
    val_end: str,
    test_start: str | None = None,
    test_end: str | None = None,
) -> Dict[str, tuple[int, int]]:
    train_stop = int(np.searchsorted(dates, np.datetime64(train_end), side="left"))
    val_stop = int(np.searchsorted(dates, np.datetime64(val_end), side="left"))
    test_begin = (
        val_stop
        if test_start is None
        else int(np.searchsorted(dates, np.datetime64(test_start), side="left"))
    )
    test_stop = (
        len(dates)
        if test_end is None
        else int(np.searchsorted(dates, np.datetime64(test_end), side="left"))
    )
    if not (2 <= train_stop < val_stop <= test_begin < test_stop <= len(dates)):
        raise ValueError(
            "invalid date split: expected train_end < val_end <= test_start "
            f"< test_end inside the data range, got "
            f"{(train_stop, val_stop, test_begin, test_stop)}"
        )
    return {
        "train": (0, train_stop),
        "val": (train_stop, val_stop),
        "test": (test_begin, test_stop),
    }


def _fit_normalizer(
    features: np.ndarray, train_end: int
) -> tuple[np.ndarray, np.ndarray]:
    train = features[:train_end]
    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    return mean, np.where(std < 1e-6, 1.0, std)


def _apply_normalizer(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    normalized = (features - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def _load_npz_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"market panel not found: {path}. See the repository README for required data files."
        )
    payload = np.load(path, allow_pickle=False)
    features = payload["features"].astype(np.float64)
    returns = payload["returns"].astype(np.float32)
    dates = payload["dates"].astype("datetime64[D]")
    tickers = [str(item) for item in payload["tickers"].tolist()]
    feature_names = [
        str(item)
        for item in payload.get("feature_names", np.array(FEATURE_NAMES)).tolist()
    ]
    if features.shape[:2] != returns.shape or features.shape[0] != len(dates):
        raise ValueError(f"inconsistent arrays in {path}")
    if features.ndim != 3 or returns.ndim != 2 or returns.shape[1] != len(tickers):
        raise ValueError(f"invalid asset or feature dimensions in {path}")
    if not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError(f"returns in {path} must be finite and greater than -1")
    if np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise ValueError(f"dates in {path} must be strictly increasing")
    return features, returns, dates, tickers, feature_names


def _align_to_reference(
    raw_features: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    tickers: list[str],
    feature_names: list[str],
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align an extended panel to the units and overlap of a frozen panel."""
    ref_features, ref_returns, ref_dates, ref_tickers, ref_names = _load_npz_arrays(
        reference_path
    )
    if tickers != ref_tickers or feature_names != ref_names:
        raise ValueError(
            "normalization reference must use identical ticker and feature ordering"
        )
    common, current_index, reference_index = np.intersect1d(
        dates, ref_dates, assume_unique=True, return_indices=True
    )
    if len(common) == 0:
        raise ValueError(
            "normalization reference has no dates in common with the panel"
        )
    close_index = feature_names.index("close")
    price_indices = [
        feature_names.index(name) for name in ("open", "close", "high", "low", "vwap")
    ]
    scale = ref_features[reference_index[-1], :, close_index] / np.clip(
        raw_features[current_index[-1], :, close_index], 1e-12, None
    )
    aligned = raw_features.copy()
    aligned[:, :, price_indices] *= scale[None, :, None]
    aligned[current_index] = ref_features[reference_index]
    aligned_returns = returns.copy()
    aligned_returns[current_index] = ref_returns[reference_index]
    return aligned, aligned_returns, ref_features, ref_dates


def load_npz_market(
    path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    train_end: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    normalization_reference: str | Path | None = None,
) -> MarketDataBundle:
    raw_features, returns, dates, tickers, feature_names = _load_npz_arrays(Path(path))
    if (train_end is None) != (val_end is None):
        raise ValueError("train_end and val_end must be provided together")
    bounds = (
        _date_split_bounds(dates, train_end, val_end, test_start, test_end)
        if train_end is not None and val_end is not None
        else _split_bounds(len(dates), train_ratio, val_ratio)
    )
    reference_features = None
    reference_dates = None
    if normalization_reference is not None:
        (
            raw_features,
            returns,
            reference_features,
            reference_dates,
        ) = _align_to_reference(
            raw_features,
            returns,
            dates,
            tickers,
            feature_names,
            Path(normalization_reference),
        )
    if reference_features is None:
        mean, std = _fit_normalizer(raw_features, bounds["train"][1])
    else:
        reference_train_end = (
            int(np.searchsorted(reference_dates, np.datetime64(train_end), side="left"))
            if train_end is not None
            else max(2, int(len(reference_dates) * train_ratio))
        )
        mean, std = _fit_normalizer(reference_features, reference_train_end)
    return MarketDataBundle(
        features=_apply_normalizer(raw_features, mean, std),
        returns=returns,
        dates=dates,
        tickers=tickers,
        feature_names=feature_names,
        split_indices=bounds,
        mean=mean.reshape(-1),
        std=std.reshape(-1),
    )


def load_market_data(config: Dict) -> MarketDataBundle:
    data_config = config["data"]
    source = data_config.get("source", "npz")
    if source == "npz":
        return load_npz_market(
            data_config["path"],
            train_ratio=float(data_config.get("train_ratio", 0.70)),
            val_ratio=float(data_config.get("val_ratio", 0.15)),
            train_end=data_config.get("train_end"),
            val_end=data_config.get("val_end"),
            test_start=data_config.get("test_start"),
            test_end=data_config.get("test_end"),
            normalization_reference=data_config.get("normalization_reference"),
        )
    raise ValueError(f"unknown data source {source!r}; expected 'npz'")


class MarketWindowDataset(Dataset):
    """Chronological market windows with future returns and source indices."""

    def __init__(
        self,
        bundle: MarketDataBundle,
        split: str,
        window: int,
        horizon: int = 1,
    ) -> None:
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be positive")
        if split not in bundle.split_indices:
            raise KeyError(f"unknown split {split!r}")
        split_start, split_end = bundle.split_indices[split]
        first_target = max(window, split_start)
        last_target = split_end - horizon
        self.indices = np.arange(
            first_target, max(first_target, last_target + 1), dtype=np.int64
        )
        self.bundle = bundle
        self.window = int(window)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        target = int(self.indices[item])
        features = self.bundle.features[target - self.window : target]
        future = self.bundle.returns[target : target + self.horizon]
        return {
            "features": torch.from_numpy(features),
            "next_return": torch.from_numpy(future[0]),
            "future_returns": torch.from_numpy(future),
            "index": torch.tensor(target, dtype=torch.long),
        }

    @property
    def dates(self) -> np.ndarray:
        return self.bundle.dates[self.indices]

    @property
    def next_returns(self) -> np.ndarray:
        return self.bundle.returns[self.indices]
