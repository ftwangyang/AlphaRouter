from __future__ import annotations

"""Dataset configuration used by the AlphaRouter experiments."""

from pathlib import Path
from typing import Any, Dict

DEFAULT_DATASETS = ("djia29", "hsi56", "csi98", "crypto18")

_PANELS = {
    "djia29": (
        "alphagat_djia29_frozen_extended_20260808.npz",
        "alphagat_djia29_public_2010_2023.npz",
    ),
    "hsi56": (
        "alphagat_hsi56_frozen_extended_20260808.npz",
        "alphagat_hsi56_public_2010_2023.npz",
    ),
    "csi98": (
        "alphagat_csi98_frozen_extended_20260808.npz",
        "alphagat_csi98_public_2010_2023.npz",
    ),
    "crypto18": (
        "alphagat_crypto18_frozen_extended_20260808.npz",
        "alphagat_crypto18_public_2018_2024.npz",
    ),
}


def dataset_configuration(
    dataset: str,
    data_dir: str | Path = "data",
) -> Dict[str, Any]:
    """Build a chronological data configuration for one research panel."""
    try:
        panel_name, reference_name = _PANELS[dataset]
    except KeyError as error:
        supported = ", ".join(DEFAULT_DATASETS)
        raise KeyError(
            f"unknown dataset {dataset!r}; choose one of: {supported}"
        ) from error

    root = Path(data_dir)
    return {
        "seed": 7,
        "data": {
            "source": "npz",
            "path": str(root / panel_name),
            "normalization_reference": str(root / reference_name),
            "train_end": "2023-01-01",
            "val_end": "2024-01-01",
            "test_start": "2024-01-01",
            "test_end": "2026-01-01",
        },
    }


def required_data_files(data_dir: str | Path = "data") -> tuple[Path, ...]:
    """Return the files required for a full four-market reproduction."""
    root = Path(data_dir)
    return tuple(root / name for pair in _PANELS.values() for name in pair)
