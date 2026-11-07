"""Identity and data provenance for the paper-equation implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

IMPLEMENTATION_ID = "alpharouter-paper-equations-v1"


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bundle_fingerprint(bundle) -> str:
    """Bind experiments to the loaded observations and split boundaries."""
    digest = hashlib.sha256()
    for array in (bundle.returns, bundle.features, bundle.dates):
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    digest.update(json.dumps(bundle.tickers).encode("utf-8"))
    digest.update(json.dumps(bundle.split_indices, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
