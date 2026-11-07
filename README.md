# AlphaRouter

Transformer-based routing among four constrained portfolio experts, using confidence-gated committee voting and a training-selected reference expert.

## Installation

Requires Python 3.9+.

```bash
python -m pip install -e .
```

## Data

Place the four market panels (DJIA29, HSI56, CSI98, CRYPTO18) and their normalization-reference NPZ files in `data/`. Expected filenames and chronological splits are defined in [config.py](alpharouter/config.py).

Each panel contains `features`, `returns`, `dates`, and `tickers`; the loader and feature-order requirements are in [data.py](alpharouter/data.py). Datasets and pretrained checkpoints are not included.

## Usage

Run from the repository root. Train five routers with validation-based checkpoint selection:

```bash
python -m alpharouter.experiment --seeds 101 202 303 404 505 --epochs 100 --device auto
```

Calibrate the committee on validation data, then evaluate the frozen configuration:

```bash
python -m alpharouter.committee_experiment --seeds 101 202 303 404 505 --device auto
```

Checkpoints are saved in `outputs/alpharouter_members_paper/`; calibration records, routing decisions, weights, and metrics are saved in `outputs/alpharouter_committee_paper/`.

Keep source code and input data unchanged between training and evaluation. Use `--help` for additional options.

## Evaluation note

The 2024–2025 data were used during development, so evaluation is retrospective. Reported paper results have not been revalidated with this revision. See [implementation notes](docs/IMPLEMENTATION_RECONCILIATION.md) for details.
