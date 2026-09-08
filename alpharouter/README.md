# AlphaRouter

AlphaRouter is the paper-facing name of the expert-routing method. It builds
the same four strategy-named, long-only, fully invested, capped experts in
every market. A risk-adjusted best-expert label is constructed separately for
every training date. Independently trained temporal Transformer members route
the current state, apply validation-selected switching calibration, and make
the final decision by hard consensus.

Historical fixed-market-label checkpoints remain reproducible in the legacy
modules but the dynamic committee evaluator refuses to load them.

Commands

```powershell
python -m alpharouter.experiment --seeds 101 202 303 404 505 --epochs 100
python -m alpharouter.committee_experiment
python -m alpharouter.report
```

The three recent baselines are trained with

```powershell
python -m recent_portfolio_baselines.experiment --datasets djia29 hsi56 csi98 crypto18 --methods MetaTrader RiskBound AlphaMix --seeds 101 202 303 404 505
```

All local comparisons use the same 2024-2025 test interval, 25 bp proportional
L1 transaction cost, long-only fully invested portfolios, and a 4/N per-asset
cap. The current panel is retrospective development evidence because it has
already been used during method development.
