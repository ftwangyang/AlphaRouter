from __future__ import annotations

"""Generate an honest audit figure for dynamic and historical router variants."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ("djia29", "hsi56", "csi98", "crypto18")
LABELS = {"djia29": "DJIA", "hsi56": "HSI", "csi98": "CSI", "crypto18": "CRYPTO"}
ROOT = Path("outputs/alpharouter_dynamic_audit_v2")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = {
        "AlphaGAT": Path(
            "outputs/alpharouter_v1/figure_cumulative_wealth_source_data.csv"
        ),
        "Historical returns-only router": Path("outputs/caper_no_condition_v1"),
        "Verified dynamic-label AlphaRouter": Path(
            "outputs/alpharouter_dynamic_committee_v2"
        ),
    }
    rows = []
    common = pd.read_csv(sources["AlphaGAT"], parse_dates=["date"])
    for dataset in DATASETS:
        alpha = common[
            (common.dataset == dataset) & common.method.str.startswith("AlphaGAT")
        ]
        rows.append(
            alpha.assign(display_method="AlphaGAT")[
                ["date", "dataset", "display_method", "wealth"]
            ]
        )
        for method in (
            "Historical returns-only router",
            "Verified dynamic-label AlphaRouter",
        ):
            folder = "CAPER" if method.startswith("Historical") else "AlphaRouter"
            path = pd.read_csv(
                sources[method] / dataset / folder / "backtest.csv",
                parse_dates=["date"],
            )
            rows.append(
                path.assign(dataset=dataset, display_method=method)[
                    ["date", "dataset", "display_method", "wealth"]
                ]
            )
    source = pd.concat(rows, ignore_index=True)
    source.to_csv(ROOT / "dynamic_router_wealth_source.csv", index=False)
    colors = {
        "AlphaGAT": "#333333",
        "Historical returns-only router": "#D55E00",
        "Verified dynamic-label AlphaRouter": "#0072B2",
    }
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 2.8))
    for index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        local = source[source.dataset == dataset]
        for method in colors:
            curve = local[local.display_method == method]
            axis.plot(
                curve.date,
                curve.wealth,
                label=method,
                color=colors[method],
                linewidth=1.8 if method.endswith("AlphaRouter") else 1.15,
            )
        axis.axhline(1.0, color="#999999", linestyle=":", linewidth=0.6)
        axis.grid(axis="y", alpha=0.22)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=28, labelsize=7)
        axis.set_xlabel(f"({chr(97 + index)}) {LABELS[dataset]}")
        if index == 0:
            axis.set_ylabel("Cumulative wealth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(
            ROOT / f"dynamic_router_audit.{suffix}", bbox_inches="tight", **kwargs
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
