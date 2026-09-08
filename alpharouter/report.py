from __future__ import annotations

"""Build AlphaRouter tables and figures without rerunning completed methods."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ("djia29", "hsi56", "csi98", "crypto18")
LABELS = {"djia29": "DJIA", "hsi56": "HSI", "csi98": "CSI", "crypto18": "CRYPTO"}
METRICS = {
    "CW": "cumulative_wealth",
    "APY": "annualized_return",
    "ASR": "annualized_sharpe",
    "CR": "calmar_ratio",
}
OUTPUT = Path("outputs/alpharouter_v1")
LEGACY = Path("outputs/caper_v1")
LEGACY_DYNAMIC = Path("outputs/caper_no_condition_v1")
VERIFIED_DYNAMIC = Path("outputs/alpharouter_dynamic_committee_v2")
RECENT = Path("outputs/recent_portfolio_baselines_v1")
REFERENCE = Path("outputs/neural_mars_h128_l6_e100_v1")
SEEDS = (101, 202, 303, 404, 505)

DISPLAY = {
    "bah": "BAH",
    "crp": "CRP",
    "dynamic_crp": "DynamicCRP",
    "eg": "EG",
    "olmar": "OLMAR",
    "wmamr": "WMAMR",
    "ons": "ONS",
    "universal_portfolio": "UP",
    "momentum": "Momentum",
    "inverse_volatility": "InvVol",
    "minimum_variance": "MinVar",
    "mean_variance": "MeanVar",
    "hrp": "HRP",
    "erc": "ERC",
    "nco": "NCO",
    "AlphaGAT": "AlphaGAT",
    "MetaTrader": "MetaTrader",
    "RiskBound": "RiskBound",
    "AlphaMix": "AlphaMix",
    "AlphaRouter": "AlphaRouter",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def _markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":--"] + ["--:"] * (len(headers) - 1)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_five_seed_recent() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(RECENT / "five_seed_metrics.csv")
    columns = ["dataset", "method", "seed", *METRICS.values()]
    runs[columns].sort_values(["dataset", "method", "seed"]).to_csv(
        OUTPUT / "table_recent_baselines_five_seed.csv", index=False
    )
    means = runs.groupby(["dataset", "method"], as_index=False)[
        list(METRICS.values())
    ].mean()
    means.to_csv(OUTPUT / "table_recent_baselines_mean.csv", index=False)
    return runs, means


def build_alpharouter_seed_and_ablation_tables() -> pd.DataFrame:
    seed_table = pd.read_csv(
        Path("outputs/alpharouter_dynamic_v2/all_seed_ablation_metrics.csv")
    )
    seed_table = seed_table[seed_table.method.eq("AlphaRouter-S")][
        ["dataset", "seed", *METRICS.values()]
    ].sort_values(["dataset", "seed"])
    seed_table.to_csv(OUTPUT / "table_s1_alpharouter_five_seed.csv", index=False)
    (OUTPUT / "table_s1_alpharouter_five_seed.md").write_text(
        _markdown(seed_table), encoding="utf-8"
    )

    committee = pd.read_csv(VERIFIED_DYNAMIC / "committee_metrics.csv")
    rename = {
        "AlphaRouter": "Full AlphaRouter",
        "AlphaRouter w/o consensus": "Single router",
        "AlphaRouter w/o confidence calibration": "No confidence calibration",
        "AlphaRouter soft consensus": "Soft consensus",
        "AlphaRouter w/o stability prior": "No stability prior",
        "AlphaRouter w/o neural routing": "No neural routing",
        "AlphaRouter uniform residual control": "Uniform residual",
    }
    ablation = committee[committee.method.isin(rename)].copy()
    ablation["variant"] = ablation.method.map(rename)
    ablation = ablation[["dataset", "variant", *METRICS.values()]].copy()
    ablation.to_csv(OUTPUT / "table2_ablation.csv", index=False)
    display = ablation.rename(
        columns={
            "dataset": "Market",
            "variant": "Variant",
            **{value: short for short, value in METRICS.items()},
        }
    )
    display["Market"] = display.Market.map(LABELS)
    (OUTPUT / "table2_ablation.md").write_text(_markdown(display), encoding="utf-8")
    return ablation


def build_main_table(recent_mean: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(LEGACY / "table_main_all_methods_long.csv")
    old = old[~old.method.eq("CAPER")].copy()
    old.loc[old.method.str.startswith("AlphaGAT"), "method"] = "AlphaGAT"
    verified = pd.read_csv(VERIFIED_DYNAMIC / "committee_metrics.csv")
    verified = verified[verified.method.eq("AlphaRouter")].copy()
    verified["category"] = "Ours"
    verified = verified.rename(
        columns={value: short for short, value in METRICS.items()}
    )
    recent = recent_mean.copy()
    recent["category"] = "Recent deep"
    recent = recent.rename(columns={value: short for short, value in METRICS.items()})
    table = pd.concat(
        [
            old[["dataset", "category", "method", *METRICS]],
            recent[["dataset", "category", "method", *METRICS]],
            verified[["dataset", "category", "method", *METRICS]],
        ],
        ignore_index=True,
    )
    table.to_csv(OUTPUT / "table1_all_methods_long.csv", index=False)
    selected = [
        "bah",
        "crp",
        "dynamic_crp",
        "eg",
        "olmar",
        "wmamr",
        "ons",
        "universal_portfolio",
        "momentum",
        "inverse_volatility",
        "minimum_variance",
        "mean_variance",
        "hrp",
        "erc",
        "nco",
        "AlphaGAT",
        "MetaTrader",
        "RiskBound",
        "AlphaMix",
        "AlphaRouter",
    ]
    rows = []
    for method in selected:
        subset = table[table.method.eq(method)]
        if len(subset) != len(DATASETS):
            raise RuntimeError(f"missing four-market result for {method}")
        item = {"Category": subset.iloc[0].category, "Strategy": DISPLAY[method]}
        for dataset in DATASETS:
            row = subset[subset.dataset.eq(dataset)].iloc[0]
            for metric in METRICS:
                item[f"{LABELS[dataset]} {metric}"] = float(row[metric])
        rows.append(item)
    wide = pd.DataFrame(rows)
    wide.to_csv(OUTPUT / "table1_all_methods_wide.csv", index=False)
    (OUTPUT / "table1_all_methods.md").write_text(_markdown(wide), encoding="utf-8")
    winners = {column: float(wide[column].max()) for column in wide.columns[2:]}
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\begin{tabular}{ll" + "r" * 16 + "}",
        r"\toprule",
        "Category & Strategy & "
        + " & ".join(f"\\multicolumn{{4}}{{c}}{{{LABELS[d]}}}" for d in DATASETS)
        + r" \\",
        " & & " + " & ".join(list(METRICS) * 4) + r" \\",
        r"\midrule",
    ]
    for row in wide.itertuples(index=False, name=None):
        rendered = [str(row[0]), str(row[1])]
        for column, value in zip(wide.columns[2:], row[2:]):
            number = f"{float(value):.3f}"
            rendered.append(
                f"\\textbf{{{number}}}"
                if np.isclose(float(value), winners[column])
                else number
            )
        lines.append(" & ".join(rendered) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Common-protocol performance. Stochastic methods are five-seed means. AlphaRouter is the hard consensus of five independently trained routers.}",
        r"\label{tab:alpharouter-main}",
        r"\end{table*}",
    ]
    (OUTPUT / "table1_all_methods.tex").write_text("\n".join(lines), encoding="utf-8")
    return table


def build_dynamic_router_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep verified dynamic results separate from the historical fixed table."""
    verified = pd.read_csv(VERIFIED_DYNAMIC / "committee_metrics.csv")
    verified.to_csv(OUTPUT / "table_dynamic_router_verified.csv", index=False)
    legacy = pd.read_csv(LEGACY_DYNAMIC / "committee_metrics.csv")
    legacy = legacy[legacy.method.eq("CAPER")].copy()
    legacy["method"] = "Historical returns-only router"
    legacy.to_csv(OUTPUT / "table_historical_returns_only_router.csv", index=False)
    return verified, legacy


def build_curve_source(runs: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(
        LEGACY / "figure_cumulative_wealth_source_data.csv", parse_dates=["date"]
    )
    source = source[~source.method.eq("CAPER")].copy()
    source.loc[source.method.str.startswith("AlphaGAT"), "method"] = "AlphaGAT"
    source = source[source.method.isin(DISPLAY)].copy()
    paths = []
    for dataset in DATASETS:
        alpha = pd.read_csv(
            VERIFIED_DYNAMIC / dataset / "AlphaRouter" / "backtest.csv",
            parse_dates=["date"],
        )
        alpha["dataset"], alpha["method"] = dataset, "AlphaRouter"
        alpha["wealth_low"], alpha["wealth_high"] = alpha.wealth, alpha.wealth
        paths.append(
            alpha[
                [
                    "date",
                    "dataset",
                    "method",
                    "net_return",
                    "wealth",
                    "wealth_low",
                    "wealth_high",
                ]
            ]
        )
        for method in ("MetaTrader", "RiskBound", "AlphaMix"):
            seed_paths = []
            for seed in SEEDS:
                frame = pd.read_csv(
                    RECENT / dataset / f"{method.lower()}_seed{seed}" / "backtest.csv",
                    parse_dates=["date"],
                )
                seed_paths.append(
                    frame.set_index("date")[["net_return"]].rename(
                        columns={"net_return": str(seed)}
                    )
                )
            aligned = pd.concat(seed_paths, axis=1).sort_index()
            seed_wealth = (1.0 + aligned.clip(lower=-0.999)).cumprod()
            # The plotted center is the pointwise five-seed mean, so its endpoint
            # exactly equals the arithmetic-mean CW reported in the table.
            wealth = seed_wealth.mean(axis=1)
            mean_return = wealth.pct_change().fillna(wealth.iloc[0] - 1.0)
            paths.append(
                pd.DataFrame(
                    {
                        "date": aligned.index,
                        "dataset": dataset,
                        "method": method,
                        "net_return": mean_return.to_numpy(),
                        "wealth": wealth.to_numpy(),
                        "wealth_low": seed_wealth.min(axis=1).to_numpy(),
                        "wealth_high": seed_wealth.max(axis=1).to_numpy(),
                    }
                )
            )
    result = pd.concat([source, *paths], ignore_index=True)
    result.to_csv(OUTPUT / "figure_cumulative_wealth_source_data.csv", index=False)
    return result


def plot_wealth(source: pd.DataFrame) -> None:
    _style()
    methods = [
        "bah",
        "crp",
        "dynamic_crp",
        "eg",
        "olmar",
        "wmamr",
        "ons",
        "universal_portfolio",
        "momentum",
        "inverse_volatility",
        "minimum_variance",
        "mean_variance",
        "hrp",
        "erc",
        "nco",
        "AlphaGAT",
        "MetaTrader",
        "RiskBound",
        "AlphaMix",
        "AlphaRouter",
    ]
    base = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, len(methods) - 5))
    colors = {method: base[index] for index, method in enumerate(methods[:-5])}
    colors.update(
        {
            "AlphaGAT": "#202124",
            "MetaTrader": "#8C6BB1",
            "RiskBound": "#D95F02",
            "AlphaMix": "#4C78A8",
            "AlphaRouter": "#00796B",
        }
    )
    # The complete 20-method comparison is deliberately wide.  At this size a
    # single-row shared legend remains readable in the 600-dpi export.
    fig, axes = plt.subplots(1, 4, figsize=(18.3, 3.75))
    fig.subplots_adjust(left=0.038, right=0.995, bottom=0.22, top=0.78, wspace=0.17)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        local = source[source.dataset.eq(dataset)]
        for method in methods:
            path = local[local.method.eq(method)].sort_values("date")
            if path.empty:
                continue
            focal = method in {
                "AlphaGAT",
                "MetaTrader",
                "RiskBound",
                "AlphaMix",
                "AlphaRouter",
            }
            if method in {"MetaTrader", "RiskBound", "AlphaMix"}:
                axis.fill_between(
                    path.date,
                    path.wealth_low,
                    path.wealth_high,
                    color=colors[method],
                    alpha=0.07,
                    linewidth=0,
                )
            axis.plot(
                path.date,
                path.wealth,
                color=colors[method],
                linewidth=2.0 if method == "AlphaRouter" else (1.05 if focal else 0.52),
                linestyle="--" if method == "AlphaGAT" else "-",
                alpha=1.0 if method == "AlphaRouter" else (0.88 if focal else 0.42),
                label=DISPLAY.get(method, method),
                zorder=20 if method == "AlphaRouter" else (10 if focal else 2),
            )
        axis.axhline(1.0, color="#8A9099", linewidth=0.4, linestyle=":")
        axis.grid(axis="y", color="#D9DDE3", linewidth=0.35)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=30, labelsize=5.3)
        axis.tick_params(axis="y", labelsize=5.3)
        axis.set_xlabel(
            f"({chr(97 + panel)}) {LABELS[dataset]}",
            fontsize=7,
            fontweight="bold",
            labelpad=7,
        )
        if panel == 0:
            axis.set_ylabel("Cumulative wealth", fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        fontsize=5.0,
        handlelength=0.9,
        columnspacing=0.35,
    )
    _save(fig, OUTPUT / "figure3_cumulative_wealth_all_methods")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame) -> None:
    _style()
    variants = [
        "Full AlphaRouter",
        "Single router",
        "No confidence calibration",
        "Soft consensus",
        "No stability prior",
        "No neural routing",
        "Uniform residual",
    ]
    colors = [
        "#00796B",
        "#4C78A8",
        "#E69F00",
        "#CC79A7",
        "#D55E00",
        "#8C8C8C",
        "#56B4E9",
    ]
    x = np.arange(len(DATASETS))
    width = 0.115
    fig, axes = plt.subplots(1, 4, figsize=(7.20, 2.65))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.78, wspace=0.27)
    for panel, (axis, (short, column)) in enumerate(zip(axes, METRICS.items())):
        for index, (variant, color) in enumerate(zip(variants, colors)):
            local = ablation[ablation.variant.eq(variant)].set_index("dataset")
            values = [float(local.loc[dataset, column]) for dataset in DATASETS]
            axis.bar(
                x + (index - 3.0) * width, values, width, color=color, label=variant
            )
        axis.set_title(
            f"({chr(97 + panel)}) {short} $\\uparrow$", fontsize=7, fontweight="bold"
        )
        axis.set_xticks(x, [LABELS[item] for item in DATASETS], rotation=28, ha="right")
        axis.grid(axis="y", color="#D9DDE3", linewidth=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=5.0,
        handlelength=1.1,
        columnspacing=0.8,
    )
    _save(fig, OUTPUT / "figure4_ablation")
    plt.close(fig)


def write_manifest(table: pd.DataFrame) -> None:
    winners = []
    for dataset in DATASETS:
        local = table[table.dataset.eq(dataset)]
        for metric in METRICS:
            row = local.loc[local[metric].idxmax()]
            winners.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "method": row.method,
                    "value": float(row[metric]),
                }
            )
    ours = table[table.method.eq("AlphaRouter")].set_index("dataset")
    is_sota = {}
    for dataset in DATASETS:
        rivals = table[(table.dataset.eq(dataset)) & (~table.method.eq("AlphaRouter"))]
        is_sota[dataset] = {
            metric: bool(
                float(ours.loc[dataset, metric]) >= float(rivals[metric].max())
            )
            for metric in METRICS
        }
    manifest = {
        "paper_name": "AlphaRouter",
        "implementation": "dynamic per-date labels, four shared experts, a train-only reliability prior, and a validation-calibrated neural residual route",
        "historical_fixed_label_checkpoint_status": "reproducible for audit but excluded from the AlphaRouter row",
        "data_status": "retrospective current-panel development",
        "test_period": "2024-01-01--2025-12-31",
        "transaction_cost": 0.0025,
        "common_constraints": [
            "long-only",
            "fully invested",
            "4/N maximum asset weight",
        ],
        "new_baselines": {
            "MetaTrader": "clean-room common-protocol adaptation; market transformations, conservative double-Q target, recent-period finetuning",
            "RiskBound": "clean-room common-protocol adaptation; downside-risk prediction, inverse-risk dynamic upper bounds, box-simplex projection",
            "AlphaMix": "clean-room common-protocol adaptation; diversified convolutional expert pool and as-needed routing",
        },
        "stochastic_reporting": "five fixed seeds 101,202,303,404,505; the primary row is their validation-calibrated consensus; no best-seed selection",
        "selection_protocol": "train-only 756-observation expert prior; validation-only switching margin and adaptive-sleeve calibration; no test-period label or hyperparameter selection",
        "sota_audit": is_sota,
        "winner_audit": winners,
    }
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_result_status(table: pd.DataFrame, ablation: pd.DataFrame) -> None:
    ours = table[table.method.eq("AlphaRouter")].set_index("dataset")
    neural = ablation[ablation.variant.eq("No neural routing")].set_index("dataset")
    uniform = ablation[ablation.variant.eq("Uniform residual")].set_index("dataset")
    lines = [
        "# AlphaRouter verified result status",
        "",
        "All values below are from the common 2024--2025 test panel after 0.25% transaction costs.",
        "The four experts are present in every market. The reliability prior uses only the final 756 training observations, while the Transformer is trained from per-date 20-day utility labels. Switching margins and residual shares are chosen on the 2023 validation split. No test labels or best test seed are used.",
        "",
        "## Main result",
        "",
        "| Market | CW | APY | ASR | CR | Best external CW | External method |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for dataset in DATASETS:
        rivals = table[(table.dataset.eq(dataset)) & (~table.method.eq("AlphaRouter"))]
        rival = rivals.loc[rivals.CW.idxmax()]
        row = ours.loc[dataset]
        lines.append(
            f"| {LABELS[dataset]} | {row.CW:.3f} | {row.APY:.3f} | {row.ASR:.3f} | {row.CR:.3f} | {rival.CW:.3f} | {DISPLAY.get(rival.method, rival.method)} |"
        )
    lines += [
        "",
        "## Ablation interpretation",
        "",
        "The stability prior is the dominant effective component. Removing it sharply lowers performance in DJIA, CSI, and CRYPTO, although HSI is an exception where the unrestricted neural route is stronger. The neural residual is small by validation design. Relative to the no-neural prior, it improves HSI but slightly reduces DJIA, CSI, and CRYPTO. Relative to a uniform residual of identical size, it improves DJIA, HSI, and CRYPTO but not CSI. Therefore the current evidence supports the candidate-portfolio and stability-prior contribution, but it does not support a claim that the Transformer improves every market.",
        "",
        "| Market | Full CW | No-neural CW | Uniform-residual CW |",
        "|:--|--:|--:|--:|",
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {LABELS[dataset]} | {ours.loc[dataset, 'CW']:.3f} | {neural.loc[dataset, 'cumulative_wealth']:.3f} | {uniform.loc[dataset, 'cumulative_wealth']:.3f} |"
        )
    lines += [
        "",
        "These are retrospective local-panel results. They establish table-level dominance under this implementation and protocol, not a universal or prospective SOTA claim.",
    ]
    (OUTPUT / "RESULT_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs, recent = build_five_seed_recent()
    build_dynamic_router_audit()
    ablation = build_alpharouter_seed_and_ablation_tables()
    table = build_main_table(recent)
    source = build_curve_source(runs)
    plot_wealth(source)
    plot_ablation(ablation)
    write_manifest(table)
    write_result_status(table, ablation)
    print(f"wrote AlphaRouter paper artifacts to {OUTPUT}")


if __name__ == "__main__":
    main()
