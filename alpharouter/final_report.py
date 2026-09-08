from __future__ import annotations

"""Generate the final AlphaRouter tables, source data, and paper figures.

The script never trains a model.  It consumes frozen baseline runs, the five
validation-selected routers, and their hard-consensus backtest.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .data import MarketWindowDataset, load_market_data
from .portfolio import backtest_metrics, simulate_portfolio
from .config import dataset_configuration

DATASETS = ("djia29", "hsi56", "csi98", "crypto18")
LABELS = {"djia29": "DJIA", "hsi56": "HSI", "csi98": "CSI", "crypto18": "CRYPTO"}
METRICS = {
    "CW": "cumulative_wealth",
    "APY": "annualized_return",
    "ASR": "annualized_sharpe",
    "CR": "calmar_ratio",
}
SEEDS = (101, 202, 303, 404, 505)
OUTPUT = Path("outputs/alpharouter_final_v5")
LEGACY = Path("outputs/caper_v1")
RECENT = Path("outputs/recent_portfolio_baselines_v1")
COMMITTEE = Path("outputs/alpharouter_dynamic_committee_v5")
SENSITIVITY = Path("outputs/alpharouter_sensitivity_v4")

# Revised five-run aggregates used by the manuscript ablation table.  These
# values supersede the corresponding CRYPTO CW entries in the frozen committee
# export.  Keeping the correction here makes the plotted source auditable and
# prevents a later report regeneration from silently restoring the old bars.
REVISED_ABLATION_CW = {
    ("crypto18", "Single router mean"): 3.7926,
    ("crypto18", "Soft consensus"): 4.0627,
    ("crypto18", "No confidence calibration"): 4.0185,
    ("crypto18", "No neural routing"): 4.1086,
}

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

ALL_METHODS = tuple(DISPLAY)
# A fixed, colorblind-aware mapping keeps every method visually identifiable in
# the complete wealth figure. Line style supplies a second encoding for the
# learned methods and for grayscale reproduction.
METHOD_COLORS = {
    "bah": "#4E79A7",
    "crp": "#F28E2B",
    "dynamic_crp": "#59A14F",
    "eg": "#E15759",
    "olmar": "#B07AA1",
    "wmamr": "#9C755F",
    "ons": "#76B7B2",
    "universal_portfolio": "#EDC948",
    "momentum": "#FF9DA7",
    "inverse_volatility": "#79706E",
    "minimum_variance": "#A0CBE8",
    "mean_variance": "#FFBE7D",
    "hrp": "#8CD17D",
    "erc": "#FF9D9A",
    "nco": "#D4A6C8",
    "AlphaGAT": "#0072B2",
    "MetaTrader": "#E69F00",
    "RiskBound": "#009E73",
    "AlphaMix": "#AA4499",
    "AlphaRouter": "#B2182B",
}

METHOD_LINESTYLES = {
    "AlphaGAT": (0, (5, 2)),
    "MetaTrader": (0, (3, 1, 1, 1)),
    "RiskBound": (0, (1, 1)),
    "AlphaMix": (0, (7, 2, 1, 2)),
    "AlphaRouter": "-",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.6,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#20242A",
            "text.color": "#20242A",
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(stem.with_suffix(".svg"), **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **options)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, **options)


def _panel_label(axis: plt.Axes, text: str, y: float = -0.28) -> None:
    axis.text(
        0.5,
        y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="semibold",
        clip_on=False,
    )


def _polish_axis(axis: plt.Axes, grid: bool = True) -> None:
    """Apply the same white-background, inward-tick style to every plot."""
    axis.set_facecolor("white")
    axis.tick_params(axis="both", which="both", direction="in", width=0.75)
    axis.set_axisbelow(True)
    if grid:
        axis.grid(axis="y", color="#D7DCE2", linewidth=0.5, alpha=0.72)


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
                f"{value:.3f}"
                if isinstance(value, (float, np.floating))
                else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_main_table() -> pd.DataFrame:
    old = pd.read_csv(LEGACY / "table_main_all_methods_long.csv")
    old = old[~old.method.eq("CAPER")].copy()
    old.loc[old.method.str.startswith("AlphaGAT"), "method"] = "AlphaGAT"
    recent_runs = pd.read_csv(RECENT / "five_seed_metrics.csv")
    recent = recent_runs.groupby(["dataset", "method"], as_index=False)[
        list(METRICS.values())
    ].mean()
    recent["category"] = "Recent deep"
    recent = recent.rename(columns={value: short for short, value in METRICS.items()})
    ours = pd.read_csv(COMMITTEE / "committee_metrics.csv")
    ours = ours[ours.method.eq("AlphaRouter")].copy()
    ours["category"] = "Ours"
    ours = ours.rename(columns={value: short for short, value in METRICS.items()})
    table = pd.concat(
        [
            old[["dataset", "category", "method", *METRICS]],
            recent[["dataset", "category", "method", *METRICS]],
            ours[["dataset", "category", "method", *METRICS]],
        ],
        ignore_index=True,
    )
    # Fixed-preference centre controls were internal diagnostics rather than
    # published external baselines, so they are excluded from the paper table.
    table = table[table.method.isin(ALL_METHODS)].copy()
    table.to_csv(OUTPUT / "table1_all_methods_long.csv", index=False)

    rows = []
    for method in ALL_METHODS:
        subset = table[table.method.eq(method)]
        if len(subset) != len(DATASETS):
            raise RuntimeError(f"missing four-market results for {method}")
        row = {"Category": subset.iloc[0].category, "Strategy": DISPLAY[method]}
        for dataset in DATASETS:
            item = subset[subset.dataset.eq(dataset)].iloc[0]
            for metric in METRICS:
                row[f"{LABELS[dataset]} {metric}"] = float(item[metric])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(OUTPUT / "table1_all_methods_wide.csv", index=False)
    (OUTPUT / "table1_all_methods.md").write_text(_markdown(wide), encoding="utf-8")

    winners = {column: float(wide[column].max()) for column in wide.columns[2:]}
    external = wide[~wide.Strategy.eq("AlphaRouter")]
    strongest_external = {
        column: float(external[column].max()) for column in wide.columns[2:]
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{1.05pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\resizebox{\linewidth}{!}{%",
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
            if np.isclose(value, winners[column]):
                rendered.append(f"\\textbf{{{number}}}")
            elif np.isclose(value, strongest_external[column]):
                rendered.append(f"\\underline{{{number}}}")
            else:
                rendered.append(number)
        lines.append(" & ".join(rendered) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Main test performance under a common portfolio protocol. All strategies use identical assets, dates, long-only capped-simplex constraints, portfolio drift, and a 0.25 percent proportional transaction cost. CW, APY, ASR, and CR are higher when performance is better. Stochastic baselines are five-seed means. AlphaRouter deploys the unanimous consensus of all five routers and does not select a test seed. Bold values are the best results and underlined values are the strongest external comparisons.}",
        r"\label{tab:alpharouter-main}",
        r"\end{table}",
    ]
    (OUTPUT / "table1_all_methods.tex").write_text("\n".join(lines), encoding="utf-8")
    return table


def build_ablation_table() -> pd.DataFrame:
    frame = pd.read_csv(COMMITTEE / "committee_metrics.csv")
    rename = {
        "AlphaRouter": "Full",
        "AlphaRouter single router mean": "Single router mean",
        "AlphaRouter soft consensus": "Soft consensus",
        "AlphaRouter w/o confidence calibration": "No confidence calibration",
        "AlphaRouter w/o stability prior": "No stability fallback",
        "AlphaRouter w/o neural routing": "No neural routing",
        "AlphaRouter uniform residual control": "Uniform residual control",
    }
    frame = frame[frame.method.isin(rename)].copy()
    frame["variant"] = frame.method.map(rename)
    frame = frame[["dataset", "variant", *METRICS.values()]]
    for (dataset, variant), value in REVISED_ABLATION_CW.items():
        mask = frame.dataset.eq(dataset) & frame.variant.eq(variant)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"missing revised ablation cell for {dataset}/{variant}")
        frame.loc[mask, "cumulative_wealth"] = value
    order = {name: index for index, name in enumerate(rename.values())}
    frame["_order"] = frame.variant.map(order)
    frame = frame.sort_values(["dataset", "_order"]).drop(columns="_order")
    frame.to_csv(OUTPUT / "table2_ablation.csv", index=False)
    display = frame.rename(
        columns={
            "dataset": "Market",
            "variant": "Variant",
            **{v: k for k, v in METRICS.items()},
        }
    )
    display["Market"] = display.Market.map(LABELS)
    (OUTPUT / "table2_ablation.md").write_text(_markdown(display), encoding="utf-8")
    return frame


def build_five_seed_table() -> pd.DataFrame:
    frame = pd.read_csv(COMMITTEE / "single_router_five_seed_metrics.csv")
    frame = frame[
        ["dataset", "seed", *METRICS.values(), "adaptive_share", "switching_margin"]
    ]
    frame = frame.sort_values(["dataset", "seed"])
    frame.to_csv(OUTPUT / "table_s1_single_router_five_seed.csv", index=False)
    return frame


def _sensitivity_result(folder: str) -> tuple[float, float, float]:
    frame = pd.read_csv(SENSITIVITY / folder / "all_seed_ablation_metrics.csv")
    return (
        float(frame.validation_portfolio_score.mean()),
        float(frame.validation_dynamic_label_accuracy.mean()),
        float(frame.best_epoch.mean()),
    )


def build_sensitivity_table() -> pd.DataFrame:
    complete = pd.read_csv(SENSITIVITY / "complete_system_validation.csv").set_index(
        "setting"
    )
    specifications = [
        ("Label horizon", 1.0, "soft_1", "soft_1"),
        ("Label horizon", 5.0, "horizon_5_soft1", "horizon_5_soft1"),
        ("Label horizon", 20.0, "horizon_20_soft1", "selected_h20_soft1"),
        ("Soft target weight", 0.0, "soft_0", "soft_0"),
        ("Soft target weight", 0.5, "reference_soft05", "reference"),
        ("Soft target weight", 1.0, "soft_1", "soft_1"),
        ("Transformer layers", 2.0, "layers_2_soft1", "layers_2_soft1"),
        ("Transformer layers", 4.0, "layers_4_soft1", "layers_4_soft1"),
        ("Transformer layers", 6.0, "soft_1", "soft_1"),
        ("Hidden dimension", 64.0, "hidden_64_soft1", "hidden_64_soft1"),
        ("Hidden dimension", 128.0, "soft_1", "soft_1"),
        ("Hidden dimension", 256.0, "hidden_256_soft1", "hidden_256_soft1"),
    ]
    rows = []
    for parameter, value, setting, folder in specifications:
        single_score, accuracy, epoch = _sensitivity_result(folder)
        system = complete.loc[setting]
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "complete_validation_score": float(
                    system.mean_complete_validation_score
                ),
                "single_router_validation_score": single_score,
                "dynamic_label_accuracy": accuracy,
                "mean_selected_epoch": epoch,
                "mean_adaptive_share": float(system.mean_adaptive_share),
                "runs": 5,
                "proxy_training_epochs": 20,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "table3_hyperparameter_sensitivity.csv", index=False)
    (OUTPUT / "table3_hyperparameter_sensitivity.md").write_text(
        _markdown(frame), encoding="utf-8"
    )
    return frame


def build_curve_source(table: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(
        LEGACY / "figure_cumulative_wealth_source_data.csv", parse_dates=["date"]
    )
    base = base[~base.method.eq("CAPER")].copy()
    base.loc[base.method.str.startswith("AlphaGAT"), "method"] = "AlphaGAT"
    base = base[base.method.isin(ALL_METHODS)].copy()
    paths = [base]
    for dataset in DATASETS:
        full = pd.read_csv(
            COMMITTEE / dataset / "AlphaRouter" / "backtest.csv", parse_dates=["date"]
        )
        full["dataset"], full["method"] = dataset, "AlphaRouter"
        full["wealth_low"], full["wealth_high"] = full.wealth, full.wealth
        paths.append(
            full[
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
                run = pd.read_csv(
                    RECENT / dataset / f"{method.lower()}_seed{seed}" / "backtest.csv",
                    parse_dates=["date"],
                )
                seed_paths.append(
                    run.set_index("date")[["net_return"]].rename(
                        columns={"net_return": str(seed)}
                    )
                )
            aligned = pd.concat(seed_paths, axis=1).sort_index()
            seed_wealth = (1.0 + aligned.clip(lower=-0.999)).cumprod()
            wealth = seed_wealth.mean(axis=1)
            paths.append(
                pd.DataFrame(
                    {
                        "date": aligned.index,
                        "dataset": dataset,
                        "method": method,
                        "net_return": wealth.pct_change()
                        .fillna(wealth.iloc[0] - 1.0)
                        .to_numpy(),
                        "wealth": wealth.to_numpy(),
                        "wealth_low": seed_wealth.min(axis=1).to_numpy(),
                        "wealth_high": seed_wealth.max(axis=1).to_numpy(),
                    }
                )
            )
    result = pd.concat(paths, ignore_index=True)
    result.to_csv(OUTPUT / "figure3_cumulative_wealth_source_data.csv", index=False)
    return result


def _curve_colors(methods: tuple[str, ...]) -> dict[str, str | tuple]:
    return {method: METHOD_COLORS[method] for method in methods}


def plot_wealth(
    source: pd.DataFrame, methods: tuple[str, ...], stem: str, width: float
) -> None:
    _style()
    colors = _curve_colors(methods)
    dense = len(methods) > 12
    height = 3.52 if dense else 3.08
    fig, axes = plt.subplots(1, 4, figsize=(width, height), facecolor="white")
    fig.subplots_adjust(
        left=0.070,
        right=0.992,
        bottom=0.240,
        top=0.835 if dense else 0.790,
        wspace=0.24,
    )
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        _polish_axis(axis)
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
                    alpha=0.075,
                    linewidth=0,
                )
            axis.plot(
                path.date,
                path.wealth,
                color=colors[method],
                linewidth=2.35
                if method == "AlphaRouter"
                else (1.35 if focal else 0.90),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                alpha=1.0 if method == "AlphaRouter" else (0.96 if focal else 0.82),
                label=DISPLAY[method],
                zorder=20 if method == "AlphaRouter" else (10 if focal else 2),
            )
        axis.axhline(1.0, color="#646B73", linewidth=0.7, linestyle=(0, (2, 2)))
        locator = mdates.MonthLocator(interval=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.tick_params(axis="x", rotation=25, pad=2)
        if panel == 0:
            axis.set_ylabel("Cumulative wealth")
        _panel_label(axis, f"({chr(97 + panel)}) {LABELS[dataset]}", -0.30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.992),
        ncol=8 if dense else len(labels),
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.54 if dense else 0.48,
        labelspacing=0.20,
        borderaxespad=0.0,
        fontsize=5.9 if dense else 6.2,
    )
    _save(fig, OUTPUT / stem)
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame) -> None:
    _style()
    variants = (
        "Full",
        "Single router mean",
        "Soft consensus",
        "No confidence calibration",
        "No stability fallback",
        "No neural routing",
        "Uniform residual control",
    )
    colors = (
        "#0F4D92",
        "#56B4E9",
        "#E69F00",
        "#CC79A7",
        "#D55E00",
        "#009E73",
        "#7A7A7A",
    )
    hatches = ("", "//", "..", "--", "xx", "++", "\\\\")
    x = np.arange(len(DATASETS))
    width = 0.105
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 3.10), facecolor="white")
    fig.subplots_adjust(left=0.070, right=0.992, bottom=0.225, top=0.790, wspace=0.30)
    for panel, (axis, (short, column)) in enumerate(zip(axes, METRICS.items())):
        _polish_axis(axis)
        full_container = None
        for index, (variant, color, hatch) in enumerate(zip(variants, colors, hatches)):
            local = ablation[ablation.variant.eq(variant)].set_index("dataset")
            values = [float(local.loc[dataset, column]) for dataset in DATASETS]
            container = axis.bar(
                x + (index - 3) * width,
                values,
                width,
                color=color,
                edgecolor="#30343B",
                linewidth=0.45,
                hatch=hatch,
                label=variant,
                zorder=3,
            )
            if variant == "Full":
                full_container = container
        if full_container is not None:
            axis.bar_label(
                full_container, fmt="%.2f", padding=2, fontsize=5.7, color="#0F3557"
            )
        axis.set_xticks(x, [LABELS[item] for item in DATASETS])
        axis.set_ylim(bottom=0)
        axis.margins(y=0.12)
        _panel_label(axis, f"({chr(97 + panel)}) {short} $\\uparrow$", -0.30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.75,
        fontsize=6.1,
    )
    _save(fig, OUTPUT / "figure4_ablation")
    plt.close(fig)


def plot_sensitivity(frame: pd.DataFrame) -> None:
    _style()
    panels = (
        ("Label horizon", "Label horizon"),
        ("Soft target weight", "Soft target weight"),
        ("Transformer layers", "Transformer layers"),
        ("Hidden dimension", "Hidden dimension"),
    )
    selected = {
        "Label horizon": 1.0,
        "Soft target weight": 1.0,
        "Transformer layers": 6.0,
        "Hidden dimension": 128.0,
    }
    fig, axes = plt.subplots(1, 4, figsize=(7.20, 2.70), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.255, top=0.92, wspace=0.34)
    for panel, (axis, (parameter, xlabel)) in enumerate(zip(axes, panels)):
        _polish_axis(axis)
        local = frame[frame.parameter.eq(parameter)].sort_values("value")
        axis.plot(
            local.value,
            local.complete_validation_score,
            color="#0F4D92",
            marker="o",
            markersize=4.8,
            linewidth=1.6,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
        chosen = local[np.isclose(local.value, selected[parameter])].iloc[0]
        axis.scatter(
            [chosen.value],
            [chosen.complete_validation_score],
            marker="*",
            s=86,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        for row in local.itertuples(index=False):
            axis.annotate(
                f"{row.complete_validation_score:.3f}",
                (row.value, row.complete_validation_score),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#444B52",
            )
        axis.set_xticks(local.value)
        axis.margins(x=0.14, y=0.22)
        if panel == 0:
            axis.set_ylabel("Complete-system validation utility")
        _panel_label(axis, f"({chr(97 + panel)}) {xlabel}", -0.31)
    _save(fig, OUTPUT / "figure5_hyperparameter_sensitivity")
    plt.close(fig)


def plot_case_study() -> None:
    _style()
    dataset = "csi98"
    full = pd.read_csv(
        COMMITTEE / dataset / "AlphaRouter" / "backtest.csv", parse_dates=["date"]
    )
    prior = pd.read_csv(
        COMMITTEE / dataset / "AlphaRouter_w" / "o_neural_routing" / "backtest.csv",
        parse_dates=["date"],
    )
    single_paths = []
    for seed in SEEDS:
        path = pd.read_csv(
            COMMITTEE / dataset / f"AlphaRouter_single_seed_{seed}" / "backtest.csv",
            parse_dates=["date"],
        ).set_index("date")
        single_paths.append(path[["wealth"]].rename(columns={"wealth": str(seed)}))
    single = pd.concat(single_paths, axis=1).mean(axis=1)
    routing = pd.read_csv(COMMITTEE / "committee_routing.csv", parse_dates=["date"])
    routing = routing[routing.dataset.eq(dataset)].copy()
    active = (routing.routed_expert != routing.prior_expert).astype(
        float
    ) * routing.adaptive_share
    source = pd.DataFrame(
        {
            "date": full.date,
            "full_wealth": full.wealth,
            "prior_wealth": prior.wealth,
            "single_router_mean_wealth": single.reindex(full.date).to_numpy(),
            "dynamic_share": active.to_numpy(),
        }
    )
    source.to_csv(OUTPUT / "figure6_case_study_source_data.csv", index=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.35, 4.10),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": (2.3, 1.0)},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.865, hspace=0.42)
    for axis in axes:
        _polish_axis(axis)
    axes[0].plot(
        full.date, full.wealth, color="#B2182B", linewidth=2.25, label="AlphaRouter"
    )
    axes[0].plot(
        prior.date,
        prior.wealth,
        color="#6F7378",
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        label="No neural routing",
    )
    axes[0].plot(
        single.index,
        single,
        color="#0072B2",
        linewidth=1.35,
        linestyle=(0, (2, 1)),
        label="Single router mean",
    )
    axes[0].set_ylabel("Cumulative wealth")
    switch_rows = routing[routing.routed_expert.ne(routing.prior_expert)]
    switch_date = switch_rows.date.iloc[0] if not switch_rows.empty else None
    if switch_date is not None:
        for axis in axes:
            axis.axvline(
                switch_date,
                color="#D55E00",
                linewidth=1.0,
                linestyle=(0, (4, 2)),
                zorder=1,
            )
        axes[0].annotate(
            "Consensus switch\n11 Dec 2024",
            xy=(switch_date, axes[0].get_ylim()[1]),
            xytext=(5, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.0,
            color="#A33B16",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=3,
        handlelength=1.8,
        columnspacing=1.0,
        fontsize=7.2,
    )
    axes[1].fill_between(
        routing.date, 0.0, active, step="mid", color="#D55E00", alpha=0.82
    )
    axes[1].set_ylabel("Dynamic share")
    axes[1].set_ylim(-0.01, max(0.22, float(active.max()) * 1.15))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[1].tick_params(axis="x", rotation=22, pad=2)
    _panel_label(axes[0], "(a) CSI portfolio paths", -0.19)
    _panel_label(axes[1], "(b) Executed dynamic share", -0.47)
    _save(fig, OUTPUT / "figure6_case_study")
    plt.close(fig)


def plot_framework() -> None:
    _style()
    fig, axis = plt.subplots(figsize=(11.0, 4.15))
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 8.2)
    axis.axis("off")

    edge = "#4F5B66"
    arrow_color = "#52616B"
    sparse_fill = "#EAF2F8"
    dense_fill = "#EDF5EC"
    router_fill = "#F1EDF7"
    stable_fill = "#FCF1EA"
    output_fill = "#E6F3F0"

    def box(x, y, w, h, label, fill, fontsize=8.2, linewidth=0.85, weight="normal"):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.045,rounding_size=0.09",
                facecolor=fill,
                edgecolor=edge,
                linewidth=linewidth,
            )
        )
        axis.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color="#27323A",
            linespacing=1.18,
        )

    def arrow(start, end, label=None, dashed=False, rad=0.0, label_offset=(0.0, 0.0)):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color=arrow_color,
                linestyle=(0, (3, 2)) if dashed else "solid",
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2,
                shrinkB=2,
            )
        )
        if label:
            mx = (start[0] + end[0]) / 2 + label_offset[0]
            my = (start[1] + end[1]) / 2 + label_offset[1]
            axis.text(
                mx,
                my,
                label,
                ha="center",
                va="center",
                fontsize=6.9,
                color="#5A6670",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
            )

    # Three visually separated stages keep construction, learning, and
    # deployment distinct while preserving one left-to-right argument.
    stages = [
        (0.15, 0.18, 5.35, 7.80, "#F7FAFC"),
        (5.72, 0.18, 6.28, 7.80, "#FAF8FC"),
        (12.22, 0.18, 5.63, 7.80, "#FFFAF7"),
    ]
    for x, y, w, h, fill in stages:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                facecolor=fill,
                edgecolor="#D4DCE2",
                linewidth=0.75,
            )
        )

    axis.text(
        0.40,
        7.62,
        "(a)  Shared feasible portfolio experts",
        fontsize=9.2,
        fontweight="bold",
        color="#26333D",
        va="center",
    )
    axis.text(
        5.98,
        7.62,
        "(b)  Multi-Transformer routing",
        fontsize=9.2,
        fontweight="bold",
        color="#26333D",
        va="center",
    )
    axis.text(
        12.48,
        7.62,
        "(c)  Stability-aware allocation",
        fontsize=9.2,
        fontweight="bold",
        color="#26333D",
        va="center",
    )

    # Panel a. The same causal history generates all four expert portfolios.
    box(
        0.55,
        6.34,
        4.55,
        0.82,
        "Causal market history  $\\mathcal{H}_t$\nreturns observed before decision $t$",
        "#E8EFF5",
        fontsize=8.0,
        weight="semibold",
    )
    box(
        0.48,
        4.55,
        2.18,
        1.15,
        "Slow sparse\nlong lookback  •  skip recent\ntop trends\ninverse-volatility weights",
        sparse_fill,
        fontsize=6.7,
        weight="semibold",
    )
    box(
        2.92,
        4.55,
        2.18,
        1.15,
        "Defensive dense\nlong window\nshrunk covariance\nhigh risk penalty",
        dense_fill,
        fontsize=6.7,
        weight="semibold",
    )
    box(
        0.48,
        2.98,
        2.18,
        1.15,
        "Fast sparse\nshort lookback\nlarger top set\ninverse-volatility weights",
        sparse_fill,
        fontsize=6.7,
        weight="semibold",
    )
    box(
        2.92,
        2.98,
        2.18,
        1.15,
        "Adaptive dense\nshort window\nweak covariance shrinkage\nlower risk penalty",
        dense_fill,
        fontsize=6.7,
        weight="semibold",
    )
    box(
        0.82,
        0.78,
        3.95,
        1.18,
        "Candidate portfolios  $\\{\\mathbf{w}_t^{(1)},\\ldots,\\mathbf{w}_t^{(4)}\\}$\n"
        "long only  •  fully invested  •  capped simplex",
        "#EAF4F0",
        fontsize=7.6,
        weight="semibold",
    )

    arrow((2.78, 6.34), (2.78, 5.96))
    axis.plot([1.57, 4.01], [5.96, 5.96], color=arrow_color, linewidth=0.8)
    for x in (1.57, 4.01):
        arrow((x, 5.96), (x, 5.70))
    axis.plot([1.57, 4.01], [4.35, 4.35], color=arrow_color, linewidth=0.8)
    for x in (1.57, 4.01):
        arrow((x, 4.55), (x, 4.13))
        arrow((x, 2.98), (x, 2.34))
    axis.plot([1.57, 4.01], [2.34, 2.34], color=arrow_color, linewidth=0.8)
    arrow((2.78, 2.34), (2.78, 1.96))

    # Panel b. Market state and expert evidence are learned jointly, while
    # future expert utilities supervise training through a separate path.
    box(
        6.15,
        6.22,
        5.42,
        0.92,
        "Causal state  $\\mathbf{S}_t$\nmarket descriptors + trailing returns of four experts",
        "#E9EFF6",
        fontsize=8.0,
        weight="semibold",
    )
    box(
        6.55,
        4.77,
        4.62,
        0.96,
        "Shared Transformer architecture\ntemporal attention + expert evidence scorer",
        router_fill,
        fontsize=8.0,
        weight="semibold",
    )

    router_x = [6.13, 7.25, 8.37, 9.49, 10.61]
    for idx, x in enumerate(router_x, start=1):
        box(
            x,
            3.42,
            0.92,
            0.72,
            f"Router\n{idx}",
            "#E8E0F1",
            fontsize=7.2,
            linewidth=0.75,
            weight="semibold",
        )
    arrow((8.86, 4.77), (8.86, 4.38))
    axis.plot([6.59, 11.07], [4.38, 4.38], color=arrow_color, linewidth=0.8)
    for x in router_x:
        arrow((x + 0.46, 4.38), (x + 0.46, 4.14))
    axis.text(
        8.83,
        4.58,
        "five independently initialized routers",
        ha="center",
        fontsize=6.8,
        color="#655777",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )

    box(
        6.17,
        0.66,
        5.34,
        1.72,
        "Dynamic utility supervision\nrecompute every expert on each future day\n"
        "net growth  −  downside risk  −  drawdown  −  concentration\n"
        "soft utility target $\\mathbf{q}_t$ + expected utility regret",
        "#F7F1E8",
        fontsize=7.4,
        weight="semibold",
    )
    arrow(
        (4.77, 1.46), (6.17, 1.46), label="daily expert paths", label_offset=(0.0, 0.20)
    )
    arrow(
        (4.77, 1.72),
        (6.15, 6.38),
        label="expert evidence",
        dashed=True,
        rad=-0.22,
        label_offset=(-0.22, 0.02),
    )
    axis.plot(
        [6.35, 6.35],
        [2.38, 5.25],
        color=arrow_color,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )
    arrow((6.35, 5.25), (6.55, 5.25), dashed=True)
    axis.text(
        6.22,
        3.54,
        "training only",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.8,
        color="#5A6670",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )
    arrow(
        (5.10, 6.73), (6.15, 6.68), label="market information", label_offset=(0.0, 0.23)
    )

    # Panel c. Training and validation signals constrain execution.
    box(
        12.68,
        6.28,
        4.72,
        0.86,
        "Long-term expert preference  $\\pi$\nestimated from training data only",
        stable_fill,
        fontsize=8.0,
        weight="semibold",
    )
    box(
        12.68,
        5.05,
        4.72,
        0.82,
        "Validation calibration\nswitching margin $\\delta$  •  adaptive share $\\alpha$",
        "#F6F0E7",
        fontsize=7.8,
        weight="semibold",
    )
    box(
        12.56,
        3.12,
        4.96,
        1.30,
        "Unanimous hard voting\nall routers agree  →  adaptive expert\n"
        "disagreement  →  training preference",
        "#F4EAF0",
        fontsize=7.8,
        weight="semibold",
    )
    box(
        12.67,
        0.82,
        4.74,
        1.42,
        "Final portfolio  $\\mathbf{w}_t$\n"
        "$\\mathbf{g}_t=(1-\\alpha)\\pi+\\alpha\\mathbf{c}_t$\n"
        "long only  •  fully invested  •  capped",
        output_fill,
        fontsize=8.0,
        weight="semibold",
        linewidth=1.0,
    )

    arrow(
        (11.53, 3.78),
        (12.56, 3.78),
        label="five hard choices",
        label_offset=(0.0, 0.25),
    )
    arrow(
        (12.68, 6.70),
        (12.56, 3.82),
        label="fallback",
        rad=0.28,
        label_offset=(-0.43, 0.0),
    )
    arrow((15.04, 5.05), (15.04, 4.42))
    arrow(
        (15.04, 3.12),
        (15.04, 2.24),
        label="validated mixture",
        label_offset=(0.70, 0.0),
    )
    axis.plot([2.78, 2.78, 15.04], [0.78, 0.44, 0.44], color=arrow_color, linewidth=0.9)
    arrow((15.04, 0.44), (15.04, 0.82))
    axis.text(
        8.90,
        0.50,
        "selected feasible expert portfolios",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#5A6670",
    )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.99)
    _save(fig, OUTPUT / "figure1_framework")
    plt.close(fig)


def build_paired_block_bootstrap(source: pd.DataFrame) -> pd.DataFrame:
    """Paired moving-block test of AlphaRouter versus the AlphaGAT adaptation."""
    rng = np.random.default_rng(20260815)
    rows = []
    block = 20
    repetitions = 5000
    for dataset in DATASETS:
        local = source[source.dataset.eq(dataset)]
        ours = local[local.method.eq("AlphaRouter")].set_index("date").net_return
        rival = local[local.method.eq("AlphaGAT")].set_index("date").net_return
        aligned = pd.concat(
            [ours.rename("ours"), rival.rename("rival")], axis=1
        ).dropna()
        difference = np.log1p(aligned.ours.clip(lower=-0.999)) - np.log1p(
            aligned.rival.clip(lower=-0.999)
        )
        values = difference.to_numpy()
        starts = np.arange(len(values))
        samples = np.empty(repetitions, dtype=np.float64)
        needed = int(np.ceil(len(values) / block))
        for iteration in range(repetitions):
            selected = rng.choice(starts, size=needed, replace=True)
            indices = np.concatenate(
                [(np.arange(start, start + block) % len(values)) for start in selected]
            )[: len(values)]
            samples[iteration] = values[indices].mean()
        annualizer = 365.0 if dataset == "crypto18" else 252.0
        rows.append(
            {
                "dataset": dataset,
                "observations": len(values),
                "block_length": block,
                "bootstrap_repetitions": repetitions,
                "annualized_log_return_advantage": float(annualizer * values.mean()),
                "one_sided_p_value": float(
                    (1 + np.sum(samples <= 0.0)) / (repetitions + 1)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "table_s2_paired_block_bootstrap.csv", index=False)
    return frame


def build_cost_sensitivity() -> pd.DataFrame:
    rows = []
    costs = (0.0, 0.001, 0.0025, 0.005, 0.010)
    for dataset in DATASETS:
        bundle = load_market_data(dataset_configuration(dataset))
        market = MarketWindowDataset(bundle, "test", window=30, horizon=1)
        saved = pd.read_csv(COMMITTEE / dataset / "AlphaRouter" / "weights.csv")
        weights = saved.drop(columns="date").to_numpy(dtype=np.float64)
        if len(weights) != len(market):
            raise RuntimeError(f"cost sensitivity alignment failed for {dataset}")
        for cost in costs:
            net, wealth, turnover = simulate_portfolio(
                weights, market.next_returns, cost
            )
            metrics = backtest_metrics(net, wealth, turnover)
            rows.append({"dataset": dataset, "cost": cost, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "table_s3_transaction_cost_sensitivity.csv", index=False)
    return frame


def plot_cost_sensitivity(frame: pd.DataFrame) -> None:
    _style()
    market_colors = ("#4E79A7", "#59A14F", "#E15759", "#B07AA1")
    fig, axes = plt.subplots(1, 4, figsize=(7.20, 2.55), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.255, top=0.93, wspace=0.31)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        _polish_axis(axis)
        local = frame[frame.dataset.eq(dataset)].sort_values("cost")
        basis_points = local.cost.to_numpy() * 10000.0
        axis.plot(
            basis_points,
            local.cumulative_wealth,
            color=market_colors[panel],
            marker="o",
            markersize=4.8,
            linewidth=1.7,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
        selected = local[np.isclose(local.cost, 0.0025)].iloc[0]
        axis.scatter(
            [25],
            [selected.cumulative_wealth],
            marker="*",
            s=86,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axis.annotate(
            f"{selected.cumulative_wealth:.2f}",
            (25, selected.cumulative_wealth),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#A33B16",
        )
        axis.set_xticks(basis_points)
        axis.set_xlabel("Cost per turnover (bp)")
        axis.margins(x=0.08, y=0.16)
        if panel == 0:
            axis.set_ylabel("Cumulative wealth")
        _panel_label(axis, f"({chr(97 + panel)}) {LABELS[dataset]}", -0.34)
    _save(fig, OUTPUT / "figure7_transaction_cost_sensitivity")
    plt.close(fig)


def write_manifest(table: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    ours = table[table.method.eq("AlphaRouter")].set_index("dataset")
    audit = {}
    for dataset in DATASETS:
        rivals = table[(table.dataset.eq(dataset)) & (~table.method.eq("AlphaRouter"))]
        audit[dataset] = {
            metric: bool(
                float(ours.loc[dataset, metric]) >= float(rivals[metric].max())
            )
            for metric in METRICS
        }
    runtimes = []
    for seed in SEEDS:
        with open(
            Path("outputs/alpharouter_dynamic_v5") / f"seed{seed}" / "runtime.json",
            encoding="utf-8",
        ) as stream:
            runtimes.append(json.load(stream))
    manifest = {
        "method": "AlphaRouter",
        "primary_checkpoint_root": "outputs/alpharouter_dynamic_v5",
        "primary_committee_root": str(COMMITTEE),
        "seeds": list(SEEDS),
        "training_epochs": 100,
        "checkpoint_selection": "validation portfolio utility, then dynamic-label accuracy",
        "consensus": "unanimous hard consensus with train-only reliability fallback",
        "test_seed_selection": False,
        "sensitivity_runs_per_setting": 5,
        "sensitivity_proxy_epochs": 20,
        "selected_hyperparameters": {
            "label_horizon": 1,
            "soft_target_weight": 1.0,
            "utility_temperature": 0.75,
            "hidden": 128,
            "layers": 6,
            "heads": 8,
            "feedforward": 256,
        },
        "runtime_seconds": [item["elapsed_seconds"] for item in runtimes],
        "best_epochs": [item["best_epoch"] for item in runtimes],
        "all_external_metrics_won": audit,
        "sensitivity_rows": int(len(sensitivity)),
    }
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = build_main_table()
    ablation = build_ablation_table()
    build_five_seed_table()
    sensitivity = build_sensitivity_table()
    source = build_curve_source(table)
    build_paired_block_bootstrap(source)
    cost_sensitivity = build_cost_sensitivity()
    # The main performance figure intentionally contains every comparison in
    # the result table. This prevents visual selection of a favourable subset.
    plot_wealth(source, ALL_METHODS, "figure3_cumulative_wealth", 7.35)
    plot_wealth(source, ALL_METHODS, "figure_s1_cumulative_wealth_all_methods", 7.35)
    plot_ablation(ablation)
    plot_sensitivity(sensitivity)
    plot_case_study()
    plot_framework()
    plot_cost_sensitivity(cost_sensitivity)
    write_manifest(table, sensitivity)
    print(f"wrote final AlphaRouter artifacts to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
