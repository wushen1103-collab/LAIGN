from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (
    DATA_LINE_WIDTH,
    ERROR_CAPSIZE,
    ERROR_LINE_WIDTH,
    FOREST_LINE_WIDTH,
    FOREST_MARKER_AREA,
    FULL_WIDTH,
    LINE_MARKER_SIZE,
    METHOD_COLOR,
    METHOD_MARKER,
    PALETTE,
    export,
    finish_axis,
    legend_above,
    panel_label,
    set_style,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "derived"
MAIN = ROOT / "main"


def plot_fig3() -> None:
    set_style()
    temporal = pd.read_csv(DATA / "fig3_temporal_scatter.csv")
    absolute = pd.read_csv(DATA / "fig3_absolute.csv")
    effect = pd.read_csv(DATA / "fig3_effect.csv")

    fig = plt.figure(figsize=(FULL_WIDTH, 3.02))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 1.20, 1.12], wspace=0.54)

    ax = fig.add_subplot(gs[0, 0])
    seen = ~temporal["unseen_ccd"].astype(bool)
    ax.scatter(
        temporal.loc[seen, "graph_ap_fixed_fusion"],
        temporal.loc[seen, "graph_ap_laign"],
        s=12,
        color=PALETTE["laign"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
        label="Seen ligand code",
        zorder=3,
    )
    ax.scatter(
        temporal.loc[~seen, "graph_ap_fixed_fusion"],
        temporal.loc[~seen, "graph_ap_laign"],
        s=17,
        marker="D",
        color=PALETTE["fixed"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        label="Unseen ligand code",
        zorder=4,
    )
    ax.plot([0.40, 1.01], [0.40, 1.01], color=PALETTE["axis"], lw=0.47, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0.40, 1.015)
    ax.set_ylim(0.40, 1.015)
    ax.set_xlabel("Fixed fusion graph AP")
    ax.set_ylabel("LAIGN graph AP")
    ax.text(
        0.04,
        0.05,
        "$N=70$\n$\\Delta=+0.0206$\n95% CI [0.0051, 0.0390]",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    legend_above(ax, ncol=2, y=1.02)
    ax.tick_params(labelsize=7.2)
    finish_axis(ax)
    panel_label(ax, "(a)", x=-0.22)

    ax = fig.add_subplot(gs[0, 1])
    conditions = ["All", "Unseen CCD", "Unseen scaffold", "Tanimoto < 0.5", "Joint OOD"]
    condition_labels = ["All", "Unseen ligand code", "Unseen scaffold", "Tanimoto < 0.5", "Joint OOD"]
    methods = ["LAIGN", "Fixed fusion", "Distance"]
    y_base = np.arange(len(conditions))[::-1]
    offsets = {"LAIGN": 0.18, "Fixed fusion": 0.0, "Distance": -0.18}
    for method in methods:
        sub = absolute.loc[absolute["method"].eq(method)].set_index("condition").loc[conditions].reset_index()
        estimate = sub["estimate"].to_numpy()
        err = np.vstack([estimate - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - estimate])
        marker_face = PALETTE["white"] if method == "Fixed fusion" else METHOD_COLOR[method]
        marker_edge = METHOD_COLOR[method] if method == "Fixed fusion" else PALETTE["axis"]
        ax.errorbar(
            estimate,
            y_base + offsets[method],
            xerr=err,
            color=METHOD_COLOR[method],
            marker="o",
            ms=LINE_MARKER_SIZE,
            lw=0,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=ERROR_CAPSIZE,
            markerfacecolor=marker_face,
            markeredgecolor=marker_edge,
            markeredgewidth=0.28,
            label=method,
            zorder=3,
        )
    n_lookup = absolute.groupby("condition")["n"].first()
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"{label}\n$n={int(n_lookup[c])}$" for c, label in zip(conditions, condition_labels)], fontsize=7.2)
    ax.set_xlim(0.62, 1.005)
    ax.set_xlabel("Graph AP")
    ax.set_ylim(-0.55, len(conditions) - 0.45)
    legend_above(ax, ncol=3, y=1.02)
    ax.tick_params(axis="x", labelsize=7.2)
    finish_axis(ax, "x")
    panel_label(ax, "(b)", x=-0.16)

    ax = fig.add_subplot(gs[0, 2])
    keys = [
        "Descriptor available",
        "Unseen CCD",
        "Unseen scaffold",
        "Tanimoto < 0.5",
        "Joint OOD",
        "Temporal Tanimoto < 0.5",
    ]
    labels = ["Descriptor\navailable", "Unseen\nligand code", "Unseen\nscaffold", "Tanimoto\n< 0.5", "Joint OOD", "Post-freeze\nTanimoto < 0.5"]
    sub = effect.set_index("condition").loc[keys].reset_index()
    y = np.arange(len(sub))[::-1]
    for i, row in sub.iterrows():
        yy = y[i]
        face = PALETTE["white"] if row["condition"].startswith("Temporal") else PALETTE["laign"]
        ax.plot([row["ci_low"], row["ci_high"]], [yy, yy], color=PALETTE["laign"], lw=FOREST_LINE_WIDTH, zorder=2)
        ax.scatter(
            row["estimate"],
            yy,
            s=FOREST_MARKER_AREA,
            marker="o",
            facecolor=face,
            edgecolor=PALETTE["laign"],
            linewidth=0.35,
            zorder=3,
        )
        ax.text(0.094, yy, f"$n={int(row['n'])}$", va="center", ha="right", fontsize=6.5)
    ax.axvline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(-0.012, 0.098)
    ax.set_xlabel("Paired $\\Delta$ graph AP\n(LAIGN - fixed fusion)")
    finish_axis(ax, "x")
    ax.tick_params(axis="x", labelsize=7.2)
    panel_label(ax, "(c)", x=-0.18)

    fig.subplots_adjust(left=0.075, right=0.987, top=0.93, bottom=0.20)
    export(fig, MAIN / "Fig3_temporal_chemical_ood")


def horizontal_interval(ax, frame: pd.DataFrame, label_col: str, xlim: tuple[float, float], xlabel: str) -> None:
    y = np.arange(len(frame))[::-1]
    for i, row in frame.iterrows():
        label = row[label_col]
        color = METHOD_COLOR.get(label, PALETTE["laign"])
        ax.plot([row["ci_low"], row["ci_high"]], [y[i], y[i]], color=color, lw=FOREST_LINE_WIDTH)
        ax.scatter(
            row["estimate"],
            y[i],
            s=FOREST_MARKER_AREA,
            marker="o",
            color=color,
            edgecolor=PALETTE["axis"],
            linewidth=0.22,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame[label_col])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    finish_axis(ax, "x")


def plot_fig4() -> None:
    set_style()
    auprc = pd.read_csv(DATA / "fig4_auprc.csv")
    effect = pd.read_csv(DATA / "fig4_effect.csv")
    topk = pd.read_csv(DATA / "fig4_topk.csv")
    method_order = ["Distance", "Fixed fusion", "LAIGN", "Raw interaction", "ProLIF", "PLIP"]
    effect_order = ["Distance", "Fixed fusion", "Raw interaction", "ProLIF", "PLIP"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FULL_WIDTH - 0.15, 3.78),
        gridspec_kw={"wspace": 0.44, "hspace": 0.52},
    )

    ax = axes[0, 0]
    sub = auprc.set_index("method").loc[method_order].reset_index()
    horizontal_interval(ax, sub, "method", (0.32, 0.65), "Site AUPRC")
    prevalence = float(sub["prevalence"].iloc[0])
    ax.axvline(prevalence, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.text(prevalence + 0.004, 0.965, "prevalence", transform=ax.get_xaxis_transform(), fontsize=5.7, ha="left", va="top")
    panel_label(ax, "(a)", x=-0.21)

    ax = axes[0, 1]
    sub = effect.set_index("comparator").loc[effect_order].reset_index()
    horizontal_interval(ax, sub, "comparator", (-0.12, 0.13), "Paired $\\Delta$ AUPRC\n(LAIGN - comparator)")
    ax.axvline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    panel_label(ax, "(b)", x=-0.21)

    for ax, k, panel in [(axes[1, 0], 5, "(c)"), (axes[1, 1], 10, "(d)")]:
        sub = topk.loc[topk["k"].eq(k)].set_index("method").loc[method_order].reset_index()
        horizontal_interval(ax, sub, "method", (0.10, 0.68), f"Disruptive-site recall@{k}")
        panel_label(ax, panel, x=-0.21)

    fig.subplots_adjust(left=0.12, right=0.985, top=0.96, bottom=0.14)
    export(fig, MAIN / "Fig4_platinum_validation")


def main() -> None:
    plot_fig3()
    plot_fig4()
    print("Generated Fig. 3 and Fig. 4")


if __name__ == "__main__":
    main()
