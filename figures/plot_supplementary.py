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
    SEED_LINE_WIDTH,
    export,
    finish_axis,
    legend_above,
    panel_label,
    set_style,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "source_data"
DATA = RAW / "derived"
OUT = ROOT / "supplementary"


def fig_s1() -> None:
    set_style()
    components = pd.read_csv(DATA / "figs1_components.csv")
    route = pd.read_csv(DATA / "figs1_route.csv")
    shuffle = pd.read_csv(DATA / "figs1_shuffle.csv")
    calibration = pd.read_csv(DATA / "figs1_calibration.csv")
    order = ["Distance only", "Pair-MLP", "Scalar ViSNet", "Typed multi-scale", "LAIGN consensus"]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH - 0.06, 3.72), gridspec_kw={"wspace": 0.34, "hspace": 0.50})

    ax = axes[0, 0]
    wide = components.pivot(index="seed", columns="component", values="auprc").loc[:, order]
    x = np.arange(len(order))
    for _, row in wide.iterrows():
        ax.plot(x, row.to_numpy(), color=PALETTE["neutral"], lw=SEED_LINE_WIDTH, alpha=0.75, marker="o", ms=LINE_MARKER_SIZE)
    mean = wide.mean(axis=0).to_numpy()
    sd = wide.std(axis=0, ddof=1).to_numpy()
    ax.errorbar(x, mean, yerr=sd, color=PALETTE["laign"], lw=DATA_LINE_WIDTH, marker="o", ms=LINE_MARKER_SIZE, elinewidth=ERROR_LINE_WIDTH, capsize=ERROR_CAPSIZE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=22, ha="right")
    ax.set_ylim(0.66, 0.97)
    ax.set_ylabel("Pooled AUPRC")
    finish_axis(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    colors = [PALETTE["laign"], PALETTE["neutral"], PALETTE["distance"], PALETTE["fixed"]]
    y = np.arange(len(route))[::-1]
    for yy, value, color in zip(y, route["graph_ap"], colors):
        ax.plot([0.68, value], [yy, yy], color=PALETTE["grid"], lw=FOREST_LINE_WIDTH, zorder=1)
        ax.scatter(value, yy, s=FOREST_MARKER_AREA, marker="o", color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(route["variant"])
    ax.set_xlim(0.68, 0.98)
    ax.set_xlabel("Graph AP")
    for yy, (_, row) in zip(y, route.iterrows()):
        value = row["graph_ap"]
        if row["variant"] == "Full LAIGN":
            ax.text(value, yy - 0.16, f"{value:.4f}", ha="center", va="top", fontsize=5.8)
        else:
            ax.text(value + 0.004, yy, f"{value:.4f}", va="center", fontsize=5.8)
    ax.set_ylim(-0.45, len(route) - 0.45)
    finish_axis(ax, "x")
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    conditions = ["Distance only", "Aligned", "Within-complex shuffle"]
    x = np.arange(len(conditions))
    for _, run in shuffle.pivot(index="seed", columns="condition", values="auprc").loc[:, conditions].iterrows():
        ax.plot(x, run.to_numpy(), color=PALETTE["neutral"], lw=SEED_LINE_WIDTH, alpha=0.75)
        ax.scatter(x, run.to_numpy(), s=FOREST_MARKER_AREA, marker="o", color=PALETTE["neutral"], edgecolor="white", linewidth=0.3)
    means = shuffle.groupby("condition")["auprc"].mean().reindex(conditions)
    ax.plot(x, means.to_numpy(), color=PALETTE["laign"], lw=DATA_LINE_WIDTH, marker="o", ms=LINE_MARKER_SIZE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=17, ha="right")
    ax.set_ylim(0.64, 0.94)
    ax.set_ylabel("Pooled AUPRC")
    finish_axis(ax, "y")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    metrics = ["ECE-15", "NLL"]
    y = np.arange(2)[::-1]
    for yy, metric in zip(y, metrics):
        sub = calibration.loc[calibration["metric"].eq(metric)].set_index("stage")
        before = float(sub.loc["Before", "value"])
        after = float(sub.loc["After", "value"])
        ax.plot([after, before], [yy, yy], color=PALETTE["distance"], lw=DATA_LINE_WIDTH)
        ax.scatter(before, yy, s=FOREST_MARKER_AREA, marker="o", color=PALETTE["neutral"], edgecolor=PALETTE["axis"], linewidth=0.4, label="Before" if metric == metrics[0] else None)
        ax.scatter(after, yy, s=FOREST_MARKER_AREA, marker="o", color=PALETTE["laign"], edgecolor="white", linewidth=0.35, label="After" if metric == metrics[0] else None)
        ax.text(before + 0.001, yy + 0.13, f"{before:.4f}", fontsize=5.7, ha="center")
        ax.text(after, yy - 0.18, f"{after:.4f}", fontsize=5.7, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 0.036)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xlabel("Calibration error")
    legend_above(ax, ncol=2)
    finish_axis(ax, "x")
    panel_label(ax, "(d)")

    fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.15)
    export(fig, OUT / "FigS1_component_reliability")


def fig_s2() -> None:
    set_style()
    pose = pd.read_csv(DATA / "figs2_pose.csv")
    source_order = ["Native", "0.5 A + 5 deg", "1 A + 10 deg", "2 A + 20 deg"]
    display_order = ["Native", "0.5 Å + 5°", "1 Å + 10°", "2 Å + 20°"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.45), gridspec_kw={"width_ratios": [1.35, 0.85], "wspace": 0.34})

    ax = axes[0]
    x = np.arange(len(source_order))
    for method in ["LAIGN", "Fixed fusion"]:
        sub = pose.loc[pose["method"].eq(method)]
        grouped = sub.groupby("condition")["graph_ap"]
        means = grouped.mean().reindex(source_order)
        sd = grouped.std(ddof=1).fillna(0).reindex(source_order)
        ax.errorbar(x, means, yerr=sd, color=METHOD_COLOR[method], marker="o", ms=LINE_MARKER_SIZE, lw=DATA_LINE_WIDTH, elinewidth=ERROR_LINE_WIDTH, capsize=ERROR_CAPSIZE, label=method)
        for _, seed_rows in sub.loc[sub["perturb_seed"].ge(0)].groupby("perturb_seed"):
            series = seed_rows.set_index("condition")["graph_ap"].reindex(source_order[1:])
            ax.plot(x[1:], series, color=METHOD_COLOR[method], alpha=0.18, lw=SEED_LINE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(display_order, rotation=16, ha="right")
    ax.set_ylim(0.54, 0.98)
    ax.set_ylabel("Graph AP")
    legend_above(ax, ncol=2)
    finish_axis(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    perturbed = pose.loc[pose["condition"].ne("Native")]
    wide = perturbed.pivot(index=["condition", "perturb_seed"], columns="method", values="graph_ap").reset_index()
    wide["delta"] = wide["LAIGN"] - wide["Fixed fusion"]
    for i, condition in enumerate(source_order[1:]):
        values = wide.loc[wide["condition"].eq(condition), "delta"].to_numpy()
        ax.scatter(np.full(len(values), i) + np.linspace(-0.05, 0.05, len(values)), values, s=FOREST_MARKER_AREA, marker="o", color=PALETTE["neutral"], edgecolor="white", linewidth=0.3)
        ax.scatter(i, values.mean(), s=FOREST_MARKER_AREA, marker="o", color=PALETTE["laign"], edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(display_order[1:], rotation=16, ha="right")
    ax.set_ylabel("$\\Delta$ graph AP")
    ax.set_ylim(-0.005, 0.08)
    finish_axis(ax, "y")
    panel_label(ax, "(b)", x=-0.18)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.27)
    export(fig, OUT / "FigS2_pose_robustness")


def fig_s3() -> None:
    set_style()
    chemical = pd.read_csv(DATA / "figs3_chemical_rows.csv")
    plinder = chemical.loc[chemical["dataset"].eq("PLINDER") & chemical["structure_resolved"].eq(True)].copy()
    order = ["[0.00,0.30)", "[0.30,0.50)", "[0.50,0.70)", "[0.70,1.00]"]
    methods = {"LAIGN": "graph_ap_laign", "Fixed fusion": "graph_ap_fixed_fusion", "Distance": "graph_ap_distance"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.58), gridspec_kw={"width_ratios": [1.05, 1.10, 0.95], "wspace": 0.43})

    ax = axes[0]
    for dataset, display, color in [("PLINDER", "PLINDER", PALETTE["laign"]), ("Temporal", "Post-freeze", PALETTE["fixed"])]:
        values = np.sort(chemical.loc[chemical["dataset"].eq(dataset) & chemical["max_train_tanimoto"].notna(), "max_train_tanimoto"].to_numpy())
        x_ecdf = np.r_[values, 1.015]
        y_ecdf = np.r_[np.arange(1, len(values) + 1) / len(values), 1.0]
        ax.step(x_ecdf, y_ecdf, where="post", color=color, lw=DATA_LINE_WIDTH, label=f"{display} ($n={len(values)}$)")
    ax.set_xlabel("Maximum training-set Tanimoto")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    legend_above(ax, ncol=2)
    finish_axis(ax, "both")
    panel_label(ax, "(a)")

    ax = axes[1]
    summary_rows = []
    rng_summary = np.random.default_rng(2026)
    x = np.arange(len(order))
    for method, col in methods.items():
        means = []
        lower = []
        upper = []
        for bin_name in order:
            values = plinder.loc[plinder["similarity_bin"].eq(bin_name), col].to_numpy()
            estimate = values.mean()
            draws = rng_summary.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            means.append(estimate)
            lower.append(estimate - lo)
            upper.append(hi - estimate)
            summary_rows.append({"bin": bin_name, "method": method, "n": len(values), "estimate": estimate, "ci_low": lo, "ci_high": hi})
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            color=METHOD_COLOR[method],
            marker="o",
            ms=LINE_MARKER_SIZE,
            lw=DATA_LINE_WIDTH,
            elinewidth=ERROR_LINE_WIDTH,
            capsize=ERROR_CAPSIZE,
            label=method,
        )
    ax.set_xticks(x)
    bin_labels = ["0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"]
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(0.62, 1.0)
    ax.set_xlabel("Maximum training-set Tanimoto bin")
    ax.set_ylabel("Graph AP")
    legend_above(ax, ncol=3)
    finish_axis(ax, "y")
    panel_label(ax, "(b)")
    pd.DataFrame(summary_rows).to_csv(DATA / "figs3_similarity_summary.csv", index=False)

    ax = axes[2]
    rng = np.random.default_rng(2027)
    effect_rows = []
    for i, bin_name in enumerate(order):
        sub = plinder.loc[plinder["similarity_bin"].eq(bin_name)]
        delta = (sub["graph_ap_laign"] - sub["graph_ap_fixed_fusion"]).to_numpy()
        draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        yy = len(order) - 1 - i
        ax.plot([lo, hi], [yy, yy], color=PALETTE["laign"], lw=FOREST_LINE_WIDTH)
        ax.scatter(delta.mean(), yy, s=FOREST_MARKER_AREA, marker="o", color=PALETTE["laign"], edgecolor="white", linewidth=0.35, zorder=3)
        effect_rows.append({"bin": bin_name, "n": len(delta), "estimate": delta.mean(), "ci_low": lo, "ci_high": hi})
    ax.axvline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(bin_labels)
    ax.set_xlabel("Paired $\\Delta$ graph AP\n(LAIGN - fixed fusion)")
    finish_axis(ax, "x")
    panel_label(ax, "(c)")
    pd.DataFrame(effect_rows).to_csv(DATA / "figs3_similarity_effect.csv", index=False)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.25)
    export(fig, OUT / "FigS3_chemical_diagnostics")


def fig_s4() -> None:
    set_style()
    sites_per = pd.read_csv(DATA / "figs4_sites_per_complex.csv")
    mutations = pd.read_csv(DATA / "figs4_mutations_per_site.csv")
    overlap = pd.read_csv(DATA / "figs4_overlap.csv")
    scaffold = pd.read_csv(DATA / "figs3_scaffold_counts.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH - 0.08, 4.22), gridspec_kw={"wspace": 0.42, "hspace": 0.78})

    ax = axes[0, 0]
    bins = np.arange(0.5, sites_per["assayed_sites"].max() + 1.5, 1)
    ax.hist(sites_per["assayed_sites"], bins=bins, color=PALETTE["laign"], alpha=0.78, edgecolor=PALETTE["axis"], linewidth=0.35)
    ax.set_xlabel("Assayed sites per complex")
    ax.set_ylabel("Complexes")
    finish_axis(ax, "y")
    panel_label(ax, "(a)", x=-0.18)

    ax = axes[0, 1]
    counts = mutations["mutation_count"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, width=0.72, color=PALETTE["fixed"], alpha=0.88, edgecolor=PALETTE["axis"], linewidth=0.35)
    ax.set_xlabel("Mutations per assayed site")
    ax.set_ylabel("Assayed sites")
    finish_axis(ax, "y")
    panel_label(ax, "(b)", x=-0.18)

    ax = axes[1, 0]
    labels = ["Exact PDB", "Exact UniProt"]
    overlap_pct = (overlap.set_index("overlap")["count"] / overlap.set_index("overlap")["total"] * 100)
    values = [overlap_pct.loc["Exact PDB"], overlap_pct.loc["Exact UniProt"]]
    y = np.array([0.64, 0.36])
    ax.barh(y, values, height=0.10, color=PALETTE["fixed"], alpha=0.88, edgecolor=PALETTE["axis"], linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Complexes (%)")
    ax.set_xlim(0, 75)
    ax.set_ylim(0.15, 0.85)
    for yy, value in zip(y, values):
        ax.text(value + 0.4, yy, f"{value:.1f}%", va="center", fontsize=5.8)
    finish_axis(ax, "x")
    panel_label(ax, "(c)", x=-0.18)

    ax = axes[1, 1]
    datasets = ["PLINDER", "Temporal"]
    categories = ["Seen scaffold", "Unseen scaffold", "Ringless / descriptor unavailable"]
    colors = [PALETTE["laign"], PALETTE["fixed"], PALETTE["neutral"]]
    x = np.array([0.36, 0.64])
    bottom = np.zeros(len(datasets))
    totals = scaffold.groupby("dataset")["count"].sum().reindex(datasets).to_numpy()
    for category, color in zip(categories, colors):
        values = scaffold.loc[scaffold["category"].eq(category)].set_index("dataset")["count"].reindex(datasets).to_numpy()
        pct = values / totals * 100
        ax.bar(x, pct, bottom=bottom, width=0.10, color=color, edgecolor=PALETTE["axis"], linewidth=0.35, label=category)
        bottom += pct
    ax.set_xticks(x)
    ax.set_xticklabels(["PLINDER", "Post-freeze"])
    ax.set_xlim(0.15, 0.85)
    ax.set_ylabel("Complexes (%)")
    ax.set_ylim(0, 100)
    legend_above(ax, ncol=1, fontsize=5.9)
    finish_axis(ax, "y")
    panel_label(ax, "(d)", x=-0.18)

    fig.subplots_adjust(left=0.105, right=0.975, top=0.95, bottom=0.14)
    export(fig, OUT / "FigS4_platinum_audit")


def fig_s5() -> None:
    set_style()
    secondary = pd.read_csv(DATA / "figs5_secondary_metrics.csv")
    threshold = pd.read_csv(DATA / "figs5_threshold.csv")
    homology = pd.read_csv(DATA / "figs5_homology.csv")
    method_order = ["Distance", "Fixed fusion", "LAIGN", "Raw interaction", "ProLIF", "PLIP"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH - 0.27, 3.65), gridspec_kw={"wspace": 0.38, "hspace": 0.48})

    for ax, metric, sd, xlabel, panel in [
        (axes[0, 0], "auroc", "auroc_sd", "Site AUROC", "(a)"),
        (axes[0, 1], "spearman", "spearman_sd", "Spearman $\\rho$ with $\\Delta\\Delta G$", "(b)"),
    ]:
        sub = secondary.set_index("method").loc[method_order].reset_index()
        y = np.arange(len(sub))[::-1]
        for i, row in sub.iterrows():
            ax.errorbar(row[metric], y[i], xerr=row[sd], color=METHOD_COLOR[row["method"]], marker="o", ms=LINE_MARKER_SIZE, lw=0, elinewidth=ERROR_LINE_WIDTH, capsize=ERROR_CAPSIZE)
        ax.set_yticks(y)
        if panel == "(a)":
            ax.set_yticklabels(sub["method"])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel(xlabel)
        finish_axis(ax, "x")
        panel_label(ax, panel, x=-0.18)

    ax = axes[1, 0]
    for method in ["LAIGN", "Fixed fusion", "Distance"]:
        sub = threshold.loc[threshold["method"].eq(method)].sort_values("threshold")
        ax.errorbar(sub["threshold"], sub["estimate"], yerr=sub["sd"], color=METHOD_COLOR[method], marker="o", ms=LINE_MARKER_SIZE, lw=DATA_LINE_WIDTH, elinewidth=ERROR_LINE_WIDTH, capsize=ERROR_CAPSIZE, label=method)
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.set_xlabel("Disruptive threshold (kcal mol$^{-1}$)")
    ax.set_ylabel("Site AUPRC")
    legend_above(ax, ncol=3)
    finish_axis(ax, "y")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    subsets = ["all", "no exact PDB", "no exact UniProt", "sequence identity <70%", "sequence identity <30%"]
    labels = ["All", "No exact PDB", "No exact UniProt", "Identity <70%", "Identity <30%"]
    x = np.arange(len(subsets))
    offsets = {"LAIGN": -0.09, "Fixed fusion": 0, "Distance": 0.09}
    for method in ["LAIGN", "Fixed fusion", "Distance"]:
        sub = homology.loc[homology["method"].eq(method)].set_index("subset").loc[subsets]
        ax.errorbar(x + offsets[method], sub["estimate"], yerr=sub["sd"], color=METHOD_COLOR[method], marker="o", ms=LINE_MARKER_SIZE, lw=0, elinewidth=ERROR_LINE_WIDTH, capsize=ERROR_CAPSIZE, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Site AUPRC")
    legend_above(ax, ncol=3)
    finish_axis(ax, "y")
    panel_label(ax, "(d)")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.17)
    export(fig, OUT / "FigS5_mutation_secondary")


def fig_s6() -> None:
    set_style()
    hit = pd.read_csv(DATA / "figs6_hit.csv")
    budget = pd.read_csv(DATA / "figs6_budget_effect.csv")
    per = pd.read_csv(DATA / "figs6_per_complex_recall10.csv")
    methods = ["LAIGN", "Fixed fusion", "Distance", "ProLIF", "PLIP"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.72), gridspec_kw={"width_ratios": [1.08, 1.08, 0.84], "wspace": 0.58})

    ax = axes[0]
    for method in methods:
        row = hit.loc[hit["method"].eq(method)].iloc[0]
        ax.plot([5, 10], [row["hit_at_5_all_residues"], row["hit_at_10_all_residues"]], color=METHOD_COLOR[method], marker="o", ms=LINE_MARKER_SIZE, lw=DATA_LINE_WIDTH, label=method)
    ax.set_xticks([5, 10])
    ax.set_xlabel("Residue budget, $K$")
    ax.set_ylabel("Complex hit rate")
    ax.set_ylim(0.30, 0.72)
    legend_above(ax, ncol=3, fontsize=5.9)
    finish_axis(ax, "y")
    panel_label(ax, "(a)")

    ax = axes[1]
    sub = budget.loc[budget["budget_k"].eq(10)].copy()
    order = ["Fixed fusion", "Distance", "ProLIF", "PLIP"]
    sub = sub.set_index("comparator").loc[order].reset_index()
    y = np.arange(len(sub))[::-1]
    for i, row in sub.iterrows():
        color = METHOD_COLOR[row["comparator"]]
        ax.plot([row["ci95_low"], row["ci95_high"]], [y[i], y[i]], color=color, lw=FOREST_LINE_WIDTH)
        ax.scatter(row["observed_delta_recall"], y[i], s=FOREST_MARKER_AREA, color=color, marker="o", edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("$\\Delta$ recall@10\n(LAIGN - comparator)")
    finish_axis(ax, "x")
    panel_label(ax, "(b)", x=-0.20)

    ax = axes[2]
    delta = per["LAIGN"] - per["Distance"]
    ax.hist(delta, bins=np.linspace(-1, 1, 17), color=PALETTE["laign"], alpha=0.72, edgecolor=PALETTE["axis"], linewidth=0.35)
    ax.axvline(delta.mean(), color=PALETTE["fixed"], lw=DATA_LINE_WIDTH, label=f"mean {delta.mean():+.3f}")
    ax.axvline(0, color=PALETTE["axis"], lw=0.65, ls=(0, (2.5, 2.5)))
    ax.set_xlabel("Per-complex $\\Delta$\nrecall@10\n(LAIGN - distance)", labelpad=2)
    ax.set_ylabel("Complexes")
    legend_above(ax, ncol=1, fontsize=5.9)
    finish_axis(ax, "y")
    panel_label(ax, "(c)")

    fig.subplots_adjust(left=0.08, right=0.965, top=0.90, bottom=0.31)
    export(fig, OUT / "FigS6_budget_extended")


def main() -> None:
    fig_s1()
    fig_s2()
    fig_s3()
    fig_s4()
    fig_s5()
    fig_s6()
    print("Generated Fig. S1-S6")


if __name__ == "__main__":
    main()
