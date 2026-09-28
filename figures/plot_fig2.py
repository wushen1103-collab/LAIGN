from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import PALETTE, export, set_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "derived" / "fig2_paired_graph_ap_source.csv"
MAIN = ROOT / "main"

ORDER = ["LABind", "ProLIF", "P2Rank", "CLAPE-SMB", "UniSite", "fpocket", "SwinSite"]
COLOR = {
    "LABind": PALETTE["fixed"],
    "ProLIF": PALETTE["interaction"],
    "P2Rank": PALETTE["neutral"],
    "CLAPE-SMB": PALETTE["neutral"],
    "UniSite": PALETTE["neutral"],
    "fpocket": PALETTE["neutral"],
    "SwinSite": PALETTE["neutral"],
}


def main() -> None:
    set_style()
    frame = pd.read_csv(SOURCE)
    fig, ax = plt.subplots(figsize=(3.45, 2.90))
    positions = np.arange(len(ORDER), dtype=float)
    rng = np.random.default_rng(2027)

    for position, method in zip(positions, ORDER):
        values = frame.loc[frame["method"].eq(method), "delta_graph_ap"].to_numpy()
        violin = ax.violinplot(
            values,
            positions=[position],
            vert=False,
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        body = violin["bodies"][0]
        body.set_facecolor(COLOR[method])
        body.set_edgecolor(PALETTE["axis"])
        body.set_linewidth(0.55)
        body.set_alpha(0.68)

        ax.scatter(
            values,
            position + rng.uniform(-0.10, 0.10, size=values.size),
            s=3.2,
            color=COLOR[method],
            edgecolor="none",
            alpha=0.24,
            zorder=2,
        )
        ax.boxplot(
            values,
            positions=[position],
            vert=False,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": PALETTE["white"], "edgecolor": PALETTE["axis"], "linewidth": 0.55},
            medianprops={"color": PALETTE["axis"], "linewidth": 0.8},
            whiskerprops={"color": PALETTE["axis"], "linewidth": 0.55},
            capprops={"color": PALETTE["axis"], "linewidth": 0.55},
        )
        ax.scatter(
            values.mean(),
            position,
            s=17,
            marker="D",
            color=PALETTE["laign"],
            edgecolor=PALETTE["white"],
            linewidth=0.45,
            zorder=4,
        )

    ax.axhline(1.5, color=PALETTE["grid"], linewidth=0.55, zorder=0)
    ax.axvline(0.0, color=PALETTE["axis"], linewidth=0.7, linestyle=(0, (2.5, 2.5)))
    ax.set_xlim(-0.25, 1.02)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks(positions)
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ graph AP (LAIGN - baseline)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.grid(False, axis="y")
    for spine in ax.spines.values():
        spine.set_color(PALETTE["axis"])
        spine.set_linewidth(0.55)
    ax.tick_params(colors=PALETTE["text"], width=0.55, length=2.5)
    fig.subplots_adjust(left=0.265, right=0.975, bottom=0.19, top=0.96)
    export(fig, MAIN / "Fig2")


if __name__ == "__main__":
    main()
