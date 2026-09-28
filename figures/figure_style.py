from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


MM = 1 / 25.4
FULL_WIDTH = 169.3 * MM
SINGLE_WIDTH = 82.0 * MM
DATA_LINE_WIDTH = 0.67
SEED_LINE_WIDTH = 0.40
ERROR_LINE_WIDTH = 0.44
FOREST_LINE_WIDTH = ERROR_LINE_WIDTH
LINE_MARKER_SIZE = 2.4
FOREST_MARKER_AREA = LINE_MARKER_SIZE**2
ERROR_CAPSIZE = 1.1
LEGEND_FONT_SIZE = 6.5

PALETTE = {
    "laign": "#6FA8DC",
    "fixed": "#F0A66E",
    "interaction": "#76B7A6",
    "neutral": "#C7CBD1",
    "distance": "#9D95C2",
    "axis": "#3F454B",
    "text": "#25292D",
    "grid": "#E9EDF0",
    "white": "#FFFFFF",
}

METHOD_COLOR = {
    "LAIGN": PALETTE["laign"],
    "Fixed fusion": PALETTE["fixed"],
    "Distance": PALETTE["distance"],
    "Raw interaction": PALETTE["interaction"],
    "PLIP": PALETTE["interaction"],
    "ProLIF": PALETTE["neutral"],
}

METHOD_MARKER = {
    "LAIGN": "o",
    "Fixed fusion": "o",
    "Distance": "o",
    "Raw interaction": "o",
    "PLIP": "o",
    "ProLIF": "o",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.fallback": "stixsans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "lines.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def finish_axis(ax, grid_axis: str | None = None) -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.45, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["axis"])
        spine.set_linewidth(0.5)
    ax.tick_params(colors=PALETTE["text"], width=0.5, length=2.2)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.05) -> None:
    label = label.strip().removeprefix("(").removesuffix(")")
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["text"],
        clip_on=False,
    )


def legend_above(ax, *, ncol: int = 2, y: float = 1.02, fontsize: float = LEGEND_FONT_SIZE, **kwargs):
    return ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        frameon=False,
        ncol=ncol,
        fontsize=fontsize,
        columnspacing=0.72,
        handlelength=1.15,
        handletextpad=0.30,
        borderaxespad=0,
        **kwargs,
    )


def export(fig, stem: Path, *, dpi: int = 600, pad_pt: float = 5.0) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": pad_pt / 72, "facecolor": "white"}
    fig.savefig(stem.with_suffix(".svg"), **kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, **kwargs)
    plt.close(fig)
