"""Matplotlib style shared by the notebooks: one validated palette, quiet axes."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#9a9893"

# Fixed colour per model, so a model keeps its colour in every chart.
MODEL_COLORS = {
    "premium_catboost": SERIES[0],
    "quantile_mlp": SERIES[1],
    "target_encoded_hgb": SERIES[2],
    "recent_median": SERIES[3],
    "csfloat_reference": NEUTRAL,
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", horizontal: bool = True) -> None:
    """Value at the tip of each bar, in text ink (never the series colour)."""
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            ax.annotate(fmt.format(value), (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(4 if value >= 0 else -4, 0), textcoords="offset points", va="center",
                        ha="left" if value >= 0 else "right", color=TEXT_SECONDARY, fontsize=9)
        else:
            value = bar.get_height()
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=9)
