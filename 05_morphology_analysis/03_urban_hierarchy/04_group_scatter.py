"""
04_group_scatter.py
===================
One combined 2x2 figure (p1-p4, one morphological group per panel):
scatter coloured by community (low-alpha background texture) +
per-community LOESS curves + a continental LOESS reference line (black
dashed, no CI band — this is a quick visual anchor, not a re-statement of
02_plot_hierarchy.py/07_size_gradient.py's own dedicated, CI-backed
figures for the continental relationship). One shared legend for the
whole figure (not per-panel), since all four panels use the same 8
communities.

Output
------
    outputs/figures/group_scatter_combined.pdf/png

Usage
-----
    python 04_group_scatter.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Register a font whose internal name is "Arial" (see other scripts in
# this project) so matplotlib doesn't silently fall back to DejaVu Sans.
_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    logging.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)

matplotlib.rcParams.update({
    "font.family":     "Arial",
    "font.size":       10,
    "svg.fonttype": "none",
})

ROOT      = Path(__file__).resolve().parent
METRICS   = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR   = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

# Same community palette as community_detection's PALETTE_9, indexed by
# the original 0-based community ID.
COMM_COLORS = {
    0: "#f36424",
    1: "#672e92",
    2: "#8bcf02",
    3: "#139ed3",
    4: "#ff0000",
    5: "#ffd700",
    6: "#293890",
    7: "#74201e",
}

GROUP_NAMES = {
    "p1": "G1 (CFL) — Cropland-Fringe Low-Rise",
    "p2": "G2 (HSF) — Heterogeneous Sparse Fabric",
    "p3": "G3 (LVC) — Large-Volume Compact Fabric",
    "p4": "G4 (DSB) — Dense Small-Building Fabric",
}

# matplotlib's `s` is marker AREA (points^2); halving the RADIUS means
# area -> area * (1/2)^2 = area / 4.
SCATTER_SIZE   = 6 / 4
LOESS_FRAC     = 0.35


# ---------------------------------------------------------------------------
# LOESS
# ---------------------------------------------------------------------------
def fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(len(x_grid), np.nan)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


# ---------------------------------------------------------------------------
# Axes style
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=7)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")
    ax.title.set_color("#222222")


# ---------------------------------------------------------------------------
# One panel
# ---------------------------------------------------------------------------
def plot_one_panel(ax, df, y_col, title, communities, x_grid, x_lo, x_hi):
    x_all = df["log_pop"].values
    y_all = df[y_col].values

    # Scatter: coloured by community, low alpha — background texture
    # showing data support, not meant to be read off directly (the
    # per-community LOESS curves are the figure's actual message).
    for comm in communities:
        mask = df["community"].values == comm
        ax.scatter(x_all[mask], y_all[mask],
                   c=COMM_COLORS[comm], s=SCATTER_SIZE, alpha=0.15,
                   edgecolors="none", linewidths=0, rasterized=True, zorder=2)

    # Per-community LOESS curves
    for comm in communities:
        mask = df["community"].values == comm
        yhat = fit_loess(x_all[mask], y_all[mask], x_grid)
        ax.plot(x_grid, yhat, color=COMM_COLORS[comm], lw=1.1, zorder=3)

    # Continental reference (quick visual anchor, no bootstrap CI band —
    # see module docstring)
    y_cont = fit_loess(x_all, y_all, x_grid)
    ax.plot(x_grid, y_cont, color="black", lw=1.6, linestyle="--", zorder=4)

    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=9, pad=5, fontweight="bold")
    ax.set_xlabel("log(population)", fontsize=8)
    ax.set_ylabel("Group share", fontsize=8)
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community", "p1", "p2", "p3", "p4"]).copy()
    df["community"] = df["community"].astype(int)
    communities = sorted(df["community"].unique())
    logging.info("  %d cities, %d communities.", len(df), len(communities))

    x_lo   = np.nanpercentile(df["log_pop"], 1)
    x_hi   = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 130 * MM))
    for ax, (y_col, title) in zip(axes.ravel(), GROUP_NAMES.items()):
        logging.info("=== %s ===", y_col)
        plot_one_panel(ax, df, y_col, title, communities, x_grid, x_lo, x_hi)
    fig.patch.set_facecolor("white")

    # One shared legend for the whole figure — all four panels use the
    # same 8 communities, so a per-panel legend would just repeat.
    # Label = 1-based community number (c+1), matching every other
    # community figure's display convention (internal ID is 0-based, but
    # shown as "1".."8" everywhere, e.g. community_map.svg's legend).
    comm_handles = [
        mlines.Line2D([], [], color=COMM_COLORS[c], lw=1.5, label=str(c + 1))
        for c in communities
    ]
    cont_handle = mlines.Line2D(
        [], [], color="black", lw=1.6, linestyle="--", label="Africa"
    )
    fig.legend(handles=comm_handles + [cont_handle], title="Community",
              fontsize=7, title_fontsize=7.5, frameon=False,
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              handlelength=1.4)

    fig.tight_layout(pad=1.0)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"group_scatter_combined.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: group_scatter_combined.pdf/png")

    logging.info("Done. Figures in %s", FIG_DIR)


if __name__ == "__main__":
    main()
