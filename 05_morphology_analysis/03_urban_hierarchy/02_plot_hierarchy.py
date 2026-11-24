"""
02_plot_hierarchy.py
====================
Continental urban hierarchy figure for publication.

Output
------
    outputs/figures/urban_hierarchy.pdf/png

    Single axes: four morphological group shares (p1–p4) vs log(population).
    LOESS smooth + 95% country-cluster bootstrap CI per group.

Usage
-----
    python 02_plot_hierarchy.py [--boots N] [--workers N]

    --boots     Bootstrap resamples (default: 500)
    --workers   Parallel workers (default: 14)
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
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
CACHE_DIR = ROOT / "outputs" / "cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

GROUP_NAMES = {
    1: "Cropland-Fringe Low-Rise",
    2: "Heterogeneous Sparse Fabric",
    3: "Large-Volume Compact Fabric",
    4: "Dense Small-Building Fabric",
}
# Same 4-group colour mapping as cluster_profiles_heatmap.svg's
# GROUP4_COLORS (01_typology_construction/cluster_interpretation/
# 04_cluster_profiles.py) — verified the group NUMBERING corresponds
# exactly (fcluster on the same D_cj matrix, cut at k=4, reproduces
# group 1={4,5}, 2={1,3,6,7,8}, 3={12,13}, 4={2,9,10,11,14,15,16,17},
# matching this module's GROUP_MAP), so group 1-4 here really is the
# same grouping as there, not just a coincidentally-same-sized partition.
GROUP_COLORS = {
    1: "#8DA0CB",
    2: "#FC8D62",
    3: "#66C2A5",
    4: "#E78AC3",
}

LOESS_FRAC        = 0.35
N_BOOTS_DEFAULT   = 500
N_WORKERS_DEFAULT = 14

# Composition-share breakpoint (09a_group_share_breakpoint.py): the slope
# "knee" of p1 (flattening), p2 (peak/zero-crossing), and p4 (flattening)
# cluster tightly around pop ~20k (p3's "starts rising" knee trails at
# ~36k but p3's total movement is small, so it's a lagged tail of the
# same transition, not a separate break) — found directly on these same
# curves, not on a derived index like H17/r17.
BREAK_POP = 20_000


# ---------------------------------------------------------------------------
# LOESS + country-cluster bootstrap CI
# ---------------------------------------------------------------------------
def _fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(len(x_grid), np.nan)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


def _loess_boot_worker(args):
    seed, x_vals, y_vals, countries, x_grid, frac = args
    rng      = np.random.default_rng(seed)
    unique_c = np.unique(countries)
    chosen   = rng.choice(unique_c, size=len(unique_c), replace=True)
    idx      = np.concatenate([np.where(countries == c)[0] for c in chosen])
    xb, yb   = x_vals[idx], y_vals[idx]
    mask     = np.isfinite(xb) & np.isfinite(yb)
    if mask.sum() < 20:
        return None
    try:
        sm = lowess(yb[mask], xb[mask], frac=frac, return_sorted=True)
        return np.interp(x_grid, sm[:, 0], sm[:, 1])
    except Exception:
        return None


def _loess_ci(df, y_col, x_grid, n_boots, n_workers, frac=LOESS_FRAC):
    """LOESS point estimate + 95% country-cluster bootstrap CI (cached).

    The cache is invalidated whenever METRICS' mtime changes, so a
    regenerated city_metrics.parquet always triggers a fresh bootstrap
    rather than silently reusing a CI computed from stale input data.
    """
    cache_file  = CACHE_DIR / f"loess_ci_{y_col}_b{n_boots}.npz"
    input_mtime = METRICS.stat().st_mtime
    x   = df["log_pop"].values
    y   = df[y_col].values
    cty = df["country"].values
    y_hat = _fit_loess(x, y, x_grid, frac=frac)

    if cache_file.exists():
        cached = np.load(cache_file)
        if float(cached.get("input_mtime", -1)) == input_mtime:
            logging.info("  Loading cached CI for %s ...", y_col)
            return y_hat, cached["ci_lo"], cached["ci_hi"]
        logging.info("  Cache stale for %s (input changed), recomputing ...", y_col)

    seeds = [int(sq.generate_state(1)[0])
             for sq in np.random.SeedSequence(42).spawn(n_boots)]
    args  = [(sd, x, y, cty, x_grid, frac) for sd in seeds]
    preds = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_loess_boot_worker, args):
            if res is not None:
                preds.append(res)
    preds = np.array(preds)
    ci_lo = np.percentile(preds, 2.5,  axis=0)
    ci_hi = np.percentile(preds, 97.5, axis=0)
    np.savez(cache_file, ci_lo=ci_lo, ci_hi=ci_hi, input_mtime=input_mtime)
    logging.info("  CI cached: %s", cache_file.name)
    return y_hat, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Axes style
# ---------------------------------------------------------------------------
def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=9)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def plot_hierarchy(df, n_boots, n_workers):
    x_lo   = np.nanpercentile(df["log_pop"], 1)
    x_hi   = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, ax = plt.subplots(figsize=(130 * MM, 100 * MM))
    fig.patch.set_facecolor("white")

    y_band_lo, y_band_hi = [], []
    for g in range(1, 5):
        y_col = f"p{g}"
        col   = GROUP_COLORS[g]
        label = GROUP_NAMES[g]
        logging.info("  LOESS + CI: group %d ...", g)
        y_hat, lo, hi = _loess_ci(df, y_col, x_grid,
                                   n_boots=n_boots, n_workers=n_workers)
        y_band_lo.append(np.nanmin(lo))
        y_band_hi.append(np.nanmax(hi))
        ax.fill_between(x_grid, lo, hi, color=col, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x_grid, y_hat, color=col, lw=1.4, label=label, zorder=3)

    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)
    # Y-range fit to the actual data (CI bands), not hardcoded — a fixed
    # ceiling tuned for one grouping silently clips curves/CIs under a
    # different grouping (e.g. a group whose LOESS share legitimately
    # exceeds that ceiling at the low- or high-population end).
    y_min = max(-0.02, min(y_band_lo) - 0.02)
    y_max = min(1.02, max(y_band_hi) + 0.02)
    ax.set_ylim(y_min, y_max)

    log_break = np.log(BREAK_POP)
    ax.axvline(log_break, color="#555555", lw=0.9, ls="--", zorder=4)
    ax.text(log_break + 0.08, y_max - 0.03, f"~{BREAK_POP:,}",
           fontsize=7.5, color="#555555", va="top", ha="left")
    ax.set_xlabel("log(population)", fontsize=10)
    ax.set_ylabel("Group share", fontsize=10)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(which="major", labelsize=9, length=4)
    ax.tick_params(which="minor", length=2, labelbottom=False, labelleft=False)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              ncol=1, handlelength=1.5, handletextpad=0.5)
    _despine(ax)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"urban_hierarchy.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: urban_hierarchy.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boots=N_BOOTS_DEFAULT, n_workers=N_WORKERS_DEFAULT):
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community", "p1", "p2", "p3", "p4"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    plot_hierarchy(df, n_boots=n_boots, n_workers=n_workers)
    logging.info("Done. Figure in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS_DEFAULT)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
