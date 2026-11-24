"""
03_feature_overlay_combined.py
-------------------------------
Combined single-figure version of 02_feature_overlay.py's two 2x3 panels,
for use as Supplementary Figure S1 (one figure, one number) rather than
two separate figures.

Top-12 morphometric features ranked by eta^2 (variance explained by cluster
membership after rank-based Inverse Normal Transform), from
../cluster_interpretation/outputs/tables/feature_importance.csv.
Complete-linkage deduplication at |rho| >= 0.90; one representative per
group. Same features, colour scale and per-panel styling as
02_feature_overlay.py -- only the layout changes (one 4x3 grid, portrait,
A4-ratio, instead of two separate 2x3 figures).

Output
------
outputs/figures/tsne_feature_overlay_combined.pdf/png/svg

Usage
-----
    python 03_feature_overlay_combined.py

Note: run 01_tsne.py first to generate outputs/tsne_coords.parquet.
Paths are defined in config.py.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import MODEL_INPUT, TSNE_PARQUET, OUT_FIG_DIR

_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    logging.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)

# ---------------------------------------------------------------------------
# Top-12 features by eta^2 (complete-linkage dedup, one rep per group),
# same ranking/order as 02_feature_overlay.py's FEATURES_1 + FEATURES_2.
# Panel titles use the full metric names loaded from column_name_mapping.csv
# (see load_full_name_map() below) -- the same Short_Name -> Full_Name map
# 04_cluster_profiles.py uses for cluster_profiles_heatmap.svg's axis
# labels -- so panel titles here match that figure's metric names exactly,
# not the abbreviated short names.
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "BldgDensN",   # eta2=0.780  rank 1
    "ShareDens",   # eta2=0.708  rank 2
    "BldgCount",   # eta2=0.706  rank 3
    "V_sum",       # eta2=0.688  rank 4
    "LC_Built",    # eta2=0.647  rank 5
    "StrOpen",     # eta2=0.639  rank 6
    "Std_NbDis",   # eta2=0.632  rank 7
    "Std_IBDis",   # eta2=0.605  rank 8
    "SegmentDen",  # eta2=0.597  rank 9
    "NodeDens",    # eta2=0.590  rank 10
    "Max_Share",   # eta2=0.584  rank 11
    "LCF_Tree",    # eta2=0.578  rank 12
]

NAME_MAPPING_CSV = Path(__file__).resolve().parent.parent / "cluster_interpretation" / "reference" / "column_name_mapping.csv"


def load_full_name_map() -> dict[str, str]:
    """Short_Name -> Full_Name, same source as 04_cluster_profiles.py."""
    mapping = pd.read_csv(NAME_MAPPING_CSV)
    return dict(zip(mapping["Short_Name"], mapping["Full_Name"]))

# ---------------------------------------------------------------------------
# Figure layout (4 rows x 3 cols -- all 12 panels in one figure).
# Portrait, sized to the A4 aspect ratio (210x297 mm) for a single SI page.
# ---------------------------------------------------------------------------
NCOLS      = 3
NROWS      = 4
PANEL_W_MM = 60
PANEL_H_MM = 60   # scatter itself is ~square; row height set by the data,
                  # not padded out -- keep close to PANEL_W_MM
MM_TO_IN   = 1 / 25.4
CMAP       = "RdBu_r"
CLIP_PCT   = 2       # clip top/bottom 2 % to suppress outlier colour saturation


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def clip_values(series: pd.Series, pct: float) -> pd.Series:
    """Clip to [pct, 100-pct] percentile to avoid colour saturation by outliers."""
    lo, hi = np.percentile(series.dropna(), [pct, 100 - pct])
    return series.clip(lo, hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_combined(df: pd.DataFrame, features: dict[str, str], stem: str) -> None:
    """Draw one 4x3 feature overlay figure (all 12 panels) and save.

    Each panel is normalised independently (clip + per-feature median
    centring -- see clip_values()/TwoSlopeNorm below), so a colour bar's
    low/mid/high labels mean the same relative thing in every panel. One
    shared legend therefore replaces what would otherwise be twelve
    identical low/mid/high colour bars.
    """
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * PANEL_W_MM * MM_TO_IN, NROWS * PANEL_H_MM * MM_TO_IN),
        dpi=600,
    )
    fig.patch.set_facecolor("white")

    sc = None
    for i, (ax, (feat, label)) in enumerate(zip(axes.flat, features.items())):
        vals = clip_values(df[feat], CLIP_PCT)
        vmin, vcenter, vmax = vals.min(), vals.median(), vals.max()
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sc = ax.scatter(
            df["tsne1"], df["tsne2"],
            c=vals, cmap=CMAP, norm=norm,
            s=0.04, alpha=0.9, linewidths=0, rasterized=True,
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.set_title(label, fontsize=7, pad=3)

    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3, rect=(0, 0, 0.93, 1))

    # One shared colour legend for all 12 panels (see docstring above for
    # why a single low/mid/high bar is sufficient). Centred on the full
    # figure height (not just one row) -- the grid is 4 rows tall now
    # (portrait/A4 layout), so a legend sized to one row's height would
    # sit off-centre relative to the whole page.
    cax = fig.add_axes((0.95, 0.4, 0.015, 0.2))
    cb = fig.colorbar(sc, cax=cax, orientation="vertical",
                      ticks=[sc.norm.vmin, sc.norm.vcenter, sc.norm.vmax])
    cb.ax.set_yticklabels(["low", "mid", "high"])
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)

    for fmt in ("svg", "png"):
        path = OUT_FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600, bbox_inches="tight", format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":  "Arial",
        "font.size":    8,
        "svg.fonttype": "none",
    })

    print(f"Loading {TSNE_PARQUET} ...")
    tsne = pq.read_table(TSNE_PARQUET).to_pandas()[
        ["GridID", "tsne1", "tsne2", "label_k17"]
    ]

    full_name_map = load_full_name_map()
    features = {feat: full_name_map[feat] for feat in FEATURE_ORDER}

    all_feats = list(features.keys())
    print(f"Loading features from {MODEL_INPUT} ...")
    feats = pd.read_parquet(MODEL_INPUT, columns=["GridID"] + all_feats)
    feats["GridID"] = feats["GridID"].astype(str)

    df = tsne.merge(feats, on="GridID", how="left")
    n_complete = df[all_feats].notna().all(axis=1).sum()
    print(f"  Joined: {len(df):,} cells, {n_complete:,} complete")
    print(f"  Plotting full dataset: {len(df):,} cells")

    plot_combined(df, features, "tsne_feature_overlay_combined")
    print("Done.")


if __name__ == "__main__":
    main()
