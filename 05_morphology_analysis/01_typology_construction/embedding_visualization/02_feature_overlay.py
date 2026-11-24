"""
02_feature_overlay.py
---------------------
Feature overlay plots on t-SNE coordinates.

Top-12 morphometric features ranked by η² (variance explained by cluster
membership after rank-based Inverse Normal Transform), from
../cluster_interpretation/outputs/tables/feature_importance.csv.
Complete-linkage deduplication at |ρ| ≥ 0.90; one representative per group.

Split into two 2×3 figures:
  tsne_feature_overlay_1.pdf/png  — rank 1–6
  tsne_feature_overlay_2.pdf/png  — rank 7–12

Output
------
outputs/figures/tsne_feature_overlay_1.pdf/png
outputs/figures/tsne_feature_overlay_2.pdf/png

Usage
-----
    python 02_feature_overlay.py

Note: run 01_tsne.py first to generate outputs/tsne_coords.parquet.
Paths are defined in config.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import MODEL_INPUT, TSNE_PARQUET, OUT_FIG_DIR

# ---------------------------------------------------------------------------
# Top-12 features by η² (complete-linkage dedup, one rep per group)
# Rank 1–6  → figure 1;  rank 7–12 → figure 2
# ---------------------------------------------------------------------------
FEATURES_1: dict[str, str] = {
    "BldgDensN":  "BldgDensN - Building density (norm.)",   # η²=0.780  rank 1
    "ShareDens":  "ShareDens - Shared-wall density",         # η²=0.708  rank 2
    "BldgCount":  "BldgCount - Building count",              # η²=0.706  rank 3
    "V_sum":      "V_sum - Total building volume",           # η²=0.688  rank 4
    "LC_Built":   "LC_Built - Built-up land cover",          # η²=0.647  rank 5
    "StrOpen":    "StrOpen - Street openness",               # η²=0.639  rank 6
}

FEATURES_2: dict[str, str] = {
    "Std_NbDis":  "Std_NbDis - Neighbour distance (std)",   # η²=0.632  rank 7
    "Std_IBDis":  "Std_IBDis - Inter-building dist. (std)", # η²=0.605  rank 8
    "SegmentDen": "SegmentDen - Street segment density",     # η²=0.597  rank 9
    "NodeDens":   "NodeDens - Street node density",          # η²=0.590  rank 10
    "Max_Share":  "Max_Share - Max. shared-wall length",     # η²=0.584  rank 11
    "LCF_Tree":   "LCF_Tree - Tree cover fraction",          # η²=0.578  rank 12
}

# ---------------------------------------------------------------------------
# Figure layout  (2 rows × 3 cols per figure)
# ---------------------------------------------------------------------------
NCOLS      = 3
NROWS      = 2
PANEL_W_MM = 60
PANEL_H_MM = 56
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
def plot_panel(df: pd.DataFrame, features: dict[str, str],
               stem: str) -> None:
    """Draw one 2×3 feature overlay figure and save as PDF + PNG."""
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * PANEL_W_MM * MM_TO_IN, NROWS * PANEL_H_MM * MM_TO_IN),
        dpi=600,
    )
    fig.patch.set_facecolor("white")

    for ax, (feat, label) in zip(axes.flat, features.items()):
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

        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                          orientation="vertical")
        cb.ax.tick_params(labelsize=5)
        cb.outline.set_visible(False)

    plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.5)

    for fmt in ("pdf", "png"):
        path = OUT_FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600, bbox_inches="tight", format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":  "sans-serif",
        "font.size":    7,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    # Load t-SNE coordinates
    print(f"Loading {TSNE_PARQUET} ...")
    tsne = pq.read_table(TSNE_PARQUET).to_pandas()[
        ["GridID", "tsne1", "tsne2", "label_k17"]
    ]

    # Load all feature values in one pass
    all_feats = list(FEATURES_1.keys()) + list(FEATURES_2.keys())
    print(f"Loading features from {MODEL_INPUT} ...")
    feats = pd.read_parquet(MODEL_INPUT, columns=["GridID"] + all_feats)
    feats["GridID"] = feats["GridID"].astype(str)

    df = tsne.merge(feats, on="GridID", how="left")
    n_complete = df[all_feats].notna().all(axis=1).sum()
    print(f"  Joined: {len(df):,} cells, {n_complete:,} complete")
    print(f"  Plotting full dataset: {len(df):,} cells")

    plot_panel(df, FEATURES_1, "tsne_feature_overlay_1")
    plot_panel(df, FEATURES_2, "tsne_feature_overlay_2")
    print("Done.")


if __name__ == "__main__":
    main()
