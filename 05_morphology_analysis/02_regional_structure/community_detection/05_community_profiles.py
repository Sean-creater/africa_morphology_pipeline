"""
05_community_profiles.py
=========================
Fig 2c: makes the 8 Leiden communities interpretable by showing, for each
community, (a) its 17-urban-form-type composition and (b) its type-level
spatial adjacency structure (log-enrichment, aggregated across member
cities) — i.e. "what mix of urban fabric does this community have, and
how does that fabric tend to sit together on the ground."

Inputs (both already built; no new analysis — pure aggregation)
-----------------------------------------------------------------
outputs/city_features.parquet
    Per city: p_1..p_17 (type composition shares), A_i_j (upper-triangle
    log-enrichment adjacency, NaN where undefined) — from
    01_build_city_features.py.
outputs/city_communities.parquet
    Per city: community (0-7) — from 02_leiden_communities.py.

Aggregation
-----------
Community-level composition = simple mean of p_i across member cities
(unweighted by city size, consistent with how community detection itself
treated each city as one unit).
Community-level adjacency = nanmean of A_ij across member cities (cities
where a given pair is undefined are simply excluded from that pair's
mean, not treated as zero).

Layout (landscape)
-------------------
Left  — 8 (communities) x 17 (types) composition heatmap.
Right — 8 small-multiple 17x17 log-enrichment adjacency heatmaps
        (2 rows x 4 cols), one per community.

Outputs
-------
outputs/figures/community_profiles.svg (+ .png)

Usage
-----
    python 05_community_profiles.py
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    logging.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)
matplotlib.rcParams.update({
    "font.family":   "Arial",
    "font.size":     8,
    "svg.fonttype":  "none",
    "mathtext.fontset": "custom",
    "mathtext.rm":       "Arial",
    "mathtext.it":       "Arial:italic",
    "mathtext.bf":       "Arial:bold",
})

ROOT       = Path(__file__).resolve().parent
FEAT_PATH  = ROOT / "outputs" / "city_features.parquet"
COMM_PATH  = ROOT / "outputs" / "city_communities.parquet"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

N_TYPES = 17

# Community colour palette (must match 03_visualize_communities.py) and
# geographic names, ranked by descending city count (community id already
# equals rank order in this dataset: 0 is largest).
PALETTE_9 = [
    "#f36424", "#672e92", "#8bcf02", "#139ed3",
    "#ff0000", "#ffd700", "#293890", "#74201e",
]
COMM_NAMES = [
    "West African Inland",
    "Gulf of Guinea Coastal",
    "Southern–East African",
    "East–Central African",
    "Sahel–Central African Interior",
    "North African–Nile",
    "Nile Urban Corridor",
    "East African Highlands",
]


def load_data():
    feat = pd.read_parquet(FEAT_PATH)
    comm = pd.read_parquet(COMM_PATH)[["country", "city", "community"]]
    df = feat.merge(comm, on=["country", "city"], how="inner")
    logging.info("Merged %d cities with features + community.", len(df))
    return df


def community_composition(df: pd.DataFrame) -> pd.DataFrame:
    """(8 x 17) mean type-share matrix, rows ordered by community id."""
    p_cols = [f"p_{i}" for i in range(1, N_TYPES + 1)]
    out = df.groupby("community")[p_cols].mean().sort_index()
    return out


def community_adjacency(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """{community: (17x17) symmetric log-enrichment matrix (nanmean across
    member cities, mirrored from the stored upper triangle)."""
    mats = {}
    for comm, sub in df.groupby("community"):
        M = np.full((N_TYPES, N_TYPES), np.nan)
        for i in range(1, N_TYPES + 1):
            for j in range(i, N_TYPES + 1):
                col = f"A_{i}_{j}"
                if col not in sub.columns:
                    continue
                val = np.nanmean(sub[col].values)
                M[i - 1, j - 1] = val
                M[j - 1, i - 1] = val
        mats[comm] = M
    return dict(sorted(mats.items()))


# Same diverging scheme as 01_typology_construction/cluster_interpretation/
# 04_cluster_profiles.py's cluster_profiles_heatmap, for visual consistency
# across the paper's heatmaps. Adjacency (log-enrichment) is signed, so it
# uses the full diverging ramp; composition (a share, always >= 0) uses
# just the red half (white -> red) since there is no "negative share".
DIV_BLUE, DIV_WHITE, DIV_RED = "#4756A0", "#FFFFFF", "#C8292A"
CMAP_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", [DIV_BLUE, DIV_WHITE, DIV_RED], N=256)
CMAP_SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "custom_seq", [DIV_WHITE, DIV_RED], N=256)

GAMMA = 0.55  # contrast boost shared by both figures (see plot_adjacency)


def plot_composition(comp: pd.DataFrame, n_cities: pd.Series):
    """Fig: 8 (communities) x 17 (types) composition heatmap alone."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    comp_vals = comp.values
    x_edges = np.arange(N_TYPES + 1) - 0.5
    y_edges = np.arange(len(comp), -1, -1) - 0.5
    vmax = np.nanpercentile(comp_vals, 97)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    ax.pcolormesh(x_edges, y_edges, comp_vals, cmap=CMAP_SEQUENTIAL, norm=norm,
                 shading="flat", edgecolors="white", linewidth=0.6)
    ax.set_xlim(-0.5, N_TYPES - 0.5)
    ax.set_ylim(-0.5, len(comp) - 0.5)
    ax.set_xticks(range(N_TYPES))
    ax.set_xticklabels([str(i) for i in range(1, N_TYPES + 1)], fontsize=6.5)
    ax.set_xlabel("Urban form type", fontsize=8)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels([f"C{c+1}" for c in comp.index][::-1], fontsize=7)
    for i, c in enumerate(comp.index):
        ax.add_patch(plt.Rectangle(
            (-1.1, len(comp) - 1 - i - 0.5), 0.5, 1.0,
            facecolor=PALETTE_9[c], edgecolor="none", clip_on=False))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Urban form type composition by community", fontsize=9,
                fontweight="bold")

    cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0,
                        0.018, ax.get_position().height])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SEQUENTIAL)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Mean share of grid cells", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6)

    out = OUT_FIG / "community_composition.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s (+ .png)", out)


def plot_adjacency(adj: dict[int, np.ndarray]):
    """Fig: 8 small-multiple 17x17 log-enrichment adjacency heatmaps."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.3))
    axes = axes.ravel()

    for c in range(8):
        axm = axes[c]
        M = adj[c]
        M_plot = np.ma.masked_invalid(M)
        # Masked (NaN = pair never observed adjacent in any member city)
        # cells are left undrawn by pcolormesh, so they show the axes'
        # own background through — set that to light grey, VISUALLY
        # DISTINCT from the colormap's own white centre (0 = pair observed,
        # just with no net enrichment or depletion). Without this, "missing"
        # and "truly neutral" were indistinguishable (both plain white).
        axm.set_facecolor("#D9D9D9")

        # Colour normalised PER COMMUNITY (own distribution of |A_ij|), not
        # pooled across all 8 — signal strength varies a lot between
        # communities (e.g. C3/C5/C8 reach |A_ij|~5 while C1/C2/C4/C6/C7 top
        # out near 2), so one shared scale left weak-signal communities
        # looking blank. vmax at the 70th (not max) percentile plus a
        # gamma<1 power both push medium magnitudes toward full saturation
        # instead of leaving everything but the extremes pale.
        finite_vals = M[np.isfinite(M)]
        vmax_c = np.nanpercentile(np.abs(finite_vals), 70) if finite_vals.size else 1.0
        vmax_c = max(vmax_c, 1e-6)
        frac = np.clip(M_plot / vmax_c, -1, 1)
        M_gamma = np.sign(frac) * np.abs(frac) ** GAMMA

        axm.pcolormesh(np.arange(N_TYPES + 1) - 0.5, np.arange(N_TYPES + 1) - 0.5,
                      M_gamma, cmap=CMAP_DIVERGING,
                      norm=mcolors.Normalize(vmin=-1, vmax=1),
                      shading="flat")
        axm.set_xlim(-0.5, N_TYPES - 0.5)
        axm.set_ylim(N_TYPES - 0.5, -0.5)
        axm.set_aspect("equal")
        # Endpoint-only ticks (1 -> 17): at this panel size the ticks only
        # need to convey the type ordering/direction, not let the reader
        # pinpoint an exact row/column by eye — labelling all 17 (or even
        # every other one) left numbers crowding the small panel. Anchoring
        # just the two ends with an arrow is the standard, compact
        # convention for small-multiple matrix panels like these.
        axm.set_xticks([0, N_TYPES - 1])
        axm.set_yticks([0, N_TYPES - 1])
        axm.set_xticklabels(["1 →", "17"], fontsize=6.5)
        axm.set_yticklabels(["1 →", "17"], fontsize=6.5)
        axm.tick_params(length=0, pad=1)
        for spine in axm.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(0.6)
        axm.set_title(f"C{c+1}", fontsize=8, fontweight="bold",
                      color=PALETTE_9[c], pad=3)

    fig.suptitle(
        "Type-level spatial adjacency by community\n"
        "colour = mean log-enrichment ($A_{ij}$), scale normalised per community",
        fontsize=9, y=1.02)

    cax = fig.add_axes([0.38, -0.02, 0.24, 0.02])
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-1, vmax=1),
                               cmap=CMAP_DIVERGING)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["depleted", "0", "enriched"])
    cb.ax.tick_params(labelsize=6)

    grey_ax = fig.add_axes([0.66, -0.025, 0.012, 0.03])
    grey_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9",
                                    edgecolor="#999999", linewidth=0.4))
    grey_ax.axis("off")
    fig.text(0.675, -0.01, "no data (pair never adjacent in any member city)",
             fontsize=6, va="center", ha="left")

    out = OUT_FIG / "community_adjacency.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s (+ .png)", out)


def main():
    df = load_data()
    comp = community_composition(df)
    adj = community_adjacency(df)
    n_cities = df["community"].value_counts().sort_index()
    plot_composition(comp, n_cities)
    plot_adjacency(adj)


if __name__ == "__main__":
    main()
