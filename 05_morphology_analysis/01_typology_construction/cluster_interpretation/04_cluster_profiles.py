"""
04_cluster_profiles.py
----------------------
Characterise each of the 17 urban-form clusters by the signed standardised
mean deviation (D_cj) of morphometric indicators across the features
identified in steps 02–03.

Method
------
For each cluster c and feature j, after rank-based Inverse Normal Transform:

    D_cj = (x̄_cj − x̄_j) / s_j

Because rank-INT maps every feature to mean ≈ 0, s ≈ 1 globally,
D_cj ≈ x̄_cj (the cluster mean on the transformed scale).  Positive values
indicate that the cluster lies above the global average for that feature;
negative values indicate below.  The magnitude reflects how many standard
deviations the cluster deviates from the full-Africa mean.

Features are ordered by their global η² rank (from step 03) within each
dimension group.  Both cluster rows and feature columns are reordered by
Ward hierarchical clustering of the D matrix to reveal block structure.

Outputs
-------
outputs/tables/cluster_profiles.csv
    (17 clusters × N features) D_cj matrix; index = cluster label.
outputs/tables/cluster_sizes.csv
    Grid-cell count and percentage share per cluster.
outputs/figures/cluster_profiles_heatmap.pdf
    Publication-ready clustered heatmap with Ward dendrograms on both
    axes, dimension-group colour bands above feature labels, and cluster
    sizes on the y-axis.

Usage
-----
    python 04_cluster_profiles.py [--top N]

    --top   Maximum number of features to include (default: all selected,
            capped at 40 for readability).
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.special import ndtri

# Register a font whose internal name is "Arial" so matplotlib's mathtext
# engine (used for the "$D_{cj}$" label) can find one — otherwise it
# silently falls back to DejaVu Sans for equation glyphs in the SVG output.
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

N_JOBS = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).resolve().parent
MODEL_INPUT        = ROOT / "outputs" / "tables" / "merged_k17.parquet"
IMPORTANCE_CSV     = ROOT / "outputs" / "tables" / "feature_importance.csv"
OUT_TABLES         = ROOT / "outputs" / "tables"
OUT_FIGURES        = ROOT / "outputs" / "figures"
CACHE_DIR          = ROOT / "outputs" / "cache"
INT_CACHE          = CACHE_DIR / "rank_int_top.parquet"  # rank-INT data + label_k17
NAME_MAPPING_CSV   = ROOT / "reference" / "column_name_mapping.csv"

# Manual fallback for the one feature absent from NAME_MAPPING_CSV (a raw
# intersection count; only its ratio counterpart "Ratio4way" is catalogued).
_FALLBACK_FULL_NAMES = {"FourWayCnt": "4-Way Intersection Count"}


def load_full_name_map() -> dict[str, str]:
    """Short_Name -> Full_Name from the project's morphometric codebook."""
    mapping = pd.read_csv(NAME_MAPPING_CSV)
    name_map = dict(zip(mapping["Short_Name"], mapping["Full_Name"]))
    name_map.update(_FALLBACK_FULL_NAMES)
    return name_map


# ---------------------------------------------------------------------------
# Rank-based Inverse Normal Transform — vectorised
# ---------------------------------------------------------------------------
def rank_int(series: pd.Series) -> np.ndarray:
    """Blom-offset rank-INT; vectorised, no row-wise Python loop."""
    arr  = pd.to_numeric(series, errors="coerce").values.astype(float)
    mask = np.isfinite(arr)
    n    = mask.sum()
    out  = np.full(len(arr), np.nan)
    if n == 0:
        return out
    sub   = arr[mask]
    order = sub.argsort()
    ranked = np.empty(len(sub))
    ranked[order] = np.arange(1, len(sub) + 1, dtype=float)
    vals_sorted = sub[order]
    i = 0
    while i < len(vals_sorted):
        j = i + 1
        while j < len(vals_sorted) and vals_sorted[j] == vals_sorted[i]:
            j += 1
        if j > i + 1:
            ranked[order[i:j]] = ranked[order[i:j]].mean()
        i = j
    out[mask] = ndtri((ranked - 0.375) / (n + 0.25))
    return out


# ---------------------------------------------------------------------------
# D_cj computation
# ---------------------------------------------------------------------------
def compute_profiles(
    df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute signed standardised mean deviation D_cj for all clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Rank-INT transformed data; columns include `features`.
    features : list[str]
        Ordered feature names.
    labels : np.ndarray
        Cluster label per grid cell (integer 1–17).

    Returns
    -------
    pd.DataFrame
        Shape (17, len(features)); index = cluster label (sorted).
    """
    global_mean = df[features].mean()
    global_std  = df[features].std().replace(0, np.nan)

    profiles = {}
    for k, grp in df.groupby(labels):
        profiles[k] = (grp[features].mean() - global_mean) / global_std

    return pd.DataFrame(profiles).T.sort_index()


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(
    profiles: pd.DataFrame,
    importance: pd.DataFrame,
    sizes: pd.DataFrame,
    out_path: Path,
    n_labelled: int = 18,
) -> None:
    """
    Landscape clustered heatmap: clusters as rows (17), the FULL deduplicated
    feature set as columns (all ~98, unlabelled), with a real two-way Ward
    dendrogram (row dendrogram cut at k=4 to show the four cluster groups;
    column dendrogram showing feature co-variation structure) — the same
    layout convention as large-N biological heatmaps (e.g. gene-expression
    atlases), where the full matrix is shown as unlabelled blocks and only a
    handful of top features are called out with leader-line callout boxes.

    Parameters
    ----------
    profiles : pd.DataFrame
        (17 × N) D_cj matrix; index = cluster label; columns = col_labels.
    importance : pd.DataFrame
        Feature metadata; must contain 'feature' and 'eta2' columns.
    sizes : pd.DataFrame
        Cluster sizes; columns label_k17, n_cells, share_pct.
    out_path : Path
        Output SVG path.
    n_labelled : int
        Number of top-η² features to call out by name (default 18).
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import fcluster

    # Cluster colour palette (cluster 1 → index 0, …, cluster 17 → index 16)
    PALETTE_17 = [
        "#DBDBDB", "#FEEFCF", "#FCBBC5", "#F5E6C8", "#E5152C",
        "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
        "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
        "#992FA3", "#C49A3C",
    ]
    cluster_color = {i + 1: PALETTE_17[i] for i in range(17)}

    col_labels   = list(profiles.columns)
    clust_labels = list(profiles.index)   # integer cluster IDs, sorted 1–17
    Z_mat        = profiles.values        # (17 × N)
    n_feat       = len(col_labels)
    n_clust      = len(clust_labels)

    # Shorten column labels: "Group N (feat)" → "feat", singletons unchanged
    def _short(lbl: str) -> str:
        if lbl.startswith("Group ") and "(" in lbl:
            return lbl.split("(", 1)[1].rstrip(")")
        return lbl
    col_labels_short = [_short(l) for l in col_labels]

    # Map short feature codes to full, human-readable English names (from the
    # project's morphometric codebook) for the name-callout labels — the raw
    # abbreviations (e.g. "Std_NbDis") are not self-explanatory on their own.
    full_name_map = load_full_name_map()
    col_labels_full = [full_name_map.get(l, l) for l in col_labels_short]

    Z_finite = np.nan_to_num(Z_mat, nan=0.0)

    # --- Rank-based merge-height compression ---------------------------------
    # Ward linkage distances grow superlinearly near the root, so the single
    # top-level split (spanning the whole tree) otherwise dominates the
    # dendrogram's height/width while every other merge is squeezed into an
    # unreadable sliver next to the matrix. Remapping each merge height to
    # its RANK among that linkage's merge heights (evenly spaced in [0, 1])
    # guarantees uniform visual spacing between successive merges regardless
    # of the raw distance skew, while still preserving merge ORDER exactly.
    def _rank_compress(link: np.ndarray):
        heights = np.sort(np.unique(link[:, 2]))
        rank_of = {h: i for i, h in enumerate(heights)}
        n = len(heights)
        def _to_rank(d):
            d = np.asarray(d, dtype=float)
            out = np.empty_like(d)
            flat_idx = np.searchsorted(heights, d.ravel())
            out = flat_idx.reshape(d.shape) / max(n - 1, 1)
            return out
        return _to_rank

    # --- Row (cluster) dendrogram: real Ward linkage on the full 17×98
    # profile matrix. This determines both the display order AND the
    # 4-group partition (cut at k=4) shown as a coloured bar next to the
    # dendrogram — the grouping is derived from the data, not asserted.
    row_link  = linkage(Z_finite, method="ward", metric="euclidean")
    row_order = dendrogram(row_link, no_plot=True)["leaves"]
    row_group4 = fcluster(row_link, t=4, criterion="maxclust")  # len == n_clust
    row_compress = _rank_compress(row_link)

    # --- Column (feature) dendrogram: Ward linkage on the same matrix,
    # transposed — reveals which of the ~98 features co-vary across the 17
    # clusters, without needing (nonexistent) semantic dimension labels.
    col_link  = linkage(Z_finite.T, method="ward", metric="euclidean")
    col_order = dendrogram(col_link, no_plot=True)["leaves"]
    col_compress = _rank_compress(col_link)

    Z_ord        = Z_finite[np.ix_(row_order, col_order)]
    clust_ord    = [clust_labels[i] for i in row_order]
    group_ord    = [int(row_group4[i]) for i in row_order]
    labels_ord   = [col_labels_short[i] for i in col_order]
    labels_ord_full = [col_labels_full[i] for i in col_order]
    n_show       = n_feat

    # --- Figure layout (ORIGINAL landscape orientation — clusters as rows,
    # features as columns). This is the exact, already-vetted layout; the
    # portrait version the user prefers is produced by rotating this whole
    # rendered figure 90° afterwards (see bottom of this function), NOT by
    # re-deriving a new set of proportions. ---
    NAME_H_PER_CHAR = 0.062   # inches of vertical space per rotated character
    max_name_len = max(len(s) for s in col_labels_full)
    fig_w  = 13.0
    mat_h  = n_clust * 0.14
    names_h = max_name_len * NAME_H_PER_CHAR
    fig_h  = max(2.2, mat_h + 0.55 + names_h)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    L_MARGIN, R_MARGIN   = 0.005, 0.005
    NAMES_H              = names_h / fig_h
    T_MARGIN             = 0.06 / fig_h
    ROW_DENDRO_W         = 0.050
    STRIP_GAP            = 0.006     # small visual gap between adjacent strips
    GROUP_BAR_W          = 0.010
    DOT_AX_W             = 0.022
    CBAR_W               = 0.018
    CBAR_GAP             = 0.012
    COL_DENDRO_H         = (0.85 * 2 / 3) / fig_h
    B_MARGIN             = COL_DENDRO_H + 0.06 / fig_h

    mat_bottom = B_MARGIN
    mat_height = 1 - T_MARGIN - NAMES_H - B_MARGIN
    mat_left   = (L_MARGIN + ROW_DENDRO_W + STRIP_GAP + GROUP_BAR_W
                  + STRIP_GAP + DOT_AX_W + STRIP_GAP)
    mat_width  = 1 - R_MARGIN - CBAR_GAP - CBAR_W - mat_left

    ax_heat = fig.add_axes([mat_left, mat_bottom, mat_width, mat_height])
    ax_cb   = fig.add_axes([
        mat_left + mat_width + CBAR_GAP, mat_bottom, CBAR_W, mat_height,
    ])

    vmax = np.percentile(np.abs(Z_ord[np.isfinite(Z_ord)]), 98)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_div", ["#4756A0", "#FFFFFF", "#C8292A"], N=256,
    )
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    x_edges = np.arange(n_show + 1) - 0.5
    y_edges = np.arange(n_clust, -1, -1) - 0.5   # n_clust-0.5 .. -0.5, descending
    ax_heat.pcolormesh(
        x_edges, y_edges, Z_ord,
        cmap=cmap, norm=norm, shading="flat",
        edgecolors="none", linewidth=0,
    )
    ax_heat.tick_params(axis="both", length=0, labelleft=False, labelbottom=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_ylim(-0.5, n_clust - 0.5)
    ax_heat.set_xlim(-0.5, n_show - 0.5)

    row_dendro_left = L_MARGIN
    ax_row_dendro = fig.add_axes([
        row_dendro_left, mat_bottom, ROW_DENDRO_W, mat_height,
    ])
    row_dendro = dendrogram(row_link, no_plot=True, orientation="left")
    def _leaf_to_row(y):
        return (np.asarray(y) / 10.0) - 0.5
    for xs, ys in zip(row_dendro["icoord"], row_dendro["dcoord"]):
        ax_row_dendro.plot(
            [-v for v in row_compress(ys)], _leaf_to_row(xs),
            color="#555555", linewidth=1.1, solid_capstyle="round",
        )
    row_dcoord_max = 1.0  # rank-compressed heights are always in [0, 1]
    ax_row_dendro.set_ylim(n_clust - 0.5, -0.5)
    ax_row_dendro.set_xlim(-row_dcoord_max * 1.08, 0)
    ax_row_dendro.axis("off")

    dot_left = row_dendro_left + ROW_DENDRO_W + STRIP_GAP
    ax_dot = fig.add_axes([dot_left, mat_bottom, DOT_AX_W, mat_height])
    ax_dot.set_xlim(0, 1)
    ax_dot.set_ylim(n_clust - 0.5, -0.5)
    ax_dot.axis("off")
    for row_i, cid in enumerate(clust_ord):
        col = cluster_color.get(int(cid), "#888888")
        # rotation=90 (+ rotation_mode='anchor'): this text is NOT pre-rotated
        # in isolation — it must cancel out the whole-figure 90° rotation
        # applied in _rotate_90cw() below, so it reads horizontally (not
        # sideways) in the final output, exactly like the feature-name
        # labels already do.
        ax_dot.text(0.55, row_i, str(int(cid)),
                    ha="right", va="center", fontsize=7.5, clip_on=False,
                    rotation=90, rotation_mode="anchor")
        ax_dot.add_patch(Rectangle(
            (0.68, row_i - 0.5), 0.32, 1.0,
            facecolor=col, linewidth=0, clip_on=False, zorder=5,
        ))

    GROUP4_COLORS = {1: "#8DA0CB", 2: "#FC8D62", 3: "#66C2A5", 4: "#E78AC3"}
    group_left = dot_left + DOT_AX_W + STRIP_GAP
    ax_group = fig.add_axes([group_left, mat_bottom, GROUP_BAR_W, mat_height])
    ax_group.set_xlim(0, 1)
    ax_group.set_ylim(n_clust - 0.5, -0.5)
    ax_group.axis("off")
    for row_i, g in enumerate(group_ord):
        ax_group.add_patch(Rectangle(
            (0, row_i - 0.5), 1, 1.0,
            facecolor=GROUP4_COLORS.get(g, "#AAAAAA"), linewidth=0,
            clip_on=False,
        ))
    boundaries = [0]
    for i in range(1, n_clust):
        if group_ord[i] != group_ord[i - 1]:
            boundaries.append(i)
    boundaries.append(n_clust)
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        g = group_ord[b0]
        ax_row_dendro.text(
            -row_dcoord_max * 1.12, (b0 + b1 - 1) / 2, f"G{g}",
            ha="right", va="center", fontsize=7.5, fontweight="bold",
            color=GROUP4_COLORS.get(g, "#AAAAAA"), clip_on=False,
            rotation=90, rotation_mode="anchor",
        )

    ax_col_dendro = fig.add_axes([mat_left, 0.0, mat_width, COL_DENDRO_H])
    col_dendro = dendrogram(col_link, no_plot=True)
    for xs, ys in zip(col_dendro["icoord"], col_dendro["dcoord"]):
        ax_col_dendro.plot(
            (np.asarray(xs) / 10.0) - 0.5, -col_compress(ys),
            color="#555555", linewidth=1.1, solid_capstyle="round",
        )
    ax_col_dendro.set_xlim(-0.5, n_show - 0.5)
    ax_col_dendro.set_ylim(-1.05, 0)   # 0 = leaves (top, at matrix), -1.05 = root (bottom)
    ax_col_dendro.axis("off")

    names_bottom = mat_bottom + mat_height
    ax_names = fig.add_axes([mat_left, names_bottom, mat_width, NAMES_H])
    ax_names.set_xlim(-0.5, n_show - 0.5)
    ax_names.set_ylim(0, 1)
    ax_names.axis("off")
    for i in range(n_show):
        # va="center" (not "bottom"): keeps each label's anchor exactly on
        # its row regardless of the label's own text length — va="bottom"
        # let longer/shorter names drift to visibly different offsets from
        # their row once the whole figure is rotated 90° (verified via an
        # isolated before/after rotation test).
        ax_names.text(
            i, 0.02, labels_ord_full[i],
            rotation=90, ha="left", va="center", fontsize=7.2,
            rotation_mode="anchor", clip_on=False, zorder=2,
        )

    cbar_vals = np.linspace(vmax, -vmax, 256).reshape(-1, 1)
    ax_cb.imshow(
        cbar_vals, cmap=cmap, norm=norm, aspect="auto",
        extent=[0, 1, -vmax, vmax],
    )
    ax_cb.set_xlim(0, 1)
    ax_cb.set_ylim(-vmax, vmax)
    ax_cb.set_xticks([])
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_label_position("right")
    # rotation=90 on both the tick labels and the axis label: like the
    # cluster-number/G-group labels above, these are NOT pre-rotated to read
    # sideways — they must cancel out the whole-figure 90° rotation applied
    # in _rotate_90cw() below, so they read horizontally in the final output.
    ax_cb.set_ylabel(
        r"Standardised deviation ($D_{cj}$)",
        fontsize=9, labelpad=5, rotation=90, va="center",
    )
    ax_cb.tick_params(axis="y", labelsize=8, length=0)
    for lbl in ax_cb.get_yticklabels():
        lbl.set_rotation(90)
        lbl.set_rotation_mode("anchor")
        lbl.set_verticalalignment("center")
    for spine in ax_cb.spines.values():
        spine.set_visible(False)

    # --- Save landscape SVG/PNG, then rotate the WHOLE rendered figure 90°
    # clockwise as a rigid transform (not a redesign): this is exactly what
    # the user did by hand to the PNG, applied here non-destructively so
    # text stays fully editable in the SVG (wrapped in a <g transform=...>,
    # not converted to outlines/paths) and every proportion already vetted
    # in the landscape version (row/column sizing, dendrogram compression,
    # font sizes, gaps) is preserved exactly, unchanged. ---
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _rotate_90cw(out_path, png_path)
    logging.info("Heatmap saved: %s (+ .png)", out_path)


def _rotate_90cw(svg_path: Path, png_path: Path) -> None:
    """
    Rotate a saved SVG+PNG figure 90 degrees clockwise IN PLACE.

    PNG: a plain pixel rotation (lossless, square pixels).
    SVG: wraps the existing content in a <g transform="translate(H,0)
    rotate(90)"> and swaps the root width/height/viewBox — a rigid
    transform applied on top of matplotlib's own output, so every
    <text> element remains real, selectable/editable text in Illustrator
    (not converted to outlines), just repositioned as a whole.
    """
    import re
    from PIL import Image

    with Image.open(png_path) as im:
        im.rotate(-90, expand=True).save(png_path)

    svg_text = svg_path.read_text()
    m = re.search(
        r'<svg\b([^>]*?)\bheight="([\d.]+)pt"([^>]*?)\bwidth="([\d.]+)pt"',
        svg_text,
    )
    if m is None:
        m = re.search(
            r'<svg\b([^>]*?)\bwidth="([\d.]+)pt"([^>]*?)\bheight="([\d.]+)pt"',
            svg_text,
        )
        if m is None:
            logging.warning("Could not parse SVG root size for rotation; SVG left unrotated.")
            return
        pre, w_str, mid, h_str = m.groups()
        old_attr_order = "wh"
    else:
        pre, h_str, mid, w_str = m.groups()
        old_attr_order = "hw"
    w_pt, h_pt = float(w_str), float(h_str)

    old_svg_tag = m.group(0)
    if old_attr_order == "hw":
        new_svg_tag = f'<svg{pre}height="{w_pt:g}pt"{mid}width="{h_pt:g}pt"'
    else:
        new_svg_tag = f'<svg{pre}width="{h_pt:g}pt"{mid}height="{w_pt:g}pt"'
    svg_text = svg_text.replace(old_svg_tag, new_svg_tag, 1)

    vb_m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    if vb_m:
        svg_text = svg_text.replace(
            vb_m.group(0), f'viewBox="0 0 {h_pt:g} {w_pt:g}"', 1,
        )

    # Wrap the whole content (everything between the <svg ...> open tag and
    # </svg>) in one rotating group. matplotlib's SVG has a <defs> block that
    # must NOT be rotated (it holds reusable path/font definitions, not
    # visible geometry) — only the visible content group(s) after it need
    # the transform, so we insert the wrapper right after </defs> if present,
    # else right after the opening <svg ...> tag.
    close_tag_idx = svg_text.rindex("</svg>")
    open_tag_end = svg_text.index(">", svg_text.index(new_svg_tag)) + 1
    defs_end = svg_text.find("</defs>", open_tag_end)
    insert_at = (defs_end + len("</defs>")) if defs_end != -1 else open_tag_end
    wrap_open = f'<g transform="translate({h_pt:g},0) rotate(90)">'
    svg_text = (
        svg_text[:insert_at]
        + wrap_open
        + svg_text[insert_at:close_tag_idx]
        + "</g>"
        + svg_text[close_tag_idx:]
    )
    svg_path.write_text(svg_text)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(features: list[str]) -> str:
    mtime = MODEL_INPUT.stat().st_mtime
    payload = json.dumps({"mtime": mtime, "features": sorted(features)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_int_cache(features: list[str]) -> pd.DataFrame | None:
    """Return cached rank-INT DataFrame (features + label_k17) if valid, else None."""
    meta = CACHE_DIR / "rank_int_top.meta.json"
    if not INT_CACHE.exists() or not meta.exists():
        return None
    if json.loads(meta.read_text())["key"] != _cache_key(features):
        return None
    return pd.read_parquet(INT_CACHE)


def save_int_cache(features: list[str], df_int: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_int.to_parquet(INT_CACHE, index=False)
    meta = CACHE_DIR / "rank_int_top.meta.json"
    meta.write_text(json.dumps({"key": _cache_key(features)}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(top_n: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Load feature importance ranking ---
    # Reduce to one representative per collinear group (intragroup_rank == 1),
    # then take the top_n groups by η² (None = all).  Singletons have
    # intragroup_rank == 1 by definition, so they are included automatically.
    importance_all = pd.read_csv(IMPORTANCE_CSV)
    importance_sorted = (
        importance_all[importance_all["intragroup_rank"] == 1]
        .sort_values("eta2", ascending=False)
        .reset_index(drop=True)
    )
    if top_n is not None:
        importance_sorted = importance_sorted.head(top_n)
    importance = importance_sorted
    features = importance["feature"].tolist()
    logging.info(
        "Using top %d representative features (one per group/singleton) for cluster profiles.",
        len(features),
    )

    # Build display labels: "Group N (feat)" for multi-feature groups, feature name for singletons.
    # dimension_label column already encodes "Group N (feat1, feat2, feat3)"; extract "Group N (top1)".
    def _col_label(row: pd.Series) -> str:
        lbl = row["dimension_label"]
        if lbl == "Singleton":
            return row["feature"]
        # lbl is "Group N (feat1, feat2[, feat3])" — rewrite to "Group N (top1_feat)"
        group_part = lbl.split("(")[0].strip()   # "Group N"
        return f"{group_part} ({row['feature']})"

    importance["col_label"] = importance.apply(_col_label, axis=1)

    # --- Load and transform data (with cache) ---
    df_int = load_int_cache(features)
    if df_int is not None:
        logging.info("Rank-INT data loaded from cache (%s).", INT_CACHE)
        labels = df_int["label_k17"].values
    else:
        logging.info("Cache miss — loading model input ...")
        df_raw = pd.read_parquet(MODEL_INPUT, columns=features + ["label_k17"])
        labels = df_raw["label_k17"].values

        logging.info("Applying rank-INT (parallel, %d workers) ...", N_JOBS)
        def _rint(f):
            return f, rank_int(df_raw[f])
        with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
            rint_results = dict(ex.map(_rint, features))
        df_int = pd.DataFrame(rint_results)
        df_int["label_k17"] = labels
        save_int_cache(features, df_int)
        logging.info("Rank-INT data cached to %s.", INT_CACHE)
        df_raw = None  # free memory

    # --- Cluster sizes ---
    sizes = (
        pd.DataFrame({"label_k17": labels})
        .groupby("label_k17").size()
        .reset_index(name="n_cells")
    )
    sizes["share_pct"] = (sizes["n_cells"] / sizes["n_cells"].sum() * 100).round(2)
    sizes.to_csv(OUT_TABLES / "cluster_sizes.csv", index=False)
    logging.info("Cluster sizes saved.")

    # --- D_cj profiles ---
    logging.info("Computing D_cj profiles ...")
    profiles = compute_profiles(df_int, features, labels)
    # Rename columns to display labels before saving
    profiles.columns = importance.set_index("feature").loc[features, "col_label"].tolist()

    # Drop features with zero variance across Africa (all-zero land-cover
    # classes such as snow/moss/needleleaf forest that never occur in the
    # African urban context): std() == 0 makes compute_profiles' division
    # produce NaN, which would otherwise render as an uninformative blank
    # (white) column in the heatmap.
    zero_var_cols = profiles.columns[profiles.isna().any()]
    if len(zero_var_cols):
        logging.info("Dropping %d zero-variance feature(s) (all-NaN column): %s",
                     len(zero_var_cols), list(zero_var_cols))
        profiles = profiles.drop(columns=zero_var_cols)

    profiles.to_csv(OUT_TABLES / "cluster_profiles.csv")
    logging.info("Profiles saved: %s", OUT_TABLES / "cluster_profiles.csv")

    # --- Heatmap ---
    plot_heatmap(
        profiles,
        importance[importance["feature"].isin(features)],
        sizes,
        out_path=OUT_FIGURES / "cluster_profiles_heatmap.svg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and visualise k=17 cluster profiles (D_cj)."
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Maximum number of features (by η²) to include (default: all).",
    )
    args = parser.parse_args()
    main(top_n=args.top)
