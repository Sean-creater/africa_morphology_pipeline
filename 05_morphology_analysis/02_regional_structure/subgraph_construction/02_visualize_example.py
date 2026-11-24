"""
02_visualize_example.py
=======================
Visualize two panels for a single example city:
  Left  — spatial cell map (hex grid coloured by urban form type)
  Right — 17-node type interaction graph
           node size ∝ type share (p_i)
           edge width ∝ |log-enrichment A_ij|
           edge alpha ∝ |log-enrichment A_ij|  (weak edges fade out)
           edge style: solid = enriched (A>0), dashed = depleted (A<0)
           edge colour: gradient between the two endpoint node colours
           self-loops: arc drawn beside each node, same encoding

Usage:
    python 02_visualize_example.py --city Accra --country Ghana
"""

import argparse
import pickle
import logging
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# Register a font whose internal name is "Arial" so matplotlib's mathtext
# engine (which resolves mathtext.rm/it/bf by actually looking up a font,
# unlike plain text under svg.fonttype=none) can find one — otherwise it
# silently falls back to DejaVu Sans for equation glyphs in the SVG output.
_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    logging.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":    8,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm":       "Arial",
    "mathtext.it":       "Arial:italic",
    "mathtext.bf":       "Arial:bold",
})

ROOT        = Path(__file__).parent
# SHP geometry + attributes per city, published by
# ../../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT    = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../02_morphology_metrics/outputs/SHP_cleaned",
))
LABELS_PATH = (ROOT.parent.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")
CACHE_DIR   = ROOT / "outputs" / "cache"
FIG_DIR     = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 17 qualitative colours — matches embedding overview (cluster 1 → index 0)
PALETTE_17 = [
    "#DBDBDB",  # C1
    "#FEEFCF",  # C2
    "#FCBBC5",  # C3
    "#F5E6C8",  # C4
    "#E5152C",  # C5
    "#FFFF4C",  # C6
    "#5BA04A",  # C7
    "#FBB732",  # C8
    "#6F391F",  # C9
    "#B0E1E7",  # C10
    "#7B67EE",  # C11
    "#E89679",  # C12
    "#58C4EC",  # C13
    "#828F9B",  # C14
    "#1C1C5F",  # C15
    "#992FA3",  # C16
    "#C49A3C",  # C17
]
TYPE_COLORS = {i + 1: PALETTE_17[i] for i in range(17)}

N_SEG = 20    # gradient segments per edge (higher = smoother, fewer artefacts)
# Each segment becomes its own <path> in SVG (LineCollection can't merge
# differently-coloured segments into one path) — with up to C(17,2)=136
# edges, 120 segments/edge produced ~14k paths, far too many vector objects
# for Illustrator to load interactively. 20 segments is still a visually
# smooth gradient at this line width, at ~6x fewer path elements.


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])


def load_graph(country: str, city: str, kind: str) -> nx.Graph:
    safe = f"{country}_{city}".replace(" ", "_")
    path = CACHE_DIR / f"{safe}_{kind}_graph.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}. "
                                "Run 01_build_city_graphs.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gdf(country: str, city: str, labels: pd.DataFrame) -> gpd.GeoDataFrame:
    shp_path = SHP_ROOT / country / city
    shps = list(shp_path.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No SHP in {shp_path}")
    gdf = gpd.read_file(shps[0])
    city_labels = labels[labels["city"] == city][["grid_id", "label_k17"]]
    gdf = gdf.merge(city_labels, left_on="GridID", right_on="grid_id", how="inner")
    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel 1 — spatial cell map
# ---------------------------------------------------------------------------

def _pick_focal(gdf, w_queen):
    """Pick a focal cell index with the most diverse neighbour types."""
    best_idx, best_div = 0, -1
    for idx, nbrs in w_queen.neighbors.items():
        if not nbrs:
            continue
        nb_types = set(int(gdf.iloc[j]["label_k17"]) for j in nbrs)
        if len(nb_types) > best_div:
            best_div = len(nb_types)
            best_idx = idx
    return best_idx


def plot_cell_map(ax, gdf: gpd.GeoDataFrame, city: str, focal_idx: int = 0):
    gdf["color"] = gdf["label_k17"].map(TYPE_COLORS)
    # rasterized=True: a city can have tens of thousands of hex cells, each
    # its own vector path — fully vectorising this one dense panel makes the
    # SVG far too heavy for Adobe Illustrator to open/edit interactively
    # (object count, not file size, is what stalls it). Everything else in
    # this figure (titles, legend, the type-interaction graph) stays vector.
    gdf.plot(ax=ax, color=gdf["color"].tolist(), linewidth=0,
             edgecolor="none", rasterized=True)
    # Highlight the focal cell and draw a zoom rectangle around it
    focal_geom = gdf.iloc[focal_idx].geometry
    gdf.iloc[[focal_idx]].plot(ax=ax, color="none",
                               edgecolor="black", linewidth=2.5, zorder=5)
    b = focal_geom.bounds
    pad = (b[2] - b[0]) * 3.5
    rect = mpatches.FancyBboxPatch(
        (b[0] - pad, b[1] - pad),
        (b[2] - b[0]) + 2 * pad, (b[3] - b[1]) + 2 * pad,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor="black", facecolor="none", zorder=6)
    ax.add_patch(rect)
    ax.set_title(f"{city} — spatial layout", fontsize=11)
    ax.axis("off")

    # --- Scale bar ---
    # The source SHP is in a UTM projection (metres), so 1 data unit = 1 m
    # exactly here — no lat/lon approximation needed. Anchor point fixed in
    # axes-fraction (bottom-left, inside the plotted extent) so it lands in
    # the same relative spot regardless of this city's actual footprint
    # size/shape; bar length is then a real, exact distance in metres.
    bar_len_m = 10_000  # 10 km, round number
    tick_h_frac = 0.012
    inv = ax.transData.inverted()
    x0, y0 = inv.transform(ax.transAxes.transform((0.04, 0.06)))
    x1 = x0 + bar_len_m
    xr0, yr0 = inv.transform(ax.transAxes.transform((0, 0)))
    xr1, yr1 = inv.transform(ax.transAxes.transform((1, 1)))
    tick_h = tick_h_frac * (yr1 - yr0)
    ax.plot([x0, x1], [y0, y0], color="#222222", linewidth=1.0,
            solid_capstyle="butt", zorder=11, clip_on=False)
    for x in (x0, x1):
        ax.plot([x, x], [y0 - tick_h / 2, y0 + tick_h / 2],
                color="#222222", linewidth=1.0, zorder=11, clip_on=False)
    ax.text((x0 + x1) / 2, y0 + tick_h / 2 + 0.01 * (yr1 - yr0),
            f"{bar_len_m // 1000} km", ha="center", va="bottom",
            fontsize=6, fontfamily="Arial", color="#222222",
            zorder=11, clip_on=False)


def plot_adjacency_schematic(ax, gdf: gpd.GeoDataFrame, focal_idx: int = 0):
    """Show one focal hex cell with its Queen neighbours."""
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, silence_warnings=True)

    neighbour_idxs = list(w.neighbors[focal_idx])

    focal      = gdf.iloc[[focal_idx]]
    neighbours = gdf.iloc[neighbour_idxs]

    focal_color = TYPE_COLORS.get(int(focal["label_k17"].iloc[0]), "#888888")

    # Draw neighbours first, then focal on top
    nb_colors = [TYPE_COLORS.get(int(r["label_k17"]), "#888888")
                 for _, r in neighbours.iterrows()]
    neighbours.plot(ax=ax, color=nb_colors, edgecolor="white",
                    linewidth=0.8, alpha=0.9)
    focal.plot(ax=ax, color=focal_color, edgecolor="black", linewidth=2.0)

    # Label all cells with cluster number (no background box)
    def _label(geom, label):
        cx = geom.centroid.x
        cy = geom.centroid.y
        ax.text(cx, cy, f"C{label}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black")

    _label(focal.geometry.iloc[0], int(focal["label_k17"].iloc[0]))
    for _, row in neighbours.iterrows():
        _label(row.geometry, int(row["label_k17"]))

    ax.set_title("Queen adjacency\n(focal cell + neighbours)", fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel 2 — 17-node type interaction graph
# ---------------------------------------------------------------------------

def _bezier_gradient(ax, p0, p1, c0, c1, width, alpha, linestyle,
                     curvature=0.25, zorder=2):
    """Quadratic Bezier with colour gradient, drawn centre-to-centre.

    The node circle (drawn on top, zorder=5) naturally clips the line ends,
    so the gradient appears to emerge from the node colour — consistent with
    how force-directed viewers (e.g. Gephi) render thick undirected edges.
    """
    vec  = p1 - p0
    dist = np.linalg.norm(vec)
    if dist < 1e-9:
        return
    unit = vec / dist
    mid  = (p0 + p1) / 2
    perp = np.array([-unit[1], unit[0]])
    ctrl = mid + perp * curvature * dist

    t_vals = np.linspace(0, 1, N_SEG + 1)
    pts = (np.outer((1 - t_vals)**2, p0)
           + np.outer(2 * (1 - t_vals) * t_vals, ctrl)
           + np.outer(t_vals**2, p1))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t_mid = (t_vals[:-1] + t_vals[1:]) / 2
    colors = c0 * (1 - t_mid[:, None]) + c1 * t_mid[:, None]
    colors_rgba = np.hstack([colors, np.full((len(colors), 1), alpha)])
    lc = LineCollection(segs, colors=colors_rgba, linewidth=width,
                        capstyle="round", linestyle=linestyle, zorder=zorder)
    ax.add_collection(lc)


def _draw_selfloop(ax, cx, cy, node_r, outward_dir, w, max_self_w):
    """Draw a self-loop as a small circle tangent to the node boundary.

    The loop sits just outside the node in the radial (outward) direction.
    Width and opacity encode |A_ii| relative to the max self-loop weight.
    """
    aw = abs(w)
    t  = aw / max_self_w if max_self_w > 0 else 0.0
    # loop radius scales with self-weight but stays smaller than node
    loop_r = node_r * (0.45 + t * 0.55)
    width  = 0.8 + t * 3.5
    alpha  = 0.50 + t * 0.45
    color  = np.array(hex_to_rgb(TYPE_COLORS.get(node_r, "#888888")))  # placeholder; overridden below
    # centre of the loop: just outside the node boundary
    lc_x = cx + outward_dir[0] * (node_r + loop_r)
    lc_y = cy + outward_dir[1] * (node_r + loop_r)

    theta = np.linspace(0, 2 * np.pi, N_SEG + 1)
    pts   = np.column_stack([lc_x + loop_r * np.cos(theta),
                              lc_y + loop_r * np.sin(theta)])
    segs  = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs,
                        color=(*color, alpha),   # uniform colour + alpha
                        linewidth=width, capstyle="round",
                        linestyle="solid" if w >= 0 else "dashed",
                        zorder=3)
    ax.add_collection(lc)


def _radial_layout(nodes, p_attr, w_attr):
    """Three-ring radial layout.

    Centrality = 0.6 × type_share + 0.4 × normalised_weighted_degree.
    Nodes are split into inner / mid / outer rings so nearby ranks occupy
    the same ring radius with well-spaced angles — avoids crowding at centre.

    Ring radii and capacities (tuned for n≤17):
      inner : r = 0.25,  up to 3 nodes
      mid   : r = 0.60,  up to 6 nodes
      outer : r = 1.00,  remainder
    """
    # Weighted degree (sum |A_ij|, j≠i)
    wdeg = {n: 0.0 for n in nodes}
    for (u, v), w in w_attr.items():
        if u == v:
            continue
        if u in wdeg:
            wdeg[u] += abs(w)
        if v in wdeg:
            wdeg[v] += abs(w)

    max_p    = max(p_attr.values()) if p_attr else 1.0
    max_wdeg = max(wdeg.values())   if wdeg   else 1.0
    centrality = {n: 0.6 * (p_attr.get(n, 0) / max_p)
                     + 0.4 * (wdeg[n] / max_wdeg)
                  for n in nodes}

    sorted_nodes = sorted(nodes, key=lambda n: -centrality[n])

    # Assign to rings
    ring_caps   = [3, 6]          # inner, mid capacities; rest go outer
    ring_radii  = [0.42, 0.72, 1.08]   # wider inner ring so large nodes don't overlap
    ring_offset = [0.0,  np.pi / 8, 0.0]  # angle offset per ring to stagger

    rings = []
    remaining = list(sorted_nodes)
    for cap in ring_caps:
        rings.append(remaining[:cap])
        remaining = remaining[cap:]
    rings.append(remaining)       # outer ring

    radii_dict  = {}
    pos         = {}
    for ri, (ring, r, off) in enumerate(zip(rings, ring_radii, ring_offset)):
        k = len(ring)
        if k == 0:
            continue
        for j, nd in enumerate(ring):
            ang = off + np.pi / 2 + 2 * np.pi * j / k
            radii_dict[nd] = r
            pos[nd] = np.array([r * np.cos(ang), r * np.sin(ang)])

    return pos, radii_dict


def plot_type_graph(ax, G_type: nx.Graph, city: str):
    p_attr = nx.get_node_attributes(G_type, "p")
    w_attr = nx.get_edge_attributes(G_type, "weight")
    nodes  = sorted(G_type.nodes())

    # --- Layout ---
    pos, radii = _radial_layout(nodes, p_attr, w_attr)

    # --- Scale for edge encoding ---
    regular  = [(u, v) for u, v in G_type.edges() if u != v]
    reg_abs  = [abs(w_attr.get((u, v), w_attr.get((v, u), 0.0)))
                for u, v in regular]
    # Use 90th-percentile cap so a handful of extreme edges don't compress
    # everything else — mid-strength edges (like C11-C8) remain visible.
    max_reg  = float(np.percentile(reg_abs, 90)) if reg_abs else 1.0

    self_abs = [abs(w_attr[(u, v)]) for u, v in w_attr if u == v and u in nodes]
    max_self = max(self_abs) if self_abs else 1.0

    # Node display radii (visual circle size ∝ type share)
    max_p = max(p_attr.values()) if p_attr else 1.0
    node_radii = {n: 0.03 + (p_attr.get(n, 0) / max_p) * 0.08 for n in nodes}

    # --- 1. Draw regular edges beneath everything ---
    # A_ij mapped linearly from [-max_reg, +max_reg] → [min_w, max_w].
    # Negative A (spatial avoidance) → thin & faint; positive → thick & opaque.
    # Single channel, no dashes, no extra dimensions.
    W_MIN, W_MAX = 0.2, 3.0
    A_MIN, A_MAX = 0.18, 0.90
    for u, v in regular:
        w   = w_attr.get((u, v), w_attr.get((v, u), 0.0))
        t   = np.clip((w + max_reg) / (2 * max_reg), 0.0, 1.0)
        width = W_MIN + t * (W_MAX - W_MIN)
        alpha = A_MIN + t * (A_MAX - A_MIN)
        _bezier_gradient(ax,
                         pos[u], pos[v],
                         hex_to_rgb(TYPE_COLORS.get(u, "#888888")),
                         hex_to_rgb(TYPE_COLORS.get(v, "#888888")),
                         width, alpha, "solid", zorder=2)

    # --- 2. Draw self-loops (zorder=3, above edges but below nodes) ---
    # Loop radius = node_r (visually matches node size).
    # Loop linewidth encodes |A_ii| so spatial self-clustering strength is readable.
    for (u, v), w in w_attr.items():
        if u != v or u not in nodes:
            continue
        cx, cy = pos[u]
        r = radii[u]
        outward = np.array([cx, cy]) / r if r > 1e-9 else np.array([1.0, 0.0])
        nr     = node_radii[u]
        loop_r = nr
        t      = np.clip((w + max_self) / (2 * max_self), 0.0, 1.0)
        width  = W_MIN + t * (W_MAX - W_MIN)
        alpha  = A_MIN + t * (A_MAX - A_MIN)
        color  = np.array(hex_to_rgb(TYPE_COLORS.get(u, "#888888")))
        lc_x   = cx + outward[0] * (nr + loop_r)
        lc_y   = cy + outward[1] * (nr + loop_r)
        theta  = np.linspace(0, 2 * np.pi, N_SEG + 1)
        pts    = np.column_stack([lc_x + loop_r * np.cos(theta),
                                  lc_y + loop_r * np.sin(theta)])
        segs   = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, color=(*color, alpha), linewidth=width,
                            capstyle="round", linestyle="solid", zorder=3)
        ax.add_collection(lc)

    # --- 3. Draw nodes on top ---
    for n in nodes:
        cx, cy = pos[n]
        r  = node_radii[n]
        nc = TYPE_COLORS.get(n, "#888888")
        circle = mpatches.Circle((cx, cy), radius=r, color=nc,
                                 zorder=5, linewidth=0.7, edgecolor="#333333")
        ax.add_patch(circle)
        ax.text(cx, cy, str(n), ha="center", va="center",
                fontsize=7, fontweight="bold", color="k", zorder=6)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_title(
        f"{city} — morphological type interaction graph\n"
        "node size = type share  ·  "
        "edge & loop width/opacity = log-enrichment A$_{ij}$  "
        "(thick = co-occur, thin = avoid)",
        fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",    default="Dar es Salaam")
    parser.add_argument("--country", default="Tanzania")
    args = parser.parse_args()

    labels = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    logging.info("Loading graphs for %s / %s ...", args.country, args.city)
    G_type = load_graph(args.country, args.city, "type")
    gdf    = load_gdf(args.country, args.city, labels)

    # figsize halved (26,9) -> (13,4.5); fontsize=... is in points (an
    # absolute physical unit, independent of figsize), so every existing
    # fontsize= call below keeps the same absolute size on the page while
    # the canvas itself shrinks 2x — text ends up proportionally larger
    # relative to the smaller figure, as requested.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              gridspec_kw={"width_ratios": [1, 2.2, 2.5]})
    fig.suptitle(f"{args.city}, {args.country} — urban morphological type subgraph",
                 fontsize=13)

    from libpysal.weights import Queen as _Queen
    _w = _Queen.from_dataframe(gdf, silence_warnings=True)
    focal_idx = _pick_focal(gdf, _w)

    plot_adjacency_schematic(axes[0], gdf, focal_idx)
    plot_cell_map  (axes[1], gdf, args.city, focal_idx)
    plot_type_graph(axes[2], G_type, args.city)

    # --- Shared 17-type colour legend (applies to all 3 panels: the same
    # TYPE_COLORS mapping colours the schematic hexagons, the spatial
    # layout, and the graph nodes) ---
    legend_handles = [
        mpatches.Patch(facecolor=TYPE_COLORS[t], edgecolor="#333333",
                       linewidth=0.4, label=f"Type {t}")
        for t in range(1, 18)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=17,
        bbox_to_anchor=(0.5, -0.02), frameon=False,
        fontsize=6, handlelength=1.0, handleheight=1.0,
        columnspacing=0.8, handletextpad=0.4,
    )

    fig.tight_layout()
    safe_city   = args.city.replace(" ", "_")
    out         = FIG_DIR / f"{args.country}_{safe_city}_subgraph_viz.svg"
    out_png     = out.with_suffix(".png")
    out_png_tr  = FIG_DIR / f"{args.country}_{safe_city}_subgraph_viz_transparent.png"
    fig.savefig(out,        dpi=200, bbox_inches="tight")
    fig.savefig(out_png,    dpi=200, bbox_inches="tight")
    fig.savefig(out_png_tr, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logging.info("Saved: %s (+ .png + _transparent.png)", out)


if __name__ == "__main__":
    main()
