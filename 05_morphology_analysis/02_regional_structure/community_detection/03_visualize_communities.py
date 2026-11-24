"""
03_visualize_communities.py
===========================
Publication-quality map of Leiden community membership across African cities.

Layout
------
One panel: Africa political boundary backdrop, city points coloured by
community, legend outside right. (Edges between cities were dropped —
their interpretation was not meaningful for the analysis.)

Inputs
------
outputs/city_communities.parquet
    Leiden partition (k=100, α=0.5): city, country, n_cells, community.
../regression/input/city.xlsx
    City coordinates: city, country, latitude, longitude.
Africa country boundary shapefile (BOUNDARY below; EPSG:4326).

Outputs
-------
outputs/plot_data.parquet           merged city+community+coordinate table (cached)
outputs/figures/community_map.pdf
outputs/figures/community_map.png   (300 dpi)

Tuning
------
All visual parameters are defined as constants at the top of the file
(POINT_SCALE, POINT_MIN_S, POINT_MAX_S, LAND_COLOR, BORDER_COLOR,
PALETTE_9).  Adjust and re-run; the merged coordinate table is cached in
outputs/plot_data.parquet and reloaded automatically, so only the drawing
step repeats.

Usage
-----
    python 03_visualize_communities.py
"""

import logging
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent
OUT_FIG  = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

COMM_PATH  = ROOT / "outputs" / "city_communities.parquet"
CITY_XLSX  = ROOT.parent / "regression" / "input" / "city.xlsx"
# African country boundary polygons (e.g. derived from Africapolis/GADM
# administrative boundaries). Override with AFRICA_BOUNDARY_SHP to point
# at your own copy.
BOUNDARY   = Path(os.environ.get(
    "AFRICA_BOUNDARY_SHP",
    "/path/to/africa_boundaries.shp",
))

# ---------------------------------------------------------------------------
# *** VISUAL TUNING PARAMETERS — edit here, re-run to update figure ***
# ---------------------------------------------------------------------------

# Backdrop
LAND_COLOR   = "white"    # polygon fill  (e.g. "white", "#f5f5f5", "#f0ede8")
BORDER_COLOR = "#aaaaaa"  # country border colour
BORDER_LW    = 0.35       # country border line width (pt)

# Points: size proportional to sqrt(n_cells)
POINT_SCALE   = 0.196  # base multiplier  (×1.4 from 0.14)
POINT_ALPHA   = 0.80
POINT_MIN_S   = 5.6    # minimum scatter marker size (pt²)  (×1.4)
POINT_MAX_S   = 210.0  # maximum scatter marker size (pt²)  (×1.4)

# Figure geometry (mm; Nature single-column = 88, 1.5-col = 120, full = 180)
FIG_W_MM  = 180
FIG_H_MM  = 160
DPI        = 300

# Palette: 9 colours (from 18167da3…png; #ffa500 excluded; #203f99→#8bcf02)
PALETTE_9 = [
    "#f36424",   # 0  orange
    "#672e92",   # 1  purple
    "#8bcf02",   # 2  lime green    ← replaced #203f99
    "#139ed3",   # 3  sky blue
    "#ff0000",   # 4  red             ← swapped with [7]
    "#ffd700",   # 5  gold            ← swapped with [8]
    "#293890",   # 6  dark blue
    "#74201e",   # 7  dark red        ← swapped with [4]
    "#bebebe",   # 8  grey            ← swapped with [5]
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mm_to_in(mm: float) -> float:
    return mm / 25.4


def _polygon_to_mpl_path(geom) -> MplPath:
    """
    Build a matplotlib Path with an explicit MOVETO at the start of every
    ring (exterior + holes, and every part of a MultiPolygon).

    geopandas' batched GeoDataFrame.plot() draws the whole boundary layer
    as one PathCollection; when serialised to SVG, the break between two
    unrelated rings (e.g. mainland Africa's coastline and Madagascar's, or
    two different countries) can be silently dropped, leaving a LINETO
    where a MOVETO belongs. matplotlib/most browsers still render this
    correctly by luck, but Adobe Illustrator's SVG importer does not: it
    draws a stray straight line connecting the end of one ring to the
    start of the next, visible as long diagonal/horizontal artifacts
    crossing empty ocean. Building the Path explicitly, ring by ring,
    avoids relying on that shared code path entirely.
    """
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    vertices, codes = [], []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            vertices.extend(coords)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(coords) - 2))
            codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mpl.rcParams.update({
        "font.family":      "Arial",
        "font.size":        8,
        "axes.linewidth":   0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "svg.fonttype": "none",
    })

    # --- Load merged city+community+coordinate table (cached) ---
    PLOT_DATA = ROOT / "outputs" / "plot_data.parquet"

    # Invalidate cache if community partition is newer than cached plot data
    use_cache = (
        PLOT_DATA.exists()
        and PLOT_DATA.stat().st_mtime >= COMM_PATH.stat().st_mtime
    )

    if use_cache:
        df = pd.read_parquet(PLOT_DATA)
        logging.info("Loaded cached plot data: %d cities.", len(df))
    else:
        comm_df = pd.read_parquet(COMM_PATH)
        logging.info("Communities loaded: %d cities, %d communities.",
                     len(comm_df), comm_df["community"].nunique())
        city_df = pd.read_excel(CITY_XLSX)[["city", "country", "latitude", "longitude"]]
        df = comm_df.merge(city_df, on=["city", "country"], how="left")
        n_missing = df["latitude"].isna().sum()
        if n_missing:
            logging.warning("%d cities missing coordinates; excluded.", n_missing)
        df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        df.to_parquet(PLOT_DATA, index=False)
        logging.info("Saved plot data cache: %d cities → %s", len(df), PLOT_DATA)

    comm_df = df  # for n_comm count below
    africa = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    # Simplify: the source boundary layer has ~2.3M vertices total (full
    # coastline/admin-border resolution), which is wildly more detail than
    # is visible at this map's scale (the whole continent across ~180mm)
    # and is the main reason the SVG was too heavy for Illustrator to open
    # (98%+ of that file size is vertex coordinates). tol=0.01 degrees
    # (~1km at the equator) cuts total vertices by ~98% with no visible
    # shape change at continental scale.
    africa["geometry"] = africa.geometry.simplify(0.01, preserve_topology=True)

    # --- Arrays ---
    lon  = df["longitude"].values
    lat  = df["latitude"].values
    comm = df["community"].values.astype(int)
    # Point size proportional to sqrt(n_cells)
    sz   = np.clip(POINT_SCALE * np.sqrt(df["n_cells"].values),
                   POINT_MIN_S, POINT_MAX_S)

    colours = np.array(PALETTE_9)[comm]

    # --- Figure ---
    fig, ax = plt.subplots(
        figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Backdrop — one PathPatch per country, each built ring-by-ring with an
    # explicit MOVETO (see _polygon_to_mpl_path), instead of a single batched
    # africa.plot() call whose merged PathCollection can lose ring breaks
    # when serialised to SVG (stray straight lines connecting unrelated
    # rings across empty space).
    #
    # Fill and stroke are drawn in TWO SEPARATE passes (all fills first,
    # then all borders on top) rather than one facecolor+edgecolor patch
    # per country. Neighbouring countries share near-identical boundary
    # coordinates; with a single pass, a later country's opaque fill can
    # sit exactly on top of an earlier country's border stroke along their
    # shared edge. Browsers' anti-aliased rasterisation often leaves a
    # faint seam there by chance, hiding the problem, but Illustrator's
    # precise vector rendering does not — the shared border vanishes
    # completely. Drawing every stroke in a second pass, after all fills,
    # guarantees no fill can ever cover any border.
    geoms = list(africa.geometry)
    for geom in geoms:
        ax.add_patch(PathPatch(
            _polygon_to_mpl_path(geom),
            facecolor=LAND_COLOR, edgecolor="none", linewidth=0, zorder=1,
        ))
    for geom in geoms:
        ax.add_patch(PathPatch(
            _polygon_to_mpl_path(geom),
            facecolor="none", edgecolor=BORDER_COLOR,
            linewidth=BORDER_LW, zorder=2,
        ))

    # Points (draw smallest first so large cities are on top)
    order = np.argsort(sz)
    sc = ax.scatter(
        lon[order], lat[order],
        s=sz[order],
        c=colours[order],
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )

    # Map extent (Africa bounding box with small margin)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- North arrow + scale bar, placed in a verified-empty patch of open
    # Atlantic water (lon -19..-14, lat 34..37.5 — checked against the
    # boundary geometry itself to confirm no land, so no country label or
    # coastline can ever land there) well clear of both legends (bottom
    # corners) and all city points (mainland only). Scale bar uses the
    # equatorial km-per-degree approximation (111.32 km/deg) — a standard
    # simplification for a small-scale reference map in unprojected
    # lat/lon; not survey-accurate at the bar's actual (non-equatorial)
    # latitude. Plain line + perpendicular end-ticks + numbers (the
    # standard scientific-figure scale-bar style), not alternating
    # black/white GIS-software blocks. ---
    KM_PER_DEG = 111.32
    arrow_x, arrow_y = -16.5, 37.4
    ax.annotate(
        "N", xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.9),
        ha="center", va="center", fontsize=7, fontweight="bold",
        fontfamily="Arial", color="#222222", zorder=11,
        arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.0,
                        shrinkA=0, shrinkB=0),
    )

    bar_x0, bar_y0 = -18.7, 34.6
    tick_h = 0.25
    bar_ticks_km = [0, 250, 500]
    bar_x = [bar_x0 + km / KM_PER_DEG for km in bar_ticks_km]
    ax.plot([bar_x[0], bar_x[-1]], [bar_y0, bar_y0],
            color="#222222", linewidth=1.0, solid_capstyle="butt", zorder=11)
    for km, x in zip(bar_ticks_km, bar_x):
        ax.plot([x, x], [bar_y0 - tick_h / 2, bar_y0 + tick_h / 2],
                color="#222222", linewidth=1.0, zorder=11)
        ax.text(x, bar_y0 + tick_h / 2 + 0.15, f"{km:,}",
                ha="center", va="bottom", fontsize=4.5, fontfamily="Arial",
                color="#222222", zorder=11)
    ax.text((bar_x[0] + bar_x[-1]) / 2, bar_y0 - tick_h / 2 - 0.25, "km",
            ha="center", va="top", fontsize=4.5, fontfamily="Arial",
            color="#222222", zorder=11)

    # --- Country name labels (topmost layer) ---
    # Skip tiny/remote territories outside the main Africa extent
    _skip = {"French Southern Territories", "Mayotte", "Reunion",
             "Seychelles", "Comoros", "Cape Verde", "Sao Tome and Principe",
             "Mauritius", "Western Sahara"}
    for _, row in africa.iterrows():
        name = row["NAME_0"]
        if name in _skip:
            continue
        pt = row.geometry.representative_point()
        x, y = pt.x, pt.y
        if not (-20 <= x <= 55 and -37 <= y <= 38):
            continue
        ax.text(x, y, name,
                fontsize=5.0, fontfamily="Arial", fontweight="bold",
                color="#333333",
                ha="center", va="center",
                zorder=10,
                clip_on=True,
                path_effects=[
                    mpl.patheffects.withStroke(linewidth=1.5, foreground="white")
                ])

    # --- Legend (communities, outside right) ---
    n_comm = comm_df["community"].nunique()
    legend_handles = []
    for i in range(n_comm):
        n_cities = (comm == i).sum()
        h = mlines.Line2D(
            [], [],
            marker="o", linestyle="none",
            markersize=5,
            color=PALETTE_9[i],
            alpha=0.9,
            label=f"{i+1}  (n={n_cities:,})",
        )
        legend_handles.append(h)

    leg = ax.legend(
        handles=legend_handles,
        title="Community",
        title_fontsize=7,
        fontsize=6.5,
        loc="lower left",
        frameon=False,              # no box, no border
        handletextpad=0.4,
        borderpad=0.0,
        labelspacing=0.45,
    )
    ax.add_artist(leg)   # keep the community legend when the size legend calls ax.legend() again

    # --- Size legend (marker area encodes n_cells, sqrt-scaled) ---
    size_ticks = [1_000, 10_000, 100_000]
    size_handles = [
        mlines.Line2D(
            [], [], marker="o", linestyle="none",
            markersize=np.sqrt(np.clip(POINT_SCALE * np.sqrt(n), POINT_MIN_S, POINT_MAX_S)),
            markerfacecolor="none", markeredgecolor="#333333", markeredgewidth=0.6,
            label=f"{n:,}",
        )
        for n in size_ticks
    ]
    ax.legend(
        handles=size_handles,
        title="City size\n(grid cells)",
        title_fontsize=7,
        fontsize=6.5,
        loc="lower right",
        frameon=False,
        handletextpad=0.8,
        borderpad=0.0,
        labelspacing=0.8,
    )

    # --- Save ---
    svg_path = OUT_FIG / "community_map.svg"
    png_path = OUT_FIG / "community_map.png"
    fig.savefig(svg_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    logging.info("Saved → %s", svg_path)
    logging.info("Saved → %s", png_path)
    plt.close(fig)

    # --- Console summary ---
    print(f"\n  Cities plotted  : {len(df)}")
    print(f"  Communities     : {n_comm}")
    print(f"  Output SVG      : {svg_path}")
    print(f"  Output PNG      : {png_path}")


if __name__ == "__main__":
    main()
