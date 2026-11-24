"""
01_define_centres.py
====================
Define three city reference centres for each African city, using population
data (Pop2022) already aggregated to the 250-m hexagonal analysis grid.

Three centre definitions
------------------------
primary_pop   — MAIN
    Population-weighted centroid of the largest contiguous high-population
    cluster. Steps:
      1. One-hop neighbour smoothing of Pop2022 (prevents single-cell peaks
         from dominating), averaged over each cell's TRUE geometric hex
         neighbours (see _hex_adjacency_radius).
      2. Candidate cells: top max(10%, 10 cells) by smoothed population.
      3. Contiguous clusters via TRUE hexagon adjacency (see _hex_clusters)
         — not an 8-neighbour rectangular-grid approximation, which does
         not preserve hex adjacency (adjacent hex rows are offset by half
         a cell width).
      4. Primary cluster = largest by total (raw) Pop2022.
      5. Centre = population-weighted centroid of primary cluster cells.

pop_centroid  — ROBUSTNESS A
    Whole-city population-weighted centroid of all hex cells.

geom_centroid — ROBUSTNESS B
    Population-unweighted (geometric) centroid of all hex cell centroids,
    equivalent to the polygon geometric centroid. Matches the previous
    analysis_input.csv latitude/longitude.

All spatial computations are performed in each city's local projected CRS
(from the SHP file), avoiding continental-scale projection distortions.
Outputs are stored as both local-CRS (x, y) and WGS84 (lon, lat) coordinates.

Polycentric indicators
----------------------
n_pop_clusters          — number of candidate high-population clusters
primary_cluster_pop     — total Pop2022 in primary cluster
primary_cluster_share   — primary cluster share of city total Pop2022
secondary_cluster_pop   — total Pop2022 in second-largest cluster (0 if none)
secondary_to_primary    — ratio secondary/primary (0 if none; proxy for polycentricity)

Compactness
-----------
compactness = 4π·A / P²  (Polsby–Popper, ∈ (0, 1]; 1 = perfect circle)
Computed from the union of all hex cell geometries. Used as a robustness
filter in downstream scripts (exclude bottom 10%).

Output
------
    outputs/tables/city_centres.parquet

Usage
-----
    python 01_define_centres.py [--workers N]

Notes
-----
- Pop2022 is estimated population count (area-weighted WorldPop 2022,
  aggregated to the 250-m hex grid; see ../../../01_hex_grid_workflow/).
  It is the primary population
  variable used throughout the analysis (see
  02_regional_structure/regression/00_prepare_data.py).
- Cities with no SHP or total Pop2022 = 0 are flagged centre_qc = False and
  fall back to geom_centroid for all three centre fields.
- The city list is read directly from SHP_ROOT's directory structure
  (SHP_ROOT/<country>/<city>/), not from analysis_input.csv, since that
  file is itself produced downstream of this script's output
  (city_centres.parquet feeds population_worldpop_2022 into
  02_regional_structure/regression/00_prepare_data.py) — reading it here
  would make the two scripts depend on each other's output.
- Parallelised over cities using ProcessPoolExecutor.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT     = Path(__file__).resolve().parent
# SHP geometry + attributes per city, published by
# ../../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../02_morphology_metrics/outputs/SHP_cleaned",
))
OUT_DIR  = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum candidate cells for high-pop cluster definition
MIN_CANDIDATE_FRAC = 0.10   # top 10%
MIN_CANDIDATE_ABS  = 10     # but at least 10 cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weighted_centroid(x, y, w):
    """Population-weighted centroid in projected coordinates."""
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(x)), float(np.mean(y))
    return float((x * w).sum() / total), float((y * w).sum() / total)


def _project_to_wgs84(x_proj, y_proj, crs_from):
    """Convert a single (x, y) point from crs_from to WGS84 lon/lat."""
    t = Transformer.from_crs(crs_from, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x_proj, y_proj)
    return float(lon), float(lat)


def _hex_adjacency_radius(cx: np.ndarray, cy: np.ndarray) -> float:
    """Adjacency radius for true hexagon-to-hexagon neighbours, calibrated
    from the actual cell-centre spacing rather than assumed from the SHP's
    nominal resolution.

    Hexagon centres in a pointy/flat-top hex tiling are NOT on a Cartesian
    grid: alternating rows are offset by half a cell width, so the
    nearest-neighbour distance (~cell "pitch") is the correct calibration
    reference for adjacency, rather than any axis-aligned spacing. A single
    true hex cell has exactly 6 neighbours at (very close to) this pitch
    distance, with a small multiplicative tolerance for floating-point /
    projection jitter.
    """
    n = len(cx)
    tree = cKDTree(np.column_stack([cx, cy]))
    k = min(7, n)
    dists, _ = tree.query(np.column_stack([cx, cy]), k=k)
    nn_dist = float(np.median(dists[:, 1])) if k > 1 else 0.0
    return tree, nn_dist * 1.15


def _hex_clusters(cx: np.ndarray, cy: np.ndarray, mask: np.ndarray):
    """
    Label contiguous clusters among the cells where `mask` is True, using
    TRUE hexagon adjacency (neighbours within the calibrated hex pitch),
    not an 8-neighbour rectangular-grid approximation.

    Hexagon centres are not on a Cartesian lattice — adjacent rows are
    offset by half a cell width — so quantising centres onto a rectangular
    row/col index and applying 8-connectivity would not preserve true
    hexagon adjacency. Building the adjacency graph directly from
    inter-cell distances (this function) is exact for any hex tiling
    (pointy-top, flat-top, or with irregular real-world cell placement)
    and does not depend on axis alignment.

    Returns an array of cluster labels (1-based, 0 = not in mask) aligned
    with the input arrays, and the total number of clusters among masked
    cells.
    """
    n = len(cx)
    labels = np.zeros(n, dtype=int)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return labels, 0

    tree, radius = _hex_adjacency_radius(cx, cy)
    # Adjacency among masked cells only: query each masked cell's true
    # geometric neighbours and keep edges where both endpoints are masked.
    pairs = tree.query_ball_point(np.column_stack([cx[idx], cy[idx]]), r=radius)
    row_i, col_j = [], []
    for local_i, neighbours in enumerate(pairs):
        gi = idx[local_i]
        for gj in neighbours:
            if gj != gi and mask[gj]:
                row_i.append(gi)
                col_j.append(gj)

    if row_i:
        adj = coo_matrix(
            (np.ones(len(row_i), dtype=bool), (row_i, col_j)), shape=(n, n)
        )
        n_clusters, comp_labels = connected_components(
            adj, directed=False, connection="weak")
    else:
        # No edges at all (e.g. a single masked cell): each masked cell is
        # its own singleton cluster.
        comp_labels = np.arange(n)
        n_clusters = n

    # Relabel 1..k over masked cells only, in first-seen order (by position
    # in the masked-cell array) for a stable, reproducible label numbering
    # independent of connected_components' internal (arbitrary) label ids.
    masked_comp_labels = comp_labels[idx]
    unique_comps, first_seen_pos = np.unique(masked_comp_labels, return_index=True)
    # unique_comps[i] first appears at position first_seen_pos[i] in
    # masked_comp_labels; sorting by that position gives first-seen order.
    order = np.argsort(first_seen_pos)
    remap = {int(unique_comps[i]): rank + 1 for rank, i in enumerate(order)}
    labels[idx] = [remap[int(c)] for c in masked_comp_labels]
    return labels, len(unique_comps)


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------
def _process_city(args):
    country, city = args

    shp_dir = SHP_ROOT / country / city
    if not shp_dir.exists():
        return None
    shps = list(shp_dir.glob("*.shp"))
    if not shps:
        return None

    try:
        gdf = gpd.read_file(shps[0])  # need the LC_* columns for coverage below
        if gdf.empty:
            return None

        crs = gdf.crs
        cx  = gdf["center_x"].to_numpy(dtype=float)   # local projected CRS
        cy  = gdf["center_y"].to_numpy(dtype=float)
        pop_raw = gdf["Pop2022"].to_numpy(dtype=float)
        pop = np.where(np.isfinite(pop_raw), pop_raw, 0.0)

        n_cells = len(gdf)

        # population_coverage: fraction of grid cells with ANY valid raster
        # data, not the fraction with Pop2022 > 0. A cell can legitimately
        # have zero population (water, bare ground, open space) while still
        # being fully covered by the source rasters — Pop2022 == 0 alone
        # cannot distinguish that from a true data gap (a cell the WorldPop
        # / land-cover rasters never touched, e.g. near a country border or
        # a raster mosaic seam). A real data gap is instead identifiable by
        # its OTHER independently-sourced fields (all 23 LC_* land-cover
        # fractions) also being exactly zero, whereas a cell with legitimate
        # zero population still carries nonzero LC_Water / LC_Built / etc.
        lc_cols = [c for c in gdf.columns if c.startswith("LC_")]
        if lc_cols:
            lc_sum = gdf[lc_cols].to_numpy(dtype=float).sum(axis=1)
            has_raster_data = np.isfinite(lc_sum) & (lc_sum > 0)
        else:
            # No LC_* columns present in this SHP: fall back to the
            # population-based proxy rather than fail outright.
            has_raster_data = np.isfinite(pop_raw) & (pop_raw > 0)

        n_valid_population_cells = int(has_raster_data.sum())
        population_coverage      = n_valid_population_cells / n_cells if n_cells > 0 else 0.0
        total_pop = pop.sum()

        # ------------------------------------------------------------------
        # ROBUSTNESS B: geometric centroid (unweighted mean of cell centres)
        # ------------------------------------------------------------------
        geom_x = float(np.mean(cx))
        geom_y = float(np.mean(cy))
        geom_lon, geom_lat = _project_to_wgs84(geom_x, geom_y, crs)

        # ------------------------------------------------------------------
        # ROBUSTNESS A: whole-city population-weighted centroid
        # ------------------------------------------------------------------
        if total_pop > 0:
            pop_cx, pop_cy = _weighted_centroid(cx, cy, pop)
        else:
            pop_cx, pop_cy = geom_x, geom_y
        pop_clon, pop_clat = _project_to_wgs84(pop_cx, pop_cy, crs)

        # ------------------------------------------------------------------
        # MAIN: primary population cluster centre
        # ------------------------------------------------------------------
        # Step 1 — one-hop neighbour smoothing, using TRUE hexagon adjacency
        # (see _hex_adjacency_radius / _hex_clusters docstrings: hexagon
        # centres are not on a Cartesian grid, so binning them into a
        # rectangular row/col index and applying an 8-neighbour rectangular
        # filter does not preserve hex adjacency).
        tree, radius = _hex_adjacency_radius(cx, cy)
        neighbour_lists = tree.query_ball_point(
            np.column_stack([cx, cy]), r=radius)

        pop_smooth = np.empty(n_cells, dtype=float)
        for i, neighbours in enumerate(neighbour_lists):
            # one-hop neighbourhood = self + true geometric neighbours
            pop_smooth[i] = pop[neighbours].mean()

        # Step 2 — candidate cells: top max(10%, 10) by smoothed population
        n_candidates = max(int(np.ceil(MIN_CANDIDATE_FRAC * n_cells)),
                           MIN_CANDIDATE_ABS)
        n_candidates = min(n_candidates, n_cells)
        threshold    = np.partition(pop_smooth, -n_candidates)[-n_candidates]
        # NOTE (methods review): `candidate` selects ALL cells >= threshold,
        # not exactly the top n_candidates -- when many cells share the
        # threshold value (common for low-population cities where
        # pop_smooth has many exact ties, e.g. several all-zero
        # neighbourhoods), the actual candidate count can run well above
        # the intended ~10%. n_candidates_actual / candidate_ties_flag
        # below are diagnostic-only and don't change this behaviour.
        candidate    = pop_smooth >= threshold
        n_candidates_actual = int(candidate.sum())
        candidate_ties_flag = n_candidates_actual > int(1.5 * n_candidates)

        # Step 3 — contiguous clusters among candidate cells, TRUE hex adjacency
        full_cluster_labels, n_clusters = _hex_clusters(cx, cy, candidate)
        cluster_labels = full_cluster_labels[candidate]

        # Step 4 — primary cluster = highest total raw Pop2022
        cand_pop = pop[candidate]
        cluster_pops = {}
        for lbl, p in zip(cluster_labels, cand_pop):
            cluster_pops[lbl] = cluster_pops.get(lbl, 0.0) + p

        sorted_clusters = sorted(cluster_pops.items(),
                                 key=lambda kv: kv[1], reverse=True)
        primary_lbl         = sorted_clusters[0][0]
        primary_cluster_pop = sorted_clusters[0][1]
        secondary_cluster_pop = sorted_clusters[1][1] if len(sorted_clusters) > 1 else 0.0
        secondary_to_primary  = (secondary_cluster_pop / primary_cluster_pop
                                  if primary_cluster_pop > 0 else 0.0)

        # Step 5 — population-weighted centroid of primary cluster
        in_primary = full_cluster_labels == primary_lbl

        prim_cx, prim_cy = _weighted_centroid(cx[in_primary], cy[in_primary],
                                              pop[in_primary])
        prim_lon, prim_lat = _project_to_wgs84(prim_cx, prim_cy, crs)

        # ------------------------------------------------------------------
        # Compactness: Polsby–Popper = 4π·A / P²
        # Using union of all hex polygons
        # ------------------------------------------------------------------
        try:
            union     = gdf.geometry.unary_union
            area_m2   = union.area
            perim_m   = union.length
            compactness = (4 * np.pi * area_m2 / (perim_m ** 2)
                           if perim_m > 0 else np.nan)
        except Exception:
            area_m2 = perim_m = compactness = np.nan

        # ------------------------------------------------------------------
        # centre_qc: True if population data is valid
        # ------------------------------------------------------------------
        centre_qc = (total_pop > 0) and np.isfinite(prim_lon)

        return {
            "city":    city,
            "country": country,
            "n_cells": n_cells,
            "total_pop2022": total_pop,
            "n_valid_population_cells": n_valid_population_cells,
            "population_coverage": population_coverage,

            # Primary population centre (MAIN)
            "primary_pop_x":   prim_cx,
            "primary_pop_y":   prim_cy,
            "primary_pop_lon": prim_lon,
            "primary_pop_lat": prim_lat,

            # Whole-city population-weighted centroid (ROBUSTNESS A)
            "pop_centroid_x":   pop_cx,
            "pop_centroid_y":   pop_cy,
            "pop_centroid_lon": pop_clon,
            "pop_centroid_lat": pop_clat,

            # Geometric centroid (ROBUSTNESS B)
            "geom_centroid_x":   geom_x,
            "geom_centroid_y":   geom_y,
            "geom_centroid_lon": geom_lon,
            "geom_centroid_lat": geom_lat,

            # Polycentric indicators
            "n_pop_clusters":        n_clusters,
            "primary_cluster_pop":   primary_cluster_pop,
            "primary_cluster_share": (primary_cluster_pop / total_pop
                                      if total_pop > 0 else np.nan),
            "secondary_cluster_pop": secondary_cluster_pop,
            "secondary_to_primary":  secondary_to_primary,

            # Candidate-selection diagnostics (methods review): flags
            # cities where ties in smoothed population inflated the
            # candidate set well past the intended ~10% (n_candidates).
            "n_candidates":          n_candidates,
            "n_candidates_actual":   n_candidates_actual,
            "candidate_ties_flag":   candidate_ties_flag,

            # Shape
            "compactness": compactness,

            # QC
            "centre_qc": centre_qc,
            "crs_epsg":  crs.to_epsg() if crs else None,
        }

    except Exception as exc:
        logging.warning("  %s / %s failed: %s", country, city, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 14) -> None:
    logging.info("Scanning city list from %s ...", SHP_ROOT)
    tasks = [
        (country_dir.name, city_dir.name)
        for country_dir in sorted(SHP_ROOT.iterdir())
        if country_dir.is_dir()
        for city_dir in sorted(country_dir.iterdir())
        if city_dir.is_dir()
    ]
    logging.info("  %d cities.", len(tasks))

    logging.info("Defining centres with %d workers ...", n_workers)
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_city, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                logging.info("  %d / %d cities done ...", done, len(tasks))
            res = fut.result()
            if res is not None:
                results.append(res)

    logging.info("  %d / %d cities produced output.", len(results), len(tasks))

    if not results:
        logging.error("No results — check warnings above.")
        return

    out_df = pd.DataFrame(results)

    # --- Validation ---
    logging.info("Validation:")
    logging.info("  centre_qc=True:  %d cities", out_df["centre_qc"].sum())
    logging.info("  centre_qc=False: %d cities", (~out_df["centre_qc"]).sum())
    logging.info("  compactness: mean=%.3f  median=%.3f  min=%.4f  max=%.4f",
                 out_df["compactness"].mean(), out_df["compactness"].median(),
                 out_df["compactness"].min(), out_df["compactness"].max())
    logging.info("  secondary_to_primary: median=%.3f  >0.5: %d cities",
                 out_df["secondary_to_primary"].median(),
                 (out_df["secondary_to_primary"] > 0.5).sum())

    # Distance between primary_pop and geom_centroid (sanity check)
    dx = out_df["primary_pop_lon"] - out_df["geom_centroid_lon"]
    dy = out_df["primary_pop_lat"] - out_df["geom_centroid_lat"]
    dist_deg = np.sqrt(dx**2 + dy**2)
    logging.info("  primary_pop vs geom_centroid distance (degrees):")
    logging.info("    median=%.4f°  P90=%.4f°  P99=%.4f°",
                 dist_deg.median(), dist_deg.quantile(0.90),
                 dist_deg.quantile(0.99))

    out_path = OUT_DIR / "city_centres.parquet"
    out_df.to_parquet(out_path, index=False)
    logging.info("Saved: %s  (%d cities)", out_path, len(out_df))
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    main(n_workers=args.workers)
