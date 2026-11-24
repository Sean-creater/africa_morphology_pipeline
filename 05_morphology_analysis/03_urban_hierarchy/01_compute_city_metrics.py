"""
01_compute_city_metrics.py
==========================
Compute city-level morphological metrics for urban hierarchy analysis.

For each city (n≈6918) this script computes:
  - p1..p4   : share of grid cells belonging to each higher-order morphology group
  - H_star   : normalised Shannon entropy  H* = H / log(4),  H* ∈ [0,1]
               0 = single group dominates, 1 = all four groups equal
  - r        : Newman categorical spatial assortativity of the four groups,
               computed from Queen adjacency built directly from SHP geometry
               r > 0 → same-group cells cluster spatially (segregation)
               r ≈ 0 → random mixing
               r < 0 → cross-group cells cluster (interleaving)
  - H17_star : robustness — normalised Shannon entropy over 17 types (Supplement)
  - r17      : robustness — assortativity over 17 types
  - H17_star_rare : H17_star recomputed on a fixed-size (N_RAREFY cells)
               random subsample of each city's grid cells, averaged over
               N_RAREFY_REPS draws; NaN for cities with n_cells < N_RAREFY.
               H17_star alone is mechanically bounded by n_cells (a city
               with n_cells < 17 cannot show all 17 types, so its ceiling
               is log(n_cells)/log(17) < 1 regardless of true diversity);
               rarefaction evaluates every included city at the same
               "sampling effort" so population-H17_star relationships
               aren't partly an artefact of population-n_cells correlation
               (r = 0.77 between log_pop and log(n_cells) in this dataset).
  - r_fail_reason : why r/r17 are NaN, for cities where they are — one of
               "no_shp" (no SHP file for this city), "empty_merge" (SHP
               loaded but merge against grid labels produced zero rows),
               or "error:<ExceptionType>" (an actual failure in reading/
               reprojecting/building the Queen graph — logged, not
               silently swallowed). NaN/None where r/r17 were computed.
  - r17_z    : permutation-null z-score for r17: hold each city's Queen
               graph AND its 17-type count distribution fixed, randomly
               permute which node gets which type label (N_PERM draws),
               recompute r17 under each permutation, then
               z = (r17_obs - mean(r17_perm)) / sd(r17_perm). Unlike the
               log(n_edges) covariate-control robustness check in
               06/08 — where n_edges is a mix of cell count, graph
               topology AND population-driven spatial extent, so
               "controlling" for it is a conditional association, not a
               clean confound removal — this directly answers "given this
               city's actual graph size, topology and type composition,
               is its type-level spatial clustering stronger or weaker
               than chance", which is the quantity population should be
               regressed against if the question is about genuine spatial
               organisation rather than raw graph scale.

Covariates joined from existing tables:
  - community  (1–8, Leiden k=100; community 0 is a valid Leiden community)
  - log_pop    (natural log of city population — WorldPop 2022, aggregated
                by 00_prepare_data.py from the area-weighted WorldPop 2022
                grid within each Africapolis urban agglomeration)
  - country
  - n_cells    (number of grid cells in city, QC)
  - n_edges    (total adjacency edges in city, QC)

Output
------
outputs/tables/city_metrics.parquet
    One row per city; all columns above plus city, country.

Usage
-----
    python 01_compute_city_metrics.py [--workers N]

    --workers   Number of parallel workers (default: 16)

Note: requires geopandas + libpysal (available in DL_env / Africa_env).
Run with:
    python 01_compute_city_metrics.py
"""

import argparse
import hashlib
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
# SHP geometry + attributes per city, published by
# ../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT    = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../02_morphology_metrics/outputs/SHP_cleaned",
))
LABELS_PATH = (ROOT.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")
COMM_PATH   = (ROOT.parent / "02_regional_structure" /
               "community_detection" / "outputs" / "city_communities.parquet")
REG_PATH    = (ROOT.parent / "02_regional_structure" /
               "regression" / "outputs" / "tables" / "analysis_input.csv")
OUT_DIR     = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Higher-order group mapping  (label_k17 → group 1..4)
# ---------------------------------------------------------------------------
# Derived from real Ward hierarchical clustering (method="ward", euclidean)
# of the 17x91 cluster-profile (D_cj) matrix, cut at k=4 — see
# 01_typology_construction/cluster_interpretation/04_cluster_profiles.py.
# This is a DATA-DRIVEN partition (not an asserted/manual grouping): the
# gap-statistic on merge heights shows k=4 is the second-most-natural cut
# in the whole dendrogram (after k=2), and {12,13,14,15} never forms a
# single branch at any k — 14,15 are structurally closer to the large
# group {2,9,10,11,16,17} than to {12,13}.
# Group 1 — Cropland-Fringe Low-Rise (CFL)     : C4, C5
# Group 2 — Heterogeneous Sparse Fabric (HSF)  : C1, C3, C6, C7, C8
# Group 3 — Large-Volume Compact Fabric (LVC)  : C12, C13
# Group 4 — Dense Small-Building Fabric (DSB)  : C2, C9, C10, C11, C14, C15, C16, C17
GROUP_MAP = {
    4: 1,  5: 1,
    1: 2,  3: 2,  6: 2,  7: 2,  8: 2,
    12: 3, 13: 3,
    2: 4,  9: 4, 10: 4, 11: 4, 14: 4, 15: 4, 16: 4, 17: 4,
}
N_GROUPS = 4
N_TYPES  = 17

# Rarefaction: fixed sample size matching the existing ">= 50 cells"
# robustness filter elsewhere in this module, so the two are directly
# comparable. Cities below this are excluded from H17_star_rare (can't
# subsample more cells than a city has).
N_RAREFY      = 50
N_RAREFY_REPS = 50

# Permutation null for r17: number of type-label permutations per city
# (graph structure and the 17-type count distribution held fixed — see
# module docstring, r17_z).
N_PERM = 200


class _EmptyMergeError(Exception):
    """SHP loaded but merge against grid labels produced zero rows."""


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------
def _shannon(counts: np.ndarray, n_cat: int) -> float:
    """Normalised Shannon entropy H* = H / log(n_cat)."""
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    return h / np.log(n_cat)


def _rarefied_h17(type_arr: np.ndarray, n_target: int, n_reps: int,
                   rng: np.random.Generator) -> float:
    """
    H17_star on a fixed-size random subsample of a city's grid cells,
    averaged over n_reps draws. Removes the mechanical dependence of
    H17_star's ceiling on n_cells (see module docstring) by evaluating
    every city at the same sampling effort. NaN if the city has fewer
    than n_target cells.
    """
    n_cells = len(type_arr)
    if n_cells < n_target:
        return np.nan
    vals = np.empty(n_reps)
    for i in range(n_reps):
        sample = rng.choice(type_arr, size=n_target, replace=False)
        counts = np.zeros(N_TYPES + 1, dtype=int)
        for t in sample:
            if 1 <= t <= N_TYPES:
                counts[t] += 1
        vals[i] = _shannon(counts[1:], N_TYPES)
    return float(np.nanmean(vals))


# ---------------------------------------------------------------------------
# Newman assortativity from edge mixing matrix
# ---------------------------------------------------------------------------
def _assortativity_from_edges(labels_a: np.ndarray, labels_b: np.ndarray,
                               n_cat: int) -> float:
    """
    Compute Newman categorical assortativity from arrays of edge endpoint labels.
    Works on undirected edges (each edge appears once).

    r = (Tr(e) - sum(a_i^2)) / (1 - sum(a_i^2))

    where e_ij = fraction of edges with endpoints in categories i, j
    and a_i = sum_j e_ij  (row marginal).
    """
    if len(labels_a) == 0:
        return np.nan
    # Build mixing matrix
    e = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if 0 < a <= n_cat and 0 < b <= n_cat:
            e[a - 1, b - 1] += 1
            e[b - 1, a - 1] += 1   # undirected: count both directions
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    trace = np.trace(e)
    a     = e.sum(axis=1)
    denom = 1 - np.sum(a ** 2)
    if abs(denom) < 1e-12:
        return np.nan
    return (trace - np.sum(a ** 2)) / denom


def _assortativity_from_edges_vec(labels_a: np.ndarray, labels_b: np.ndarray,
                                   n_cat: int) -> float:
    """
    Same computation as _assortativity_from_edges, vectorised (no Python
    loop over edges). Used for the r17 permutation null, where this runs
    N_PERM times per city and the Python-loop version would dominate
    runtime; kept as a separate function (rather than replacing the
    original) so the main r/r17 computation's existing, already-verified
    behaviour is untouched.
    """
    valid = (labels_a > 0) & (labels_a <= n_cat) & (labels_b > 0) & (labels_b <= n_cat)
    if not np.any(valid):
        return np.nan
    ia, ib = labels_a[valid] - 1, labels_b[valid] - 1
    # bincount on flattened (row, col) indices is substantially faster
    # than np.add.at for this — matters here since this function runs
    # N_PERM times per city across thousands of cities (permutation null).
    flat = np.concatenate([ia * n_cat + ib, ib * n_cat + ia])
    e = np.bincount(flat, minlength=n_cat * n_cat).astype(float).reshape(n_cat, n_cat)
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    trace = np.trace(e)
    a     = e.sum(axis=1)
    denom = 1 - np.sum(a ** 2)
    if abs(denom) < 1e-12:
        return np.nan
    return (trace - np.sum(a ** 2)) / denom


def _r17_permutation_z(r17_obs: float, t17_vals: np.ndarray,
                        focal_idx: np.ndarray, neighbour_idx: np.ndarray,
                        n_perm: int, rng: np.random.Generator):
    """
    Permutation null for r17: shuffle which node holds which type label
    (graph edges — focal_idx/neighbour_idx — and the type-count
    distribution both held fixed, since a shuffle of an existing array is
    a relabelling, not a resample), recompute r17 under each shuffle, and
    return (z, p) where
        z = (r17_obs - mean(r17_perm)) / sd(r17_perm)
        p = two-sided permutation p-value
    (nan, nan) if r17_obs itself is nan or the null distribution is
    degenerate (sd == 0, e.g. a city too small/homogeneous for the
    permuted assortativity to ever vary).
    """
    if np.isnan(r17_obs) or len(focal_idx) == 0:
        return np.nan, np.nan
    perm_vals = np.empty(n_perm)
    labels = t17_vals.copy()
    for i in range(n_perm):
        rng.shuffle(labels)
        perm_vals[i] = _assortativity_from_edges_vec(
            labels[focal_idx], labels[neighbour_idx], N_TYPES)
    perm_vals = perm_vals[np.isfinite(perm_vals)]
    if perm_vals.size < n_perm // 2:
        return np.nan, np.nan   # too many degenerate draws to trust mean/sd
    mu, sd = perm_vals.mean(), perm_vals.std()
    if sd < 1e-12:
        return np.nan, np.nan
    z = (r17_obs - mu) / sd
    p = (np.sum(np.abs(perm_vals - mu) >= np.abs(r17_obs - mu)) + 1) / (perm_vals.size + 1)
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Per-city computation
# ---------------------------------------------------------------------------
def _process_city(args):
    """
    Worker function.  Loads SHP, computes Queen adjacency, returns metrics dict.
    """
    country, city, city_labels_df = args

    # ---- Label arrays ----
    grid_ids  = city_labels_df["grid_id"].values
    type_arr  = city_labels_df["label_k17"].values
    group_arr = np.array([GROUP_MAP.get(int(t), 0) for t in type_arr])

    n_cells = len(grid_ids)
    if n_cells == 0:
        return None

    # ---- Group share & entropy ----
    group_counts = np.zeros(N_GROUPS + 1, dtype=int)
    for g in group_arr:
        if 1 <= g <= N_GROUPS:
            group_counts[g] += 1
    type_counts = np.zeros(N_TYPES + 1, dtype=int)
    for t in type_arr:
        if 1 <= t <= N_TYPES:
            type_counts[t] += 1

    n_valid = int(group_counts[1:].sum())
    if n_valid == 0:
        return None
    p = group_counts[1:] / n_valid

    h_star   = _shannon(group_counts[1:], N_GROUPS)
    h17_star = _shannon(type_counts[1:],  N_TYPES)

    # Deterministic per-city seed so rarefaction draws are reproducible
    # across re-runs. Python's builtin hash() is randomised per-process
    # (PYTHONHASHSEED) and would give different seeds on every run — use
    # a stable hash (md5) instead.
    seed = int(hashlib.md5(f"{country}|{city}".encode()).hexdigest()[:8], 16)
    rng  = np.random.default_rng(seed)
    h17_star_rare = _rarefied_h17(type_arr, N_RAREFY, N_RAREFY_REPS, rng)

    # ---- Queen adjacency from SHP ----
    r = np.nan; r17 = np.nan; n_edges = 0
    r17_z = np.nan; r17_perm_p = np.nan
    r_fail_reason = None
    shp_dir = SHP_ROOT / country / city
    shps    = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
    if not shps:
        r_fail_reason = "no_shp"
    else:
        try:
            import geopandas as gpd
            from libpysal.weights import Queen

            gdf = gpd.read_file(shps[0])
            # Merge labels onto gdf
            lbl_df = pd.DataFrame({"GridID": grid_ids,
                                   "label_k17": type_arr,
                                   "group4": group_arr})
            gdf = gdf.merge(lbl_df, on="GridID", how="inner")
            if len(gdf) == 0:
                raise _EmptyMergeError("Empty merge")

            # Reproject to Web Mercator (EPSG:3857) before Queen computation,
            # for consistent adjacency detection across countries.
            if gdf.crs is not None and gdf.crs.to_epsg() != 3857:
                gdf = gdf.to_crs(epsg=3857)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = Queen.from_dataframe(gdf, silence_warnings=True)

            # Collect edge endpoint labels
            gdf = gdf.reset_index(drop=True)
            g4_vals  = gdf["group4"].values
            t17_vals = gdf["label_k17"].values

            ea, eb     = [], []
            e17a, e17b = [], []
            focal_idx, neighbour_idx = [], []   # node INDICES, for the
                                                 # r17 permutation null below
            for focal, neighbours in w.neighbors.items():
                ga = g4_vals[focal]
                ta = t17_vals[focal]
                for nb in neighbours:
                    if nb > focal:  # count each undirected edge once
                        gb = g4_vals[nb]
                        tb = t17_vals[nb]
                        ea.append(ga); eb.append(gb)
                        e17a.append(ta); e17b.append(tb)
                        focal_idx.append(focal); neighbour_idx.append(nb)

            n_edges = len(ea)
            r   = _assortativity_from_edges(np.array(ea),  np.array(eb),  N_GROUPS)
            r17 = _assortativity_from_edges(np.array(e17a), np.array(e17b), N_TYPES)

            # Permutation null for r17 (see module docstring, r17_z): hold
            # this SAME graph (focal_idx/neighbour_idx) and type-count
            # distribution (t17_vals) fixed, shuffle labels across nodes.
            r17_z, r17_perm_p = _r17_permutation_z(
                r17, t17_vals, np.array(focal_idx), np.array(neighbour_idx),
                N_PERM, rng)

        except _EmptyMergeError:
            r_fail_reason = "empty_merge"
        except Exception as e:
            # Real failure (bad geometry, CRS issue, Queen construction
            # failure, ...) — distinct from the mathematically-undefined
            # cases already handled inside _assortativity_from_edges
            # (single type / zero edges / zero-variance mixing matrix,
            # which return nan without raising). Logged, not silent, so
            # these are visible and distinguishable from "no_shp" and
            # "empty_merge" in the fail-reason summary printed in main().
            r_fail_reason = f"error:{type(e).__name__}"
            logging.warning("r/r17 failed for %s/%s: %s: %s",
                           country, city, type(e).__name__, e)

    # Distinguish the two mathematically-undefined cases that
    # _assortativity_from_edges already handles internally (returning nan
    # without raising) rather than leaving them both as an uninformative
    # "no error, but still nan": zero Queen-adjacency edges (e.g.
    # single-cell cities) vs. a zero-variance mixing matrix (every edge
    # connects the same single type/group — r is undefined because
    # Newman's r needs non-zero category variance in the first place, not
    # because anything failed).
    if r_fail_reason is None and np.isnan(r):
        r_fail_reason = "zero_edges" if n_edges == 0 else "zero_variance"

    return {
        "country":  country,
        "city":     city,
        "n_cells":  n_cells,
        "n_edges":  n_edges,
        "p1":       p[0],
        "p2":       p[1],
        "p3":       p[2],
        "p4":       p[3],
        "H_star":   h_star,
        "H17_star": h17_star,
        "H17_star_rare": h17_star_rare,
        "r":        r,
        "r17":      r17,
        "r17_z":    r17_z,
        "r17_perm_p": r17_perm_p,
        "r_fail_reason": r_fail_reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 16) -> None:
    # --- Load grid labels ---
    logging.info("Loading grid labels ...")
    labels_df = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    city_groups = (
        labels_df.groupby(["country", "city"])
        .apply(lambda x: x[["grid_id", "label_k17"]], include_groups=False)
        .reset_index(level=[0, 1])
        .groupby(["country", "city"])
    )

    tasks = [
        (name[0], name[1], grp.reset_index(drop=True))
        for name, grp in city_groups
    ]
    logging.info("  %d cities to process.", len(tasks))

    # --- Parallel processing ---
    results = []
    logging.info("Computing metrics with %d workers ...", n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_city, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                logging.info("  %d / %d cities done ...", done, len(tasks))
            res = fut.result()
            if res is not None:
                results.append(res)

    metrics = pd.DataFrame(results)
    logging.info("Metrics computed for %d cities.", len(metrics))

    # --- Join community ---
    comm = pd.read_parquet(COMM_PATH)[["country", "city", "community"]]
    metrics = metrics.merge(comm, on=["country", "city"], how="left")

    # --- Join log_pop and covariates ---
    # Population is WorldPop 2022 (see 02_regional_structure/regression/
    # 00_prepare_data.py's docstring for the population-definition policy);
    # kept as the `log_pop` column name here since every downstream script
    # in 03_urban_hierarchy and 04_center_periphery reads city_metrics.parquet
    # expecting that name — only the underlying source column changed.
    reg = pd.read_csv(REG_PATH, usecols=[
        "city", "country", "log_pop_worldpop_2022", "climate_zone", "colonial", "log_gdp"
    ]).rename(columns={"log_pop_worldpop_2022": "log_pop"})
    metrics = metrics.merge(reg, on=["country", "city"], how="left")

    # --- QC ---
    n_miss_comm = metrics["community"].isna().sum()
    n_miss_pop  = metrics["log_pop"].isna().sum()
    n_miss_r    = metrics["r"].isna().sum()
    n_rare      = metrics["H17_star_rare"].notna().sum()
    logging.info("QC — missing community: %d, missing log_pop: %d, missing r: %d",
                 n_miss_comm, n_miss_pop, n_miss_r)
    logging.info("r range: %.3f – %.3f (median %.3f)",
                 metrics["r"].min(), metrics["r"].max(), metrics["r"].median())
    logging.info("H* range: %.3f – %.3f (median %.3f)",
                 metrics["H_star"].min(), metrics["H_star"].max(),
                 metrics["H_star"].median())
    logging.info("H17_star_rare computed for %d / %d cities (n_cells >= %d)",
                 n_rare, len(metrics), N_RAREFY)
    n_r17z = metrics["r17_z"].notna().sum()
    logging.info("r17_z (permutation null, N_PERM=%d) computed for %d / %d cities",
                 N_PERM, n_r17z, len(metrics))

    # r/r17 failure-reason breakdown — replaces the previous silent
    # except-pass, so "mathematically can't be defined" (no_shp /
    # cities never reaching this block) is visible and separable from
    # genuine code/data failures ("error:<ExceptionType>").
    reason_counts = metrics.loc[metrics["r"].isna(), "r_fail_reason"].value_counts(dropna=False)
    logging.info("r/r17 missing — reason breakdown:\n%s", reason_counts.to_string())

    # --- Save ---
    out = OUT_DIR / "city_metrics.parquet"
    metrics.to_parquet(out, index=False)
    logging.info("Saved: %s  (%d rows × %d cols)", out, len(metrics), len(metrics.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute city-level morphological metrics for urban hierarchy analysis."
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel worker processes (default: 16)."
    )
    args = parser.parse_args()
    main(n_workers=args.workers)
