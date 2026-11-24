"""
02_cell_radial_position.py
==========================
For every hex cell in every African city, compute its normalised radial
distance from each of three city reference centres and assign radial rings.

Three centre versions (from 01_define_centres.py)
--------------------------------------------------
primary_pop   — MAIN: population-weighted centroid of dominant high-pop cluster
pop_centroid  — ROBUSTNESS A: whole-city population-weighted centroid
geom_centroid — ROBUSTNESS B: unweighted geometric centroid of hex cell centres

Distance computation
--------------------
Distances are computed in each city's LOCAL projected CRS using the
center_x / center_y columns stored in the SHP file (same CRS as the SHP).
This avoids continental-scale projection distortions that arise from using
a single EPSG:3857 or ESRI:102022 projection for all of Africa.

Normalised radial position
--------------------------
    dist_norm = min(dist_m / P98(city), 1.0)  ∈ [0, 1]

    P98 is computed independently for each centre version.

Radial rings
------------
    ring5: 5 bands of width 0.2  — [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]
    ring3: 3 bands of width 0.33 — [0,.33) [.33,.67) [.67,1]  (robustness)

Output
------
    outputs/tables/cell_radial_position.parquet

    Columns:
        GridID, city, country
        label_k17, group4
        cx, cy                     — cell centroid in local CRS (metres)
        dist_m_primary             — distance to primary_pop centre
        dist_norm_primary
        ring5_primary, ring3_primary
        dist_m_pop                 — distance to pop_centroid
        dist_norm_pop
        ring5_pop, ring3_pop
        dist_m_geom                — distance to geom_centroid
        dist_norm_geom
        ring5_geom, ring3_geom

    Default downstream scripts use *_primary columns as "main".

Usage
-----
    python 02_cell_radial_position.py [--workers N]
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT        = Path(__file__).resolve().parent
# SHP geometry + attributes per city, published by
# ../../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT    = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../02_morphology_metrics/outputs/SHP_cleaned",
))
LABELS_PATH = (ROOT.parent.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")
CENTRES_PATH = ROOT / "outputs" / "tables" / "city_centres.parquet"
OUT_DIR      = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data-driven Ward k=4 cut of the 17x91 cluster-profile matrix — see
# 01_typology_construction/cluster_interpretation/04_cluster_profiles.py.
# Group 1 Cropland-Fringe Low-Rise (CFL), 2 Heterogeneous Sparse Fabric (HSF),
# 3 Large-Volume Compact Fabric (LVC), 4 Dense Small-Building Fabric (DSB).
GROUP_MAP = {
    4: 1,  5: 1,
    1: 2,  3: 2,  6: 2,  7: 2,  8: 2,
    12: 3, 13: 3,
    2: 4,  9: 4, 10: 4, 11: 4, 14: 4, 15: 4, 16: 4, 17: 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _radial(cx, cy, ref_x, ref_y):
    """Euclidean distance and normalised radial position from one reference."""
    dx   = cx - ref_x
    dy   = cy - ref_y
    dist = np.sqrt(dx * dx + dy * dy).astype("float32")
    p98  = float(np.percentile(dist, 98))
    if p98 < 1.0:
        return None, None, None, None   # degenerate city
    norm = np.minimum(dist / p98, 1.0).astype("float32")

    bins5 = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    ring5 = np.digitize(norm, bins5).astype("int8")   # 1..5

    bins3 = np.array([0.0, 0.333, 0.667])
    ring3 = np.digitize(norm, bins3).astype("int8")   # 1..3

    return dist, norm, ring5, ring3


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------
def _process_city(args):
    (country, city, city_labels_df,
     prim_x, prim_y,
     pop_x,  pop_y,
     geom_x, geom_y) = args

    shp_dir = SHP_ROOT / country / city
    if not shp_dir.exists():
        return None
    shps = list(shp_dir.glob("*.shp"))
    if not shps:
        return None

    try:
        # Load needed columns: GridID + local-CRS centroid + population
        gdf = gpd.read_file(shps[0],
                            columns=["GridID", "center_x", "center_y", "Pop2022"])
        if gdf.empty:
            return None

        # center_x / center_y are already in the city's local projected CRS
        cx = gdf["center_x"].to_numpy(dtype="float64")
        cy = gdf["center_y"].to_numpy(dtype="float64")

        # Merge with label_k17
        lbl = (city_labels_df[["grid_id", "label_k17"]]
               .rename(columns={"grid_id": "GridID"}))
        df = gdf[["GridID", "Pop2022"]].merge(lbl, on="GridID", how="inner")
        if df.empty:
            return None

        # Re-align cx/cy to merged subset using GridID key
        df = df.copy()
        coord_map = gdf.set_index("GridID")[["center_x", "center_y"]]
        df["cx"] = coord_map.loc[df["GridID"], "center_x"].values
        df["cy"] = coord_map.loc[df["GridID"], "center_y"].values

        # Clean Pop2022: negative / nan → 0
        pop2022 = pd.to_numeric(df["Pop2022"], errors="coerce").fillna(0.0).clip(lower=0.0)

        cx_m = df["cx"].to_numpy(dtype="float64")
        cy_m = df["cy"].to_numpy(dtype="float64")

        # --- Three centre versions ---
        dm_p, dn_p, r5_p, r3_p = _radial(cx_m, cy_m, prim_x,  prim_y)
        dm_w, dn_w, r5_w, r3_w = _radial(cx_m, cy_m, pop_x,   pop_y)
        dm_g, dn_g, r5_g, r3_g = _radial(cx_m, cy_m, geom_x,  geom_y)

        if dm_p is None:   # primary degenerate — fallback to geom
            dm_p, dn_p, r5_p, r3_p = dm_g, dn_g, r5_g, r3_g
        if dm_w is None:
            dm_w, dn_w, r5_w, r3_w = dm_g, dn_g, r5_g, r3_g
        if dm_g is None:
            return None   # all degenerate — skip

        group4 = (df["label_k17"].map(GROUP_MAP).fillna(0).astype("int8"))

        return pd.DataFrame({
            "GridID":    df["GridID"].to_numpy(),
            "city":      city,
            "country":   country,
            "label_k17": df["label_k17"].to_numpy().astype("int8"),
            "group4":    group4.to_numpy(),
            "Pop2022":   pop2022.to_numpy(dtype="float32"),
            "cx":        cx_m.astype("float32"),
            "cy":        cy_m.astype("float32"),
            # primary_pop centre (MAIN)
            "dist_m_primary":    dm_p,
            "dist_km_primary":   (dm_p / 1000.0).astype("float32"),
            "dist_norm_primary": dn_p,
            "ring5_primary":     r5_p,
            "ring3_primary":     r3_p,
            # pop_centroid (ROBUSTNESS A)
            "dist_m_pop":    dm_w,
            "dist_km_pop":   (dm_w / 1000.0).astype("float32"),
            "dist_norm_pop": dn_w,
            "ring5_pop":     r5_w,
            "ring3_pop":     r3_w,
            # geom_centroid (ROBUSTNESS B)
            "dist_m_geom":    dm_g,
            "dist_km_geom":   (dm_g / 1000.0).astype("float32"),
            "dist_norm_geom": dn_g,
            "ring5_geom":     r5_g,
            "ring3_geom":     r3_g,
        })

    except Exception as exc:
        logging.warning("  %s / %s failed: %s", country, city, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 14) -> None:
    # Load labels
    logging.info("Loading grid labels (%s) ...", LABELS_PATH.name)
    labels_df = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])
    logging.info("  %d cells.", len(labels_df))

    # Load centres
    logging.info("Loading city centres (%s) ...", CENTRES_PATH.name)
    centres = pd.read_parquet(CENTRES_PATH)
    logging.info("  %d cities.", len(centres))

    # Build centre lookup: (city, country) → (prim_x, prim_y, pop_x, pop_y, geom_x, geom_y)
    # Centres are in WGS84 lon/lat — convert to local CRS is NOT feasible globally.
    # Instead we use the stored _x/_y columns which are in the city's local projected CRS.
    centre_map = {
        (row["city"], row["country"]): (
            row["primary_pop_x"],   row["primary_pop_y"],
            row["pop_centroid_x"],  row["pop_centroid_y"],
            row["geom_centroid_x"], row["geom_centroid_y"],
        )
        for _, row in centres.iterrows()
    }

    # Build task list
    city_groups = labels_df.groupby(["country", "city"])
    tasks = []
    skipped = 0
    for (country, city), grp in city_groups:
        key = (city, country)
        if key not in centre_map:
            skipped += 1
            continue
        prim_x, prim_y, pop_x, pop_y, geom_x, geom_y = centre_map[key]
        tasks.append((country, city, grp.reset_index(drop=True),
                      prim_x, prim_y, pop_x, pop_y, geom_x, geom_y))

    logging.info("  %d cities to process, %d skipped (no centre).",
                 len(tasks), skipped)

    # Parallel processing
    results = []
    done = 0
    logging.info("Computing radial positions with %d workers ...", n_workers)
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

    out_df = pd.concat(results, ignore_index=True)
    logging.info("  Total cells: %d", len(out_df))

    out_path = OUT_DIR / "cell_radial_position.parquet"
    out_df.to_parquet(out_path, index=False)
    logging.info("Saved: %s", out_path)

    # Validation
    logging.info("Validation (primary_pop centre):")
    logging.info("  dist_norm range: [%.4f, %.4f]",
                 out_df["dist_norm_primary"].min(), out_df["dist_norm_primary"].max())
    logging.info("  ring5_primary distribution:\n%s",
                 out_df["ring5_primary"].value_counts().sort_index().to_string())

    # Compare ring5_primary vs ring5_geom agreement
    agree = (out_df["ring5_primary"] == out_df["ring5_geom"]).mean()
    logging.info("  ring5 agreement primary==geom: %.1f%%", agree * 100)

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    main(n_workers=args.workers)
