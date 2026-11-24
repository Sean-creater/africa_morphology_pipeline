"""
13_embedding_radial_join.py
============================
Foundational data-build step for the embedding-based center-periphery
framework used throughout extended/13-33. Joins, per grid cell:

  - the 256-dim deep-clustering embedding (L2-normalised), from the
    continent-scale embedding table published by
    ../../../04_morphology_embedding/ (release asset
    africa_embeddings.npz/.parquet, one row per GridID — see
    EMBEDDINGS_PATH below)
  - radial position: q_relative = dist_m_primary / R98_m (same
    definition as 01_radial_coordinates.py's "q" alternative coordinate
    — population-weighted-primary-centre distance, scaled by each city's
    98th-percentile radial extent R98)
  - built-up coverage (LC_Built, LCF_Built — raw land-cover features,
    from merged_k17.parquet, NOT part of the 17-type label)
  - city-level covariates: log_pop, community, R98_km (from
    city_tiers.parquet, n_cells>=50 cities only — same inclusion filter
    as the rest of extended/)

Embeddings are stored as a separate float32 .npy array (NOT inlined into
the parquet) — 1,000,200 cells x 256 dims as float64-in-parquet would be
slow to query repeatedly; a row-aligned .npy lets every downstream script
do cheap numpy indexing/vector ops directly.

The join's inclusion filters (n_cells>=50 cities, valid embedding and
radial position) yield 885,428 cells across 2,504 cities, the sample
used by every downstream extended/ script.

Output
------
  outputs/tables/cell_embeddings_index.parquet
      GridID | city | country | row_idx | q_relative | dist_m_primary
      | label_k17 | Pop2022 | LC_Built | LCF_Built | log_pop | R98_km
      | community
      row_idx indexes into cell_embeddings.npy (same row order).
  outputs/arrays/cell_embeddings.npy
      (N, 256) float32, row i corresponds to row_idx == i above.

Usage
-----
    python 13_embedding_radial_join.py
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent
PART4_DIR   = ROOT.parent / "core" / "outputs" / "tables"
OUT_TABLES  = ROOT / "outputs" / "tables"
OUT_ARRAYS  = ROOT / "outputs" / "arrays"

CELL_PQ      = PART4_DIR / "cell_radial_position.parquet"
R98_PQ       = PART4_DIR / "city_R98.parquet"
PARAMS_JSON  = PART4_DIR / "allometric_params.json"
TIERS_PQ     = OUT_TABLES / "city_tiers.parquet"
MERGED_K17   = (ROOT.parent.parent / "01_typology_construction" / "cluster_interpretation"
               / "outputs" / "tables" / "merged_k17.parquet")

# Continent-scale embedding table (africa_embeddings.npz, one row per
# GridID). Not produced by any script in this pipeline -- download it
# from ../../../04_morphology_embedding/'s release assets.
# Set AFRICA_EMBEDDINGS_PATH to wherever you place the downloaded file.
EMBEDDINGS_PATH = Path(os.environ.get(
    "AFRICA_EMBEDDINGS_PATH",
    "../../../04_morphology_embedding/africa_embeddings.npz",
))

EMB_DIM = 256


def _load_all_embeddings(cities: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Read the continent-scale embedding table (GridID, city, country,
    256-dim embedding per row) and restrict to cities present in
    `cities` (city, country). Returns (index_df, emb_array) where
    index_df has one row per grid cell with GridID/city/country and
    emb_array[i] is that row's 256-dim embedding.
    """
    d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    index_df = pd.DataFrame({
        "GridID":  d["grid_ids"],
        "city":    d["city"],
        "country": d["country"],
    })
    emb_array = d["embeddings"].astype(np.float32)

    wanted = set(zip(cities["country"], cities["city"]))
    mask = [(c, ci) in wanted for c, ci in zip(index_df["country"], index_df["city"])]
    index_df = index_df[mask].reset_index(drop=True)
    emb_array = emb_array[mask]

    n_cities_found = index_df.drop_duplicates(["city", "country"]).shape[0]
    log.info("  %d cities with embeddings matched out of %d requested.",
             n_cities_found, len(wanted))
    return index_df, emb_array


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_ARRAYS.mkdir(parents=True, exist_ok=True)

    log.info("Loading city_tiers.parquet (n_cells>=50 city set + log_pop) ...")
    tiers = pd.read_parquet(TIERS_PQ,
                            columns=["city", "country", "log_pop",
                                     "community", "R98_km"])
    log.info("  %d cities.", len(tiers))

    log.info("Loading embeddings for these cities (this scans the "
            "Student_embeddings directory tree, ~7900 city folders) ...")
    emb_index, emb_array = _load_all_embeddings(tiers)
    log.info("  %d cells with embeddings, dim=%d.",
             len(emb_index), emb_array.shape[1])

    log.info("Loading cell_radial_position.parquet (GridID, distances, Pop2022) ...")
    cells = pd.read_parquet(
        CELL_PQ,
        columns=["GridID", "city", "country", "label_k17",
                 "Pop2022", "dist_m_primary"])

    log.info("Loading R98 for q_relative = dist_m_primary / R98_m ...")
    r98 = pd.read_parquet(R98_PQ, columns=["city", "country", "R98_km"]) \
            .drop_duplicates(["city", "country"])
    r98["R98_m"] = r98["R98_km"] * 1000.0
    cells = cells.merge(r98[["city", "country", "R98_m"]],
                        on=["city", "country"], how="inner")
    cells["q_relative"] = cells["dist_m_primary"] / cells["R98_m"]
    cells.drop(columns=["R98_m"], inplace=True)

    log.info("Loading built-up coverage (LC_Built, LCF_Built) from merged_k17.parquet ...")
    built = pd.read_parquet(MERGED_K17, columns=["GridID", "LC_Built", "LCF_Built"])

    log.info("Joining: embeddings <- cells (GridID) <- built-up coverage (GridID) "
            "<- city covariates (city, country) ...")
    # Keep the original positional index into emb_array through every
    # merge, so the embedding array can be re-sliced exactly once at the
    # very end (merges below only ever drop/reorder rows, never duplicate
    # a GridID, since GridID is unique within a city and the join keys
    # are exact matches on GridID/city/country).
    emb_index["_emb_row"] = np.arange(len(emb_index))
    joined = emb_index.merge(
        cells[["GridID", "city", "country", "label_k17", "Pop2022", "q_relative"]],
        on=["GridID", "city", "country"], how="inner")
    joined = joined.merge(built, on="GridID", how="left")
    joined = joined.merge(tiers, on=["city", "country"], how="inner")

    log.info("  %d cells after all joins (from %d embedded cells, %d radial cells).",
             len(joined), len(emb_index), len(cells))

    n_miss_built = joined["LCF_Built"].isna().sum()
    n_miss_q     = joined["q_relative"].isna().sum()
    log.info("QC — missing LCF_Built: %d (%.2f%%), missing q_relative: %d (%.2f%%)",
             n_miss_built, 100 * n_miss_built / len(joined),
             n_miss_q, 100 * n_miss_q / len(joined))

    # Slice the embedding array to the final joined row set/order, then
    # replace _emb_row with a fresh positional row_idx (0..N-1) that
    # indexes directly into the saved array.
    joined = joined.reset_index(drop=True)
    emb_final = emb_array[joined["_emb_row"].values]
    joined = joined.drop(columns=["_emb_row"])
    joined["row_idx"] = np.arange(len(joined))

    out_npy = OUT_ARRAYS / "cell_embeddings.npy"
    np.save(out_npy, emb_final)
    log.info("Saved: %s  shape=%s", out_npy, emb_final.shape)

    out_pq = OUT_TABLES / "cell_embeddings_index.parquet"
    joined.to_parquet(out_pq, index=False)
    log.info("Saved: %s  (%d rows x %d cols)", out_pq, len(joined), len(joined.columns))

    log.info("Done.")


if __name__ == "__main__":
    main()
