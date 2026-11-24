"""
01_merge_data.py
----------------
Merge per-city morphological metrics (168 indicators) with k=17 cluster labels
derived from Ward HAC on M6 embeddings.

Outputs
-------
outputs/tables/merged_k17.parquet
    Full merged dataset (~1M grid cells × 172 columns).
    Parquet format preserves dtypes and loads ~10× faster than CSV.

Usage
-----
    python 01_merge_data.py [--workers N]
"""

import argparse
import glob
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# Per-city cleaned morphology CSVs, produced by
# ../../../02_morphology_metrics/. Override with AFRICA_CLEANED_CSV_ROOT
# if your copy lives elsewhere.
CLEANED_DIR = Path(os.environ.get(
    "AFRICA_CLEANED_CSV_ROOT",
    "../../../02_morphology_metrics/outputs/Cleaned",
))
LABELS_CSV = ROOT.parent / "typology_construction" / "outputs" / "clusters" / "grid_type_labels.csv"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_FILE = OUT_DIR / "merged_k17.parquet"

# ---------------------------------------------------------------------------
# Morphological feature columns to retain
# (excludes spatial metadata: GridID, center_x/y, City, Area_m2,
#  Center_X_1, Center_Y_1, and population/land-cover ancillaries)
# The selection below keeps the 168 morphometric indicators used in M6 training.
# ---------------------------------------------------------------------------
_EXCLUDE_META = {"GridID", "center_x", "center_y", "City", "Area_m2",
                 "Center_X_1", "Center_Y_1"}


def _load_city(csv_path: str) -> pd.DataFrame:
    """Load a single city CSV and return only morphometric columns + GridID."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        keep = ["GridID"] + [c for c in df.columns if c not in _EXCLUDE_META]
        return df[keep].copy()
    except Exception as exc:
        logging.warning("Failed to load %s: %s", csv_path, exc)
        return pd.DataFrame()


def load_all_metrics(cleaned_dir: Path, n_workers: int = 8) -> pd.DataFrame:
    """
    Parallel load of all per-city metric CSVs.

    Parameters
    ----------
    cleaned_dir : Path
        Root directory containing per-country subfolders of city CSVs.
    n_workers : int
        Number of parallel reader processes.

    Returns
    -------
    pd.DataFrame
        Concatenated metrics for all grid cells.
    """
    csv_paths = sorted(glob.glob(str(cleaned_dir / "**" / "*.csv"), recursive=True))
    logging.info("Found %d city CSV files.", len(csv_paths))

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_city, p): p for p in csv_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            df = fut.result()
            if not df.empty:
                frames.append(df)
            if i % 500 == 0:
                logging.info("  Loaded %d / %d files.", i, len(csv_paths))

    logging.info("Concatenating %d non-empty city frames.", len(frames))
    metrics = pd.concat(frames, ignore_index=True)
    logging.info("Total grid cells in metrics: %d", len(metrics))
    return metrics


def merge_with_labels(metrics: pd.DataFrame, labels_csv: Path) -> pd.DataFrame:
    """
    Left-join cluster labels onto the metrics frame.

    Parameters
    ----------
    metrics : pd.DataFrame
        Full metrics table with 'GridID' column.
    labels_csv : Path
        CSV with columns [grid_id, city, country, label_k17].

    Returns
    -------
    pd.DataFrame
        Merged frame; rows without a matching label are dropped with a warning.
    """
    labels = pd.read_csv(labels_csv, usecols=["grid_id", "city", "country", "label_k17"])
    labels.rename(columns={"grid_id": "GridID"}, inplace=True)

    # GridID is expected to be globally unique on both sides (it embeds a
    # country code + city index + cell index, e.g. "DZA_0050_000008"), so
    # this merge should never fan out. Assert uniqueness explicitly and
    # pass validate="one_to_one" so a future GridID collision — e.g. a
    # city-numbering clash introduced upstream — raises immediately
    # instead of silently producing a many-to-many join with duplicated
    # rows.
    if not metrics["GridID"].is_unique:
        dupes = metrics.loc[metrics["GridID"].duplicated(keep=False), "GridID"]
        raise ValueError(
            f"metrics['GridID'] is not unique ({dupes.nunique()} duplicated "
            f"values, {len(dupes)} rows) — merge_with_labels assumes a "
            "one-to-one join. Examples: "
            f"{dupes.unique()[:5].tolist()}"
        )
    if not labels["GridID"].is_unique:
        dupes = labels.loc[labels["GridID"].duplicated(keep=False), "GridID"]
        raise ValueError(
            f"labels['GridID'] (from grid_id) is not unique ({dupes.nunique()} "
            f"duplicated values, {len(dupes)} rows) — merge_with_labels assumes "
            f"a one-to-one join. Examples: {dupes.unique()[:5].tolist()}"
        )

    merged = metrics.merge(labels, on="GridID", how="inner", validate="one_to_one")
    n_dropped = len(metrics) - len(merged)
    if n_dropped > 0:
        logging.warning(
            "%d grid cells in metrics had no matching label and were dropped.", n_dropped
        )
    logging.info("Merged dataset: %d grid cells, %d columns.", *merged.shape)
    return merged


def main(n_workers: int = 8) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=== Step 1: Load morphometric metrics ===")
    metrics = load_all_metrics(CLEANED_DIR, n_workers=n_workers)

    logging.info("=== Step 2: Merge with k=17 cluster labels ===")
    merged = merge_with_labels(metrics, LABELS_CSV)

    # Enforce integer label type
    merged["label_k17"] = merged["label_k17"].astype("int8")

    logging.info("=== Step 3: Save to Parquet ===")
    merged.to_parquet(OUT_FILE, index=False, compression="zstd")
    logging.info("Saved: %s  (%.1f MB)", OUT_FILE, OUT_FILE.stat().st_size / 1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metrics with k=17 cluster labels.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel CSV reader processes (default: 8).")
    args = parser.parse_args()
    main(n_workers=args.workers)
