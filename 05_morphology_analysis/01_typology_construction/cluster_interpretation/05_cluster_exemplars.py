"""
05_cluster_exemplars.py
------------------------
For each of the 17 urban-form clusters, select the N grid cells closest to
the cluster's centroid in the 256-dim M6 embedding space, then collect the
matching composite exemplar images (satellite + building/road overlay +
hexagon mask) for expert/reader inspection (Fig. 1c).

"Closest to centroid" means most typical / least ambiguous for that cluster
— the same notion of representativeness as the clustering itself, since
centroid distance in this embedding space is exactly what the k-means
objective minimises.

Candidate pool: composite images already rendered by the k-selection
pipeline (`../k_selection/outputs/images`,
`../k_selection/within_cluster/outputs/images`,
`../k_selection/outputs/images_fix` — about 2,400 cells across all 17
clusters, rendered via Earth Engine + local building/road SHPs; not
bundled here, see CANDIDATE_IMAGE_DIRS below). Selection is restricted to
cells with an existing image rather than the full ~1M-cell universe, so
the reported centroid is computed over each cluster's available (imaged)
members.

Pipeline
--------
1. Load the continent-scale embedding table and join with
   GridID -> (city, country, label_k17) from the cached t-SNE coordinate
   table (embedding_visualization/outputs/tsne_coords.parquet), restricted
   to GridIDs with a pre-rendered composite image on disk.
2. For each cluster c: compute centroid = mean(embeddings of available
   members), rank those members by Euclidean distance to that centroid,
   take the closest N_EXEMPLARS.
3. Save the selected cell list (outputs/tables/cluster_exemplars.csv).
4. Copy the matching pre-rendered images into
   outputs/exemplar_images/{label_k17:02d}/.

Output
------
outputs/tables/cluster_exemplars.csv
    (17 * N_EXEMPLARS rows, fewer per cluster if unavailable) GridID, city,
    country, label_k17, rank, dist_to_centroid.
outputs/exemplar_images/{label_k17:02d}/{GridID}.png
    1000x1000 RGBA composite images, one sub-folder per cluster.

Usage
-----
    python 05_cluster_exemplars.py [--n 30] [--select-only]
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
TSNE_PARQUET = ROOT.parent / "embedding_visualization" / "outputs" / "tsne_coords.parquet"
OUT_TABLES   = ROOT / "outputs" / "tables"
OUT_IMAGES   = ROOT / "outputs" / "exemplar_images"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# Continent-scale embedding table (one row per GridID). Not produced by
# any script in this pipeline -- download it from
# ../../../04_morphology_embedding/'s release assets.
# Set AFRICA_EMBEDDINGS_PATH to wherever you place the downloaded file.
EMBEDDINGS_PATH = Path(os.environ.get(
    "AFRICA_EMBEDDINGS_PATH",
    "../../../04_morphology_embedding/africa_embeddings.npz",
))

# Pre-rendered composite images from the k-selection pipeline (see module
# docstring). Not bundled here (several GB). Override with
# AFRICA_KSELECTION_IMAGE_DIRS (colon-separated list of directories) if
# your copies live elsewhere.
_default_image_dirs = [
    str(ROOT.parent / "k_selection" / "outputs" / "images"),
    str(ROOT.parent / "k_selection" / "within_cluster" / "outputs" / "images"),
    str(ROOT.parent / "k_selection" / "outputs" / "images_fix"),
]
CANDIDATE_IMAGE_DIRS = [
    Path(p) for p in os.environ.get(
        "AFRICA_KSELECTION_IMAGE_DIRS", ":".join(_default_image_dirs)
    ).split(":")
]

N_EXEMPLARS_DEFAULT = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load the continent-scale embedding table
# ---------------------------------------------------------------------------
def load_all_embeddings():
    import time
    t0 = time.time()
    log.info("Loading embeddings from %s ...", EMBEDDINGS_PATH)
    d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = d["embeddings"].astype(np.float32)
    grid_ids   = d["grid_ids"].astype(str)
    log.info("Loaded %s cells  [%.1fs]", f"{len(embeddings):,}", time.time() - t0)
    return embeddings, grid_ids


def index_existing_images() -> pd.DataFrame:
    """Return (GridID, src_path) for every pre-rendered composite PNG, deduped."""
    records = []
    for d in CANDIDATE_IMAGE_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.png"):
            records.append((p.stem, p))
    df = pd.DataFrame(records, columns=["GridID", "src_path"])
    df = df.drop_duplicates("GridID")
    log.info("Indexed %d existing composite images across %d source dirs",
             len(df), len(CANDIDATE_IMAGE_DIRS))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Select N closest-to-centroid cells per cluster, restricted to
# cells that already have a pre-rendered composite image on disk.
# ---------------------------------------------------------------------------
def select_exemplars(n_exemplars: int) -> pd.DataFrame:
    labels_df = pd.read_parquet(
        TSNE_PARQUET, columns=["GridID", "city", "country", "label_k17"]
    )
    labels_df["GridID"] = labels_df["GridID"].astype(str)
    log.info("Loaded %s labelled cells from %s", f"{len(labels_df):,}", TSNE_PARQUET)

    existing = index_existing_images()
    labels_df = labels_df.merge(existing, on="GridID", how="inner")
    log.info("%s labelled cells have a pre-rendered image available",
             f"{len(labels_df):,}")

    embeddings, grid_ids = load_all_embeddings()
    emb_df = pd.DataFrame({"GridID": grid_ids, "_row": np.arange(len(grid_ids))})

    merged = labels_df.merge(emb_df, on="GridID", how="inner")
    log.info("Matched %s cells with embeddings (%d unmatched)",
             f"{len(merged):,}", len(labels_df) - len(merged))

    rows = []
    for cluster_id in sorted(merged["label_k17"].unique()):
        sub = merged[merged["label_k17"] == cluster_id]
        rows_idx = sub["_row"].to_numpy()
        emb_sub = embeddings[rows_idx]
        # Centroid computed from this cluster's *available* (imaged) members
        # only — see module docstring on why the candidate pool is restricted.
        centroid = emb_sub.mean(axis=0)
        dist = np.linalg.norm(emb_sub - centroid, axis=1)

        n_take = min(n_exemplars, len(sub))
        order = np.argsort(dist)[:n_take]
        picked = sub.iloc[order].copy()
        picked["dist_to_centroid"] = dist[order]
        picked["rank"] = np.arange(1, len(picked) + 1)
        rows.append(picked[["GridID", "city", "country", "label_k17",
                            "rank", "dist_to_centroid", "src_path"]])
        log.info("  Cluster %2d (n_available=%4d): selected %d, "
                 "dist range [%.3f, %.3f]",
                 cluster_id, len(sub), n_take,
                 dist[order].min(), dist[order].max())

    result = pd.concat(rows, ignore_index=True)
    out_csv = OUT_TABLES / "cluster_exemplars.csv"
    result.drop(columns=["src_path"]).to_csv(out_csv, index=False)
    log.info("Saved: %s (%d rows)", out_csv, len(result))
    return result


# ---------------------------------------------------------------------------
# Step 3 — Copy the pre-rendered composite image for each selected exemplar
# into outputs/exemplar_images/{cluster:02d}/.
# ---------------------------------------------------------------------------
def copy_exemplars(exemplars: pd.DataFrame) -> None:
    import shutil

    copied, missing = 0, 0
    for row in exemplars.itertuples():
        out_dir = OUT_IMAGES / f"{int(row.label_k17):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{row.GridID}.png"
        src = Path(row.src_path)
        if not src.exists():
            log.warning("MISSING source image for %s: %s", row.GridID, src)
            missing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    log.info("Done. copied=%d, missing=%d, total=%d", copied, missing, len(exemplars))
    log.info("Images saved to: %s", OUT_IMAGES)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_EXEMPLARS_DEFAULT,
                        help="Exemplars per cluster (default: %(default)s)")
    parser.add_argument("--select-only", action="store_true",
                        help="Only run centroid-distance selection; skip copying images")
    args = parser.parse_args()

    exemplars = select_exemplars(args.n)

    if args.select_only:
        return

    copy_exemplars(exemplars)


if __name__ == "__main__":
    main()
