"""
sample_within_pairs.py
-----------------------
Sample within-cluster pairs for each candidate k, to be combined with the
existing cross-boundary pairs into a discrimination score:

    score(k) = P(no | cross) - P(no | within)

This difference removes a systematic bias that a cross-boundary-only score
has at low k:
- at low k, clusters are large and internally heterogeneous, so
  within-cluster "no" (not the same type) rates are also high, compressing
  the difference;
- at high k, clusters are small and internally homogeneous, so
  within-cluster "no" rates are low, and the difference more faithfully
  tracks true discriminative power;
- the optimal k maximises the difference, i.e. jointly maximises
  within-cluster cohesion and between-cluster separation.

Sampling design:
- for each k, sample pairs of grid cells from within each cluster;
- every cluster gets at least one pair, with additional pairs allocated in
  proportion to cluster size;
- pairs are drawn from the middle third of the within-cluster distance
  distribution, avoiding both near-duplicate and outlier pairs;
- 20 pairs per k, matching the cross-boundary pair count.

Outputs:
- outputs/within_pairs_k02_k30.csv
- outputs/pairs_for_annotation/within_kXX_YY/  (A_xxx.png + B_xxx.png)

Depends on cached files produced by
../sampling/build_dendrogram_and_sample.py:
- ../outputs/dendrogram/centroids.npy
- ../outputs/dendrogram/centroid_labels.npy
- ../outputs/dendrogram/linkage_matrix.npy
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # within_cluster/

# Continent-scale embedding table (one row per GridID). Not produced by
# any script in this pipeline -- download it from
# ../../../../04_morphology_embedding/'s release assets.
# Set AFRICA_EMBEDDINGS_PATH to wherever you place the downloaded file.
EMBEDDINGS_PATH = Path(os.environ.get(
    "AFRICA_EMBEDDINGS_PATH",
    "../../../../04_morphology_embedding/africa_embeddings.npz",
))
DENDRO_DIR    = ROOT.parent / "outputs" / "dendrogram"   # reuses existing cache
OUT_CSV       = ROOT / "outputs" / "within_pairs_k02_k30.csv"
OUT_ANNOT_DIR = ROOT / "outputs" / "pairs_for_annotation"
OUT_IMAGES    = ROOT / "outputs" / "images"
LOG_DIR       = ROOT.parent / "logs"

for d in [OUT_ANNOT_DIR, OUT_IMAGES, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_MIN        = 2
K_MAX        = 30
PAIRS_PER_K  = 20    # matches the cross-boundary pair count
RANDOM_SEED  = 123   # distinct from the cross-boundary seed, avoids overlap

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sample_within_pairs.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load embeddings
# ---------------------------------------------------------------------------
log.info("Loading embeddings …")
d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
embeddings = normalize(d["embeddings"].astype(np.float32), norm="l2")
grid_ids   = d["grid_ids"].astype(str)
cities     = d["city"]
countries  = d["country"]
log.info(f"Loaded {len(grid_ids):,} cells, shape={embeddings.shape}")


# ---------------------------------------------------------------------------
# Step 2: Load cached centroids and linkage (not recomputed)
# ---------------------------------------------------------------------------
log.info("Loading cached centroids and linkage …")
centroids           = normalize(np.load(DENDRO_DIR / "centroids.npy").astype(np.float32), norm="l2")
centroid_labels_all = np.load(DENDRO_DIR / "centroid_labels.npy")
Z                   = np.load(DENDRO_DIR / "linkage_matrix.npy")
log.info(f"Centroids: {centroids.shape}, linkage: {Z.shape}")

# centroid -> underlying grid-cell index mapping
N_CENTROIDS = len(centroids)
centroid_to_indices: dict[int, list[int]] = {i: [] for i in range(N_CENTROIDS)}
for cell_idx, cent_lbl in enumerate(centroid_labels_all):
    centroid_to_indices[cent_lbl].append(cell_idx)


# ---------------------------------------------------------------------------
# Step 3: For each k, sample within-cluster pairs
# ---------------------------------------------------------------------------
rng  = np.random.default_rng(RANDOM_SEED)
rows = []

log.info(f"Sampling {PAIRS_PER_K} within-cluster pairs per k={K_MIN}..{K_MAX} …")

for k in range(K_MIN, K_MAX + 1):
    cent_labels_k   = fcluster(Z, k, criterion="maxclust")  # 1-indexed
    unique_clusters = np.unique(cent_labels_k)

    # per-cluster cell pool
    cluster_pools: dict[int, list[int]] = {}
    for clbl in unique_clusters:
        cents = np.where(cent_labels_k == clbl)[0]
        pool  = [i for c in cents for i in centroid_to_indices[c]]
        if len(pool) >= 2:
            cluster_pools[clbl] = pool

    if not cluster_pools:
        log.warning(f"k={k}: no cluster with ≥2 cells, skipping")
        continue

    # allocate quota proportional to cluster size; every cluster gets >=1 pair
    total_cells = sum(len(p) for p in cluster_pools.values())
    quota: dict[int, int] = {}
    remaining = PAIRS_PER_K
    clbls = sorted(cluster_pools.keys())

    for clbl in clbls:
        quota[clbl] = 1
        remaining  -= 1

    for clbl in sorted(clbls, key=lambda c: -len(cluster_pools[c])):
        if remaining <= 0:
            break
        add = min(remaining,
                  max(0, round(len(cluster_pools[clbl]) / total_cells * PAIRS_PER_K)))
        quota[clbl] += add
        remaining   -= add

    if remaining > 0:
        quota[clbls[0]] += remaining

    pair_idx = 0
    for clbl in clbls:
        n_pairs = min(quota[clbl], len(cluster_pools[clbl]) // 2)
        if n_pairs == 0:
            continue
        pool = cluster_pools[clbl]

        # select pairs from the middle third of the embedding-distance distribution
        candidate_n = min(len(pool), max(50, n_pairs * 10))
        cands       = rng.choice(pool, size=candidate_n, replace=False)
        cand_embs   = embeddings[cands]

        pair_dists = [
            (i, j, float(1.0 - np.dot(cand_embs[i], cand_embs[j])))
            for i in range(len(cands))
            for j in range(i + 1, len(cands))
        ]
        if not pair_dists:
            continue

        pair_dists.sort(key=lambda x: x[2])
        lo = len(pair_dists) // 3
        hi = 2 * len(pair_dists) // 3
        mid_pairs = pair_dists[lo:hi] if hi > lo else pair_dists

        selected, used = [], set()
        rng.shuffle(mid_pairs)  # type: ignore[arg-type]
        for i, j, dist in mid_pairs:
            if cands[i] not in used and cands[j] not in used:
                selected.append((cands[i], cands[j], dist))
                used |= {cands[i], cands[j]}
            if len(selected) >= n_pairs:
                break

        for ia, ib, dist in selected:
            pair_idx += 1
            rows.append({
                "split_id":        f"within_k{k:02d}_{pair_idx:02d}",
                "k":               k,
                "cluster_label":   int(clbl),
                "pair_index":      pair_idx,
                "grid_id_a":       grid_ids[ia],
                "grid_id_b":       grid_ids[ib],
                "city_a":          cities[ia],
                "city_b":          cities[ib],
                "country_a":       countries[ia],
                "country_b":       countries[ib],
                "cosine_dist":     round(float(dist), 4),
                "expert_judgment": "",
            })

    actual = sum(1 for r in rows if r["k"] == k)
    log.info(f"k={k:>2}: {len(cluster_pools)} clusters → {actual} within-pairs")


# ---------------------------------------------------------------------------
# Step 4: Save CSV
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
log.info(f"\nTotal: {len(df)} within-pairs → {OUT_CSV}")


# ---------------------------------------------------------------------------
# Step 5: Copy existing images into pairs_for_annotation/within_kXX_YY/
# ---------------------------------------------------------------------------
log.info("Copying existing images to annotation folders …")
cross_images = ROOT.parent / "outputs" / "images"  # cross-boundary-pair image directory

def find_image(grid_id: str) -> Path | None:
    for base in [OUT_IMAGES, cross_images]:
        hits = list(base.rglob(f"{grid_id}.png"))
        if hits:
            return hits[0]
    return None

copied = missing = 0
for _, row in df.iterrows():
    folder = OUT_ANNOT_DIR / row["split_id"]
    folder.mkdir(exist_ok=True)
    for side, gid in [("A", row["grid_id_a"]), ("B", row["grid_id_b"])]:
        src = find_image(gid)
        if src is None:
            missing += 1
        else:
            shutil.copy2(src, folder / f"{side}_{gid}.png")
            copied += 1

log.info(f"Images: {copied} copied, {missing} missing (need generate_images.py)")
log.info("Done.")
