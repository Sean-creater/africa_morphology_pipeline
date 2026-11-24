"""
build_dendrogram_and_sample.py
-------------------------------
1. Load full-Africa M6 embeddings (1,000,207 x 256) from the continent-
   scale embedding table.
2. Subsample to N_CENTROIDS representative points via Mini-batch K-Means.
   Mini-batch K-Means gives compact, well-separated cluster centres that
   uniformly cover the embedding space — better seeds for Ward HAC than
   random subsampling and faster than FPS at this scale.
3. Run Ward HAC on the K-Means centroids → dendrogram.
4. For each split point k = K_MIN .. K_MAX, sample PAIRS_PER_SPLIT
   cross-boundary pairs uniformly at random from the FULL cell pool
   (all 1M cells, not just centroids) of each child cluster.
5. Export to CSV for expert annotation.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Continent-scale embedding table (one row per GridID). Not produced by
# any script in this pipeline -- download it from
# ../../../../04_morphology_embedding/'s release assets.
# Set AFRICA_EMBEDDINGS_PATH to wherever you place the downloaded file.
EMBEDDINGS_PATH = Path(os.environ.get(
    "AFRICA_EMBEDDINGS_PATH",
    "../../../../04_morphology_embedding/africa_embeddings.npz",
))

OUT_DENDRO_DIR = ROOT / "outputs" / "dendrogram"
OUT_PAIRS_DIR  = ROOT / "outputs" / "split_pairs"
LOG_DIR        = ROOT / "logs"

for d in [OUT_DENDRO_DIR, OUT_PAIRS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CENTROIDS     = 2_000   # K-Means cluster centres for HAC
K_MIN           = 2       # first k to examine
K_MAX           = 30      # last k to examine
PAIRS_PER_SPLIT = 20      # cross-boundary pairs per split point
RANDOM_SEED     = 42

# Mini-batch K-Means parameters
# n_init=3  → 3 random initialisations, keep best inertia
# max_iter=300 → allow thorough convergence
# batch_size=131072 → ~128 K samples per mini-batch (fast yet stable)
# reassignment_ratio=0.01 → standard; allows under-used centres to be reset
MBKM_N_INIT           = 3
MBKM_MAX_ITER         = 300
MBKM_BATCH_SIZE       = 131_072
MBKM_REASSIGN_RATIO   = 0.01


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_dendrogram_and_sample.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a_n, b_n))


# ---------------------------------------------------------------------------
# Step 1: Load the continent-scale embedding table
# ---------------------------------------------------------------------------
log.info(f"Loading embeddings from {EMBEDDINGS_PATH} …")
d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
embeddings = d["embeddings"].astype(np.float32)
grid_ids   = d["grid_ids"].astype(str)
cities     = d["city"]
countries  = d["country"]
log.info(f"  Loaded {len(grid_ids):,} cells  shape={embeddings.shape}")

embeddings = normalize(embeddings, norm="l2")

# ---------------------------------------------------------------------------
# Step 2: Mini-batch K-Means → N_CENTROIDS representative centres
# Cached: if centroids.npy and centroid_labels.npy exist, skip refit.
# ---------------------------------------------------------------------------
CENTROIDS_NPY       = OUT_DENDRO_DIR / "centroids.npy"
CENTROID_LABELS_NPY = OUT_DENDRO_DIR / "centroid_labels.npy"
LINKAGE_NPY         = OUT_DENDRO_DIR / "linkage_matrix.npy"

if CENTROIDS_NPY.exists() and CENTROID_LABELS_NPY.exists():
    log.info("Loading cached centroids and centroid labels …")
    centroids           = np.load(CENTROIDS_NPY).astype(np.float32)
    centroid_labels_all = np.load(CENTROID_LABELS_NPY)
    log.info(f"  centroids={centroids.shape}, labels={centroid_labels_all.shape}")
else:
    log.info(
        f"Running Mini-batch K-Means: k={N_CENTROIDS}, "
        f"n_init={MBKM_N_INIT}, max_iter={MBKM_MAX_ITER}, "
        f"batch_size={MBKM_BATCH_SIZE} …"
    )
    mbkm = MiniBatchKMeans(
        n_clusters=N_CENTROIDS,
        n_init=MBKM_N_INIT,
        max_iter=MBKM_MAX_ITER,
        batch_size=MBKM_BATCH_SIZE,
        reassignment_ratio=MBKM_REASSIGN_RATIO,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    centroid_labels_all = mbkm.fit_predict(embeddings)
    centroids = mbkm.cluster_centers_.astype(np.float32)
    centroids = normalize(centroids, norm="l2")
    np.save(CENTROIDS_NPY, centroids)
    np.save(CENTROID_LABELS_NPY, centroid_labels_all)
    log.info(f"Mini-batch K-Means done. Inertia = {mbkm.inertia_:.4f}")

# Map centroid index → original cell indices (for sampling real grid cells)
centroid_to_indices: dict[int, list[int]] = {i: [] for i in range(N_CENTROIDS)}
for cell_idx, cent_lbl in enumerate(centroid_labels_all):
    centroid_to_indices[cent_lbl].append(cell_idx)

sizes = [len(v) for v in centroid_to_indices.values()]
log.info(
    f"Centroid→cell mapping done. Sizes — mean: {np.mean(sizes):.0f}, "
    f"min: {min(sizes)}, max: {max(sizes)}"
)

# ---------------------------------------------------------------------------
# Step 3: Ward HAC on K-Means centroids
# Cached: if linkage_matrix.npy exists, skip recompute.
# ---------------------------------------------------------------------------
if LINKAGE_NPY.exists():
    log.info("Loading cached linkage matrix …")
    Z = np.load(LINKAGE_NPY)
else:
    log.info("Computing pairwise cosine distances for HAC …")
    dist_condensed = pdist(centroids, metric="cosine")
    log.info("Running Ward linkage …")
    Z = linkage(dist_condensed, method="ward")
    np.save(LINKAGE_NPY, Z)
    log.info(f"Linkage matrix saved → {LINKAGE_NPY}")

    fig, ax = plt.subplots(figsize=(20, 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(
        f"Ward HAC Dendrogram — M6 Triplet-Distilled Embeddings "
        f"({N_CENTROIDS} Mini-batch K-Means centroids)"
    )
    ax.set_xlabel("Centroid index")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(OUT_DENDRO_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    log.info(f"Dendrogram figure saved → {OUT_DENDRO_DIR / 'dendrogram.png'}")

# ---------------------------------------------------------------------------
# Step 4: Sample cross-boundary pairs at each HAC split k_from → k_from+1
#
# The HAC tree on N_CENTROIDS centroids is strictly hierarchical: exactly one
# cluster splits at each step. We use fcluster() on the centroids to identify
# which centroid group splits, then map centroid membership back to real grid
# cells via centroid_to_indices. This guarantees the split pools are exactly
# the two child subtrees of the HAC dendrogram — a well-defined, nested
# partition of the embedding space at each resolution.
# ---------------------------------------------------------------------------
log.info(f"Sampling {PAIRS_PER_SPLIT} pairs per split for k={K_MIN}→{K_MAX} …")
rng  = np.random.default_rng(RANDOM_SEED)
rows = []

for k_from in range(K_MIN - 1, K_MAX):
    k_to = k_from + 1

    # fcluster labels are 1-indexed and assigned to centroids (not grid cells)
    labels_before = fcluster(Z, k_from, criterion="maxclust")
    labels_after  = fcluster(Z, k_to,   criterion="maxclust")

    # Find the one centroid-group whose members split into exactly 2 children
    split_parent  = None
    child_a_label = None
    child_b_label = None
    for parent_lbl in np.unique(labels_before):
        mask     = labels_before == parent_lbl
        children = np.unique(labels_after[mask])
        if len(children) == 2:
            split_parent  = parent_lbl
            child_a_label = int(children[0])
            child_b_label = int(children[1])
            break

    if split_parent is None:
        log.warning(f"k={k_from}→{k_to}: could not identify HAC split, skipping.")
        continue

    # Centroid indices belonging to each child subtree
    child_a_cents = np.where(
        (labels_before == split_parent) & (labels_after == child_a_label))[0]
    child_b_cents = np.where(
        (labels_before == split_parent) & (labels_after == child_b_label))[0]

    # Map centroid membership → real grid cell indices
    pool_a = [i for c in child_a_cents for i in centroid_to_indices[c]]
    pool_b = [i for c in child_b_cents for i in centroid_to_indices[c]]

    n_pairs  = min(PAIRS_PER_SPLIT, len(pool_a), len(pool_b))
    sample_a = rng.choice(pool_a, size=n_pairs, replace=False)
    sample_b = rng.choice(pool_b, size=n_pairs, replace=False)

    centroid_a = centroids[child_a_cents].mean(axis=0)
    centroid_b = centroids[child_b_cents].mean(axis=0)
    cent_dist  = cosine_distance(centroid_a, centroid_b)

    for pair_idx in range(n_pairs):
        ia, ib = int(sample_a[pair_idx]), int(sample_b[pair_idx])
        rows.append({
            "split_id":        f"split_k{k_to:02d}_{pair_idx + 1:02d}",
            "k_from":          k_from,
            "k_to":            k_to,
            "split_index":     pair_idx + 1,
            "grid_id_a":       grid_ids[ia],
            "grid_id_b":       grid_ids[ib],
            "city_a":          cities[ia],
            "city_b":          cities[ib],
            "country_a":       countries[ia],
            "country_b":       countries[ib],
            "expert_judgment": "",
        })

    log.info(
        f"k={k_from}→{k_to}: split parent={split_parent}, "
        f"|A|={len(pool_a):,} |B|={len(pool_b):,}, "
        f"centroid_dist={cent_dist:.4f}, pairs sampled={n_pairs}"
    )

# ---------------------------------------------------------------------------
# Step 5: Export CSV
# ---------------------------------------------------------------------------
# Step 5: Export CSV
# ---------------------------------------------------------------------------
out_csv = OUT_PAIRS_DIR / "split_pairs_k02_to_k30.csv"
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
log.info(f"\nDone. {len(df)} pairs written → {out_csv}")
log.info(
    f"Splits covered: k={df['k_to'].min()}→{df['k_to'].max()}, "
    f"{df['k_to'].nunique()} split points, "
    f"{df['grid_id_a'].nunique() + df['grid_id_b'].nunique()} unique grid cells"
)
