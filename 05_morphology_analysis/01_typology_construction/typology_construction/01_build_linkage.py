"""
01_build_linkage.py
===================
Build a Ward hierarchical agglomerative clustering dendrogram on the
Mini-batch K-Means centroids of the African urban morphology embeddings.

Ward linkage minimises the increase in total within-cluster variance at each
merge. Applied here to 2,000 L2-normalised centroids in Euclidean space;
on unit vectors, Euclidean and cosine distances are monotonically equivalent
(d_E² = 2 · d_cos), so the angular similarity structure of the embeddings
is preserved.

Outputs
-------
outputs/clusters/linkage_matrix.npy    Ward linkage matrix Z  (shape: N-1 × 4)
outputs/clusters/centroids.npy         centroids used (audit copy)
outputs/logs/01_build_linkage.log
"""

import logging
import time
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import CENTROIDS_NPY, CLUSTER_DIR, LOG_DIR, LINKAGE_METHOD, LINKAGE_METRIC

LOG_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / '01_build_linkage.log', mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.info

log('=' * 60)
log('01_BUILD_LINKAGE')
log('=' * 60)

# ── Load centroids ─────────────────────────────────────────────────────────────
log(f'\nLoading centroids: {CENTROIDS_NPY}')
centroids = np.load(CENTROIDS_NPY)
log(f'  Shape: {centroids.shape}  dtype: {centroids.dtype}')

norms = np.linalg.norm(centroids, axis=1)
log(f'  L2 norm — mean: {norms.mean():.6f}  std: {norms.std():.6f}  '
    f'min: {norms.min():.6f}  max: {norms.max():.6f}')

# ── Ward hierarchical clustering ──────────────────────────────────────────────
log(f'\nBuilding linkage (method={LINKAGE_METHOD!r}, metric={LINKAGE_METRIC!r}) …')
t0 = time.time()
Z = linkage(centroids.astype(np.float64), method=LINKAGE_METHOD, metric=LINKAGE_METRIC)
log(f'  Completed in {time.time() - t0:.1f}s')
log(f'  Linkage matrix shape: {Z.shape}')
log(f'  Merge distance range: [{Z[:, 2].min():.4f}, {Z[:, 2].max():.4f}]')

# ── Save ──────────────────────────────────────────────────────────────────────
out_z   = CLUSTER_DIR / 'linkage_matrix.npy'
out_cen = CLUSTER_DIR / 'centroids.npy'

np.save(out_z,   Z)
np.save(out_cen, centroids)

log(f'\nSaved:')
log(f'  {out_z}   ({out_z.stat().st_size / 1e3:.1f} KB)')
log(f'  {out_cen} ({out_cen.stat().st_size / 1e6:.1f} MB)')
log('\nNext: python3 02_assign_labels.py')
