"""
03_kselection_diagnostics.py
============================
Compute internal clustering quality metrics across k=2–30 to support
expert-guided selection of the optimal number of morphological types.

Metrics
-------
- Silhouette score (cosine distance, subsampled)
  Measures mean intra-cluster cohesion vs inter-cluster separation.
- Calinski–Harabasz index (Euclidean)
  Ratio of between-cluster to within-cluster dispersion; higher is better.
- Davies–Bouldin index (Euclidean)
  Mean similarity of each cluster to its most similar neighbour; lower is better.

These metrics provide complementary quantitative guidance. The final k is
determined by expert pairwise evaluation, cross-checked for stability across
neighbouring k (see ../k_selection/).

Outputs
-------
outputs/figures/kselection_curves.pdf/.png
outputs/kselection_metrics.csv
outputs/logs/03_kselection_diagnostics.log
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBEDDINGS_PATH, CLUSTER_DIR, LOG_DIR, FIG_DIR,
    K_LIST, K_FOCAL, SIL_SAMPLE, RANDOM_SEED,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / '03_kselection_diagnostics.log', mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.info

log('=' * 60)
log('03_KSELECTION_DIAGNOSTICS')
log('=' * 60)

# ── Load embeddings ────────────────────────────────────────────────────────────
log(f'\nLoading embeddings from:\n  {EMBEDDINGS_PATH}')
t0 = time.time()
d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
embeddings = d['embeddings'].astype(np.float32)
grid_ids   = d['grid_ids'].astype(str)
N = len(grid_ids)
log(f'  Loaded {N:,} cells in {time.time() - t0:.1f}s')

# ── Load type labels ───────────────────────────────────────────────────────────
labels_path = CLUSTER_DIR / 'grid_type_labels.csv'
assert labels_path.exists(), f'Run 02_assign_labels.py first: {labels_path}'

log(f'\nLoading type labels: {labels_path}')
df_labels = pd.read_csv(labels_path)

# Align embeddings and labels on grid_id
df_idx = pd.DataFrame({'grid_id': grid_ids, 'emb_idx': np.arange(N)})
df_labels = df_idx.merge(df_labels, on='grid_id', how='inner')
N_aligned = len(df_labels)
log(f'  Aligned: {N_aligned:,} cells')

emb = embeddings[df_labels['emb_idx'].values]

# ── Subsample for silhouette ───────────────────────────────────────────────────
log(f'\nSubsampling {SIL_SAMPLE:,} cells for silhouette …')
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(N_aligned, size=SIL_SAMPLE, replace=False)
emb_sample = emb[sample_idx]

# ── Compute metrics across k ───────────────────────────────────────────────────
log('\nComputing metrics …')
sil_list, ch_list, db_list = [], [], []

for k in K_LIST:
    col    = f'label_k{k:02d}'
    labels = df_labels[col].values

    sil = silhouette_score(emb_sample, labels[sample_idx], metric='cosine')
    ch  = calinski_harabasz_score(emb, labels)
    db  = davies_bouldin_score(emb, labels)

    sil_list.append(sil)
    ch_list.append(ch)
    db_list.append(db)

    marker = '  ← expert-selected k' if k == K_FOCAL else ''
    log(f'  k={k:2d}:  silhouette={sil:.4f}  CH={ch:.1f}  DB={db:.4f}{marker}')

# ── Save metrics table ─────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    'k':                    K_LIST,
    'silhouette_cosine':    sil_list,
    'calinski_harabasz':    ch_list,
    'davies_bouldin':       db_list,
})
out_csv = Path(__file__).parent / 'outputs' / 'kselection_metrics.csv'
df_out.to_csv(out_csv, index=False)
log(f'\nSaved metrics table: {out_csv}')

# ── Figure ─────────────────────────────────────────────────────────────────────
log('\nPlotting …')
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

panels = [
    (sil_list, 'Silhouette score (cosine)\n↑ higher = better separated clusters'),
    (ch_list,  'Calinski–Harabász index\n↑ higher = more compact and separated'),
    (db_list,  'Davies–Bouldin index\n↓ lower = better cluster quality'),
]

for ax, (vals, ylabel) in zip(axes, panels):
    ax.plot(K_LIST, vals, color='#2C6FAC', linewidth=2, marker='o', markersize=5)
    ax.axvline(K_FOCAL, color='#E5152C', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.text(K_FOCAL + 0.3, ax.get_ylim()[0], f'k = {K_FOCAL}',
            fontsize=9, color='#E5152C', va='bottom')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(K_LIST)
    ax.set_xticklabels(K_LIST, fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Number of morphological types (k)', fontsize=11)
fig.suptitle(
    'Clustering quality across candidate typologies (k = 2–30)\n'
    f'Ward hierarchical clustering · {N_aligned:,} African grid cells',
    fontsize=12, fontweight='bold',
)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = FIG_DIR / f'kselection_curves.{ext}'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    log(f'  Saved: {p}')
plt.close()

log(f'\nSummary at expert-selected k={K_FOCAL}:')
idx = K_LIST.index(K_FOCAL)
log(f'  Silhouette (cosine): {sil_list[idx]:.4f}')
log(f'  Calinski–Harabász:   {ch_list[idx]:.1f}')
log(f'  Davies–Bouldin:      {db_list[idx]:.4f}')
log('\nAll done.')
