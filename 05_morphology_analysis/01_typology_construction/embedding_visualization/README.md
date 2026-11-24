# Embedding Visualisation

2D projections of the continent-scale student embeddings, for visual
inspection of cluster separation and how individual morphometric features
vary across the embedding space.

## Method

1. `01_tsne.py` — full-dataset t-SNE (openTSNE, Barnes-Hut, cosine metric)
   on all ~1M embeddings, coloured by k = 17 cluster label (Fig. 1a).
2. `02_feature_overlay.py` — the same t-SNE coordinates, coloured by the
   top-12 morphometric features (ranked by η² in
   `../cluster_interpretation/`) instead of cluster label, as two 2x3
   figures.
3. `03_feature_overlay_combined.py` — the same top-12 feature overlays as
   `02_feature_overlay.py`, combined into the single 4x3 figure published
   as Supplementary Fig. S1.

Embeddings are L2-normalised before projection (cosine metric).
Degenerate embeddings (L2 norm outside the IQR-based range) and projection
outliers (>5σ on either axis) are dropped before plotting; see each
script's docstring for the exact thresholds.

## Data requirements

- Continent-scale student embeddings (`EMBEDDINGS_PATH` in `config.py`),
  a manually-downloaded release asset of `../../../04_morphology_embedding/`.
- `model_input_k17.csv` (`MODEL_INPUT_CSV` in `config.py`), produced by
  `../cluster_interpretation/01_merge_data.py`.
- `../cluster_interpretation/outputs/tables/feature_importance.csv`,
  produced by `../cluster_interpretation/03_feature_importance.py`.
- `../cluster_interpretation/reference/column_name_mapping.csv`
  (bundled) — short-code to full-name mapping, used by
  `03_feature_overlay_combined.py` for panel titles.

## Usage

```bash
python 01_tsne.py
python 02_feature_overlay.py            # requires 01_tsne.py to have run first
python 03_feature_overlay_combined.py   # requires 01_tsne.py to have run first
```

## Outputs

- `outputs/tsne_coords.parquet` — cached 2D coordinates (re-running the
  script skips recomputation if its cache exists).
- `outputs/figures/tsne_k17.*` — cluster-coloured scatter plot (Fig. 1a).
- `outputs/figures/tsne_feature_overlay_1.*` / `_2.*` — feature-coloured
  scatter plots, top-12 features by η² (two 2x3 figures).
- `outputs/figures/tsne_feature_overlay_combined.*` — the same top-12
  feature overlays as one 4x3 figure (Supplementary Fig. S1).
