# Cluster Interpretation

Characterises the k = 17 morphological types in terms of the underlying
morphometric indicators: which features distinguish the clusters, and how
each cluster deviates from the continent-wide average on those features.

## Method

1. `01_merge_data.py` — merges the 168 per-cell morphometric indicators
   with the k = 17 type labels from `../typology_construction/`.
2. `02_deduplicate_features.py` — removes redundant indicators via
   Spearman-correlation hierarchical clustering (complete linkage,
   |ρ| ≥ 0.90), retaining one representative per collinear group, ranked
   by η².
3. `03_feature_importance.py` — ranks the deduplicated features by global
   discriminative power (η², after rank-based Inverse Normal Transform)
   across the 17 clusters.
4. `04_cluster_profiles.py` — for each cluster and feature, the signed
   standardised mean deviation from the continent-wide average
   (D_cj), giving each cluster's morphological profile relative to the
   full dataset (Fig. 1b).
5. `05_cluster_exemplars.py` — for each cluster, the grid cells closest
   to its centroid in embedding space, with their pre-rendered composite
   images (Fig. 1c).

η² is reported as a descriptive variance-decomposition ratio; no p-values
are computed, consistent with the descriptive framework used throughout
this analysis at continental (~1M-cell) scale. See the manuscript's
Methods and Supplementary Information for the full statistical treatment.

## Data requirements

- Per-city cleaned morphology CSVs (`CLEANED_DIR` in `01_merge_data.py`),
  produced by `../../02_morphology_metrics/`.
- `grid_type_labels.csv`, produced by
  `../typology_construction/02_assign_labels.py`.
- `reference/column_name_mapping.csv` (bundled) — short-code to
  full-name mapping for the 164 urban form metrics (Supplementary
  Table S1's data source), used by `04_cluster_profiles.py` for the
  heatmap's feature-name labels.
- `../embedding_visualization/outputs/tsne_coords.parquet` (for
  `05_cluster_exemplars.py`), produced by
  `../embedding_visualization/01_tsne.py`.
- Pre-rendered composite images from the k-selection pipeline (for
  `05_cluster_exemplars.py`; not bundled, several GB — see its
  docstring and `AFRICA_KSELECTION_IMAGE_DIRS`).

## Usage

```bash
python 01_merge_data.py
python 02_deduplicate_features.py
python 03_feature_importance.py
python 04_cluster_profiles.py
python 05_cluster_exemplars.py
```

## Outputs

- `outputs/tables/model_input_k17.csv` — merged indicator + label table,
  the shared input to steps 2–4 and to `../embedding_visualization/`.
- `outputs/tables/feature_groups.csv` / `selected_features.csv` — the
  redundancy-grouped and deduplicated feature lists.
- `outputs/tables/feature_importance.csv` — deduplicated features ranked
  by η².
- `outputs/tables/cluster_profiles.csv` / `cluster_sizes.csv` — per-cluster
  morphological profile (D_cj matrix) and grid-cell counts.
- `outputs/figures/cluster_profiles_heatmap.svg` — Fig. 1b.
- `outputs/tables/cluster_exemplars.csv` / `outputs/exemplar_images/` —
  Fig. 1c's representative grid cells and their composite images.
