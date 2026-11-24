# Typology Construction

Builds a hierarchical typology of African urban morphology from the
continent-scale student embeddings, and assigns a morphological type label
to every grid cell.

## Method

1. `01_build_linkage.py` — Ward hierarchical agglomerative clustering
   (Euclidean distance) on 2,000 Mini-batch K-Means centroids of the
   L2-normalised 256-dim embeddings (for tractability at continental
   scale).
2. `02_assign_labels.py` — every grid cell is mapped to its nearest
   centroid, then the dendrogram is cut at each candidate k = 2–30 to
   produce a full set of candidate type labels per cell.
3. `03_kselection_diagnostics.py` — internal clustering-quality metrics
   (silhouette, Calinski–Harabász, Davies–Bouldin) across k = 2–30, as
   complementary quantitative context alongside the expert evaluation in
   `../k_selection/`.

k = 17 is the number of morphological types used throughout this
repository and the companion analysis modules. It was selected via expert
pairwise evaluation of sample cell pairs at each candidate split, and
cross-checked for stability against neighbouring k (see `../k_selection/`
for the sampling/evaluation materials). See the manuscript's Methods and
Supplementary Information for the full selection procedure.

## Data requirements

- Continent-scale student embeddings (`EMBEDDINGS_ROOT` in `config.py`),
  produced by `../../../04_morphology_embedding/`.
- K-Means centroids (`CENTROIDS_NPY` in `config.py`), produced by
  `../k_selection/sampling/build_dendrogram_and_sample.py`.

## Usage

```bash
python 01_build_linkage.py
python 02_assign_labels.py
python 03_kselection_diagnostics.py   # optional, diagnostic only
```

## Outputs

- `outputs/clusters/linkage_matrix.npy` — Ward linkage matrix.
- `outputs/clusters/grid_type_labels.csv` — one row per grid cell:
  `grid_id`, `city`, `country`, `label_k02` … `label_k30`.
- `outputs/kselection_metrics.csv` / `outputs/figures/kselection_curves.*`
  — internal quality metrics across candidate k (diagnostic only).

`grid_type_labels.csv` is the input to `../cluster_interpretation/` and to
every downstream analysis module in this repository.
