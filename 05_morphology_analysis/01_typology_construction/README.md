# Typology Construction

Builds a data-driven typology of African urban morphology from the
continent-scale embeddings, and characterises the resulting morphological
types.

## Contents

- [`typology_construction/`](typology_construction/) — hierarchical
  clustering of the embeddings into k = 17 morphological types and
  cell-level label assignment.
- [`k_selection/`](k_selection/) — sampling, rendering, and evaluation
  materials supporting the choice of k = 17.
- [`cluster_interpretation/`](cluster_interpretation/) — which
  morphometric indicators distinguish the 17 types, and each type's
  profile relative to the continent-wide average.
- [`embedding_visualization/`](embedding_visualization/) — 2D projections
  (t-SNE, UMAP) of the embedding space, coloured by cluster and by
  individual features.

## Pipeline order

```
k_selection/  ──centroids.npy──►  typology_construction/  ──grid_type_labels.csv──►  cluster_interpretation/
                                                                                              │
                                                                                              ▼
                                                                                   embedding_visualization/
```

`k_selection/` need not be re-run to reproduce the rest of the pipeline —
its output (`outputs/dendrogram/centroids.npy`) is a static input to
`typology_construction/`; see that module's README for what to substitute
if reproducing from scratch.

## Data requirements

Continent-scale student embeddings, produced by `../04_morphology_embedding/`,
and per-city morphometric indicator CSVs, produced by
`../02_morphology_metrics/`. See each submodule's README for the
specific paths to set.
