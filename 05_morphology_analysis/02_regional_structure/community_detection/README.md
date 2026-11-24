# Community Detection

Groups cities into communities of similar morphological composition and
spatial configuration, using the per-city type-interaction graphs from
`../subgraph_construction/`.

## Method

1. `01_build_city_features.py` — reduces each city's type graph to a fixed
   two-part feature vector: composition (`p`, 17-dim, share of cells per
   type) and spatial configuration (`A`, up to 153-dim, log-enrichment of
   observed vs. expected adjacency between type pairs).
2. `02_leiden_communities.py` — builds a continent-level city similarity
   graph (composition distance + configuration distance, weighted by α),
   then detects communities with the Leiden algorithm on a k-NN graph.
   Distance matrices are cached; parameter sweeps over α and k reuse the
   cache. The main partition (α = 0.5, k = 100) is used throughout;
   sensitivity to both parameters is reported alongside it.
3. `03_visualize_communities.py` — continent-wide map of community
   membership (Fig. 2a).
4. `05_community_profiles.py` — per-community type composition and
   type-adjacency heatmaps (Fig. 2c,d).

## Robustness checks

- `check_knn_graph_row_order_invariance.py` — confirms the k-NN graph
  construction in `02_leiden_communities.py` is invariant to the input
  city table's row order. Run once; not part of the main pipeline.

## Data requirements

- Per-city type graphs (`CACHE_DIR`), produced by
  `../subgraph_construction/01_build_city_graphs.py`.
- City coordinates (`../regression/input/city.xlsx`).
- An Africa country boundary shapefile (`BOUNDARY` in
  `03_visualize_communities.py`), for the map figure only.

## Usage

```bash
python 01_build_city_features.py
python 02_leiden_communities.py
python 03_visualize_communities.py
python 05_community_profiles.py
```

## Outputs

- `outputs/city_communities.parquet` — main partition (α = 0.5, k = 100):
  one row per city, with its community assignment. Consumed by
  `../regression/00_prepare_data.py` as
  `../community_detection/outputs/city_communities.parquet`.
- `outputs/sensitivity/city_communities_alpha{a}_k{k}.parquet` —
  partitions across the α × k sensitivity sweep.
- `outputs/figures/community_map.*` — Fig. 2a.
