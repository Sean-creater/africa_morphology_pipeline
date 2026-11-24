# Regional Structure

Builds a continent-level picture of how the k = 17 morphological types
combine at the city scale, and groups cities into communities of similar
composition and spatial configuration.

## Contents

- [`subgraph_construction/`](subgraph_construction/) — per-city spatial
  graphs and compact type-interaction summaries.
- [`community_detection/`](community_detection/) — Leiden communities of
  cities, from a similarity graph over the per-city summaries.
- [`regression/`](regression/) — merges city- and country-level
  covariates (coordinates, population, GDP, climate zone, colonial
  history) with the community labels into the per-city covariate table
  used by `../03_urban_hierarchy/` and `../04_center_periphery/core/`.

## Pipeline order

```
subgraph_construction/  ──type graphs──►  community_detection/  ──community labels──►  regression/
```

## Data requirements

Per-city cleaned SHP grid geometries and `grid_type_labels.csv`, produced
by `../02_morphology_metrics/` and `../01_typology_construction/`
respectively. See each submodule's README for the specific paths to set.
