# Center-Periphery

Analyses the radial organisation of urban morphology from each city's
population centre to its edge, and how that radial transition differs
across cities of different size.

## Contents

- [`core/`](core/) — city centre definitions and the population-scaled
  radial coordinate, the shared input to `extended/`.
- [`extended/`](extended/) — the radial transition stratified by city-size
  tier, connecting it to regional context and testing its robustness.

## Pipeline order

```
core/  ──city centres + radial coordinates──►  extended/
```

`extended/` also draws on `../03_urban_hierarchy/` (population-size tiers)
and `../02_regional_structure/regression/`'s merged covariate table
(climate/colonial/GDP context).

## Data requirements

Per-city cleaned SHP grid geometries and `grid_type_labels.csv`, produced
by `../02_morphology_metrics/` and `../01_typology_construction/`
respectively. See each submodule's README for the specific paths to set.
