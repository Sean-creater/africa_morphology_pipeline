# Urban Hierarchy

Relates city-level morphological composition and spatial organisation to
city population, testing whether cities occupy a common position on a
continent-wide size gradient or whether that gradient differs by
geographic community or country.

## Method

1. `01_compute_city_metrics.py` — for each of ~6,918 cities, computes:
   - `p1`–`p4`: share of grid cells in each of four higher-order
     morphology groups (a coarser grouping of the k = 17 types, see
     `GROUP_MAP` in the script);
   - `H_star` / `H17_star`: normalised Shannon entropy of the group/type
     composition (0 = one group dominates, 1 = all groups equal);
   - `r` / `r17`: Newman categorical spatial assortativity of the
     group/type composition, computed from Queen adjacency built directly
     from grid geometry;
   - `log_pop`, `community`, and `country`, joined from
     `../02_regional_structure/`.
2. `02_plot_hierarchy.py` — the continent-wide trend panel (Fig. 3a, top):
   all four group shares vs. log(population), LOESS smooth with
   bootstrap CI.
3. `03_country_slopes.py` — country-specific OLS slopes for the same
   outcomes, to check how consistent the continental pattern is across
   countries.
4. `04_group_scatter.py` — the same relationships disaggregated by
   community (Fig. 3a's lower panels), to check how consistent the
   pattern is across the communities identified in
   `../02_regional_structure/community_detection/`.
5. `06_scaling_regression.py` — standardised OLS scaling regressions for
   `H17_star` and `r17` against log(population), with and without
   community fixed effects.
6. `07_size_gradient.py` — the main-text scaling figure for `H17_star` and
   `r17`.
7. `08_robustness.py` — robustness checks for the scaling regressions:
   community fixed effects, country fixed effects, a minimum-cell-count
   filter, and the 4-group outcome versions.
8. `09a_group_share_breakpoint.py` — the population range where each
   4-group composition share (p1-p4) shifts fastest, used to annotate
   the breakpoint in `02_plot_hierarchy.py`'s figure.

## Data requirements

- Per-city cleaned SHP grid geometries (`SHP_ROOT` in
  `01_compute_city_metrics.py`), produced by `../02_morphology_metrics/`.
- `grid_type_labels.csv`, produced by
  `../01_typology_construction/typology_construction/02_assign_labels.py`.
- `city_communities.parquet` and `analysis_input.csv` (for community and
  population covariates), produced by
  `../02_regional_structure/community_detection/` and
  `../02_regional_structure/regression/` respectively.

## Usage

```bash
python 01_compute_city_metrics.py
python 02_plot_hierarchy.py
python 03_country_slopes.py
python 04_group_scatter.py
python 06_scaling_regression.py
python 07_size_gradient.py
python 08_robustness.py
python 09a_group_share_breakpoint.py
```

## Outputs

- `outputs/tables/city_metrics.parquet` — one row per city, the shared
  input to steps 2–8.
- `outputs/tables/scaling_regression.csv` / `robustness.csv` /
  `country_slopes.csv` — regression coefficient tables.
- `outputs/figures/` — all figures from steps 2–7.
