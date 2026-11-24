# Center-Periphery (Extended): Morphological Transition

Extends the center-periphery radial framework in `../core/` with city-size
stratification, asking how the radial transition in urban form differs
across small, medium, large, and metropolitan cities, and how much of the
cross-city variation it captures is attributable to national/regional
context versus population-driven spatial organisation alone.

## Canonical pipeline

```
core/01_define_centres.py -> core/02_cell_radial_position.py -> core/00_allometric_scaling.py
extended/00_define_size_tiers.py -> extended/01_radial_coordinates.py
extended/13_embedding_radial_join.py
extended/20_trajectory_by_africapolis_tier.py -> extended/25_trajectory_by_tier_combined.py
extended/30_tier_radial_composition_divergence.py (composition explanation for the tier
                                              trajectory, parallel to community's composition_by_
                                              community.svg -- JS-distance curves + country-
                                              block bootstrap CI, 17-type and 4-group
                                              resolutions; reuses 20's composition tables
                                              directly, no new metric. Imports
                                              20_trajectory_by_africapolis_tier.py by path for
                                              its aggregation functions.)
extended/33_tier_coverage_divergence.py (for each Africapolis size tier SEPARATELY, full
                                              pairwise cross-city embedding divergence
                                              (V_between) at each city's own built-up coverage
                                              level -- "at X% coverage, how unpredictable are
                                              same-tier cities from each other". Two figures,
                                              core and periphery, reusing 23's qualifying
                                              636-city sample and core/outer embeddings. Finding:
                                              at the CORE, Intermediate and Large tiers converge
                                              with rising coverage (less cross-city divergence at
                                              high coverage) while Small tier DIVERGES (more
                                              divergence at high coverage) -- opposite direction,
                                              not just a different slope. At the PERIPHERY, all
                                              three tiers diverge with rising coverage (same
                                              direction), so "higher coverage -> more predictable
                                              form" only holds for mid/large cities' cores, not
                                              small cities' cores and not any tier's periphery.
                                              No population density/Pop2022 used.)
extended/21_trajectory_by_community.py (single-panel combined figure, self-contained)
extended/23_scale_dependent_convergence.py  (between-city divergence, reversal P*~126k)
extended/29_core_outer_divergence_mechanism.py (mechanism decomposition --
                                              core vs periphery levels separately;
                                              reuses 23's dispersion table, no new metric)
extended/14_coverage_vs_morphology.py       (built-up coverage vs. morphology: out-of-sample R^2 and
                                              density-ratio result, see caveat below)
extended/17_figure_main_embedding.py        (main 4-panel figure; panel b sources from 14 above)
extended/11_accra_rings_viz.py              (illustrative single-city ring visualisation,
                                              embedded as 17's panel a)
extended/34_fig_s10_sensitivity.py          (Supplementary Fig. S10: sensitivity of the Fig. 4c/4d
                                              findings to alternative center definitions,
                                              neighborhood sizes, and core-periphery boundaries)
```

**Caveats:**
- `14_coverage_vs_morphology.py` predicts WorldPop-constrained population
  (or log-population) from built coverage + embedding. WorldPop's own
  constrained-disaggregation methodology uses building footprints / GHSL
  built-settlement masks / roads as covariates, so this risks re-discovering
  WorldPop's modelling inputs rather than finding independent morphology
  evidence; its R²/density-ratio numbers (feeding `17_figure_main_embedding.py`
  panel b) should be read as descriptive, not as a causal morphology effect.

**Exploratory / deprecated -- do not treat as the main-text source:**
- `19_trajectory_curve_fit_stratified.py` -- T1-T4-tier curve fits,
  superseded by `20_trajectory_by_africapolis_tier.py`'s Africapolis tiers.
- `22_community_vs_population_confound.py` -- a confound check for
  `21_trajectory_by_community.py`; informative but not a main-text figure.
- `24_continuous_q_dispersion_heatmap.py` -- a continuous-q visualisation
  of the SAME pattern `23_scale_dependent_convergence.py` reports formally;
  23 remains the formal result with the country-block bootstrap CI.
- `26_radial_coverage_vs_morphology.py` -- methods follow-up to `14`,
  stopped mid-run with no saved output.

## Method

The T1–T4 fixed population-size tiers computed in `00_define_size_tiers.py`
are a legacy framing, superseded by the continuous-population embedding
framework (`13`–`33` below) as the basis for population-driven claims in
the manuscript. `00_define_size_tiers.py` and `01_radial_coordinates.py`
are retained as data-preparation steps: `city_tiers.parquet` (log
population, R98) and `radial_cells.parquet` (per-cell radial position)
are read directly by `13_embedding_radial_join.py` and
`11_accra_rings_viz.py` respectively. A parallel set of scripts applying
the T1-T4 tiers to type-composition-level outcomes (17-type radial
profiles, diversity, assortativity, divergence, and their regional/
robustness follow-ups) was not carried into the manuscript or
Supplementary Information and has been removed.

1. `00_define_size_tiers.py` — four fixed population-size tiers (T1–T4,
   inherited from `../../03_urban_hierarchy/`'s scaling breakpoints), plus
   a supplementary two-breakpoint segmented regression checking that
   `H17_star`/`r17` also show non-linear change near those boundaries.
2. `01_radial_coordinates.py` — 10 equal-width radial bins (core = bin 0,
   periphery = bin 9) over the common-support range of the population-
   scaled distance from `../core/`, applied consistently across all
   distance coordinates.
3. `11_accra_rings_viz.py` — illustrative single-city (Accra) ring
   visualisation of the radial coordinate, embedded as a panel in
   `17_figure_main_embedding.py`.

See the canonical-pipeline list above for `13`–`33`.

## Data requirements

- `city_centres.parquet`, `cell_radial_position.parquet`, `city_R98.parquet`,
  `allometric_params.json`, produced by `../core/`.
- `city_metrics.parquet`, produced by `../../03_urban_hierarchy/`.
- `analysis_input.csv`, produced by `../../02_regional_structure/regression/`.

## Python environment

Run everything inside the repository's pinned conda environment, not a
bare `python3` / `python` off the shell's default `$PATH` -- a system or
other default Python build (e.g. 3.12/3.13) can fail to read this
project's parquet files (`city_centres.parquet` and others raise a
"Repetition level histogram size mismatch" pyarrow error under an
unpinned interpreter):

```bash
conda env create -f ../../environment.yml
conda activate africa_morphology_analysis
python <script>.py
```

## Usage

```bash
python 00_define_size_tiers.py
python 01_radial_coordinates.py
python 13_embedding_radial_join.py
python 14_coverage_vs_morphology.py
python 20_trajectory_by_africapolis_tier.py
python 21_trajectory_by_community.py
python 23_scale_dependent_convergence.py
python 25_trajectory_by_tier_combined.py
python 29_core_outer_divergence_mechanism.py
python 30_tier_radial_composition_divergence.py
python 33_tier_coverage_divergence.py
python 11_accra_rings_viz.py
python 17_figure_main_embedding.py
python 34_fig_s10_sensitivity.py
```

## Outputs

- `outputs/tables/city_tiers.parquet` / `radial_cells.parquet` — the
  shared tier and radial-bin assignments, consumed by `13` and `11`.
- `outputs/tables/cell_embeddings_index.parquet` /
  `outputs/arrays/cell_embeddings.npy` — the embedding-joined cell table
  from `13`, the shared input to `14`–`33`.
- `outputs/figures/figure_main_embedding.*` — the main 4-panel figure
  (`17_figure_main_embedding.py`).
- `outputs/figures/fig_s10_sensitivity.*` — Supplementary Fig. S10
  (`34_fig_s10_sensitivity.py`).
