# City/Country Covariate Table

Merges city- and country-level covariates with Leiden community labels
into one row per city, for use by downstream analyses
(`../../03_urban_hierarchy/`, `../../04_center_periphery/core/`).

## Method

`00_prepare_data.py` merges city- and country-level covariates
(`input/city.xlsx`, `input/Country.xlsx`) with the Leiden community
labels from `../community_detection/`, into one row per city. No
category merging or recoding beyond what each input column already
represents.

## Data requirements

- `input/city.xlsx` / `input/Country.xlsx` — city- and country-level
  covariates (coordinates, population, GDP, climate zone, colonial
  history), bundled in this repository.
- `../community_detection/outputs/city_communities.parquet` — community
  labels, produced by `../community_detection/02_leiden_communities.py`.

## Usage

```bash
python 00_prepare_data.py
```

## Outputs

- `outputs/tables/analysis_input.csv` — merged city-level table (one row
  per city), consumed by `../../03_urban_hierarchy/01_compute_city_metrics.py`
  and `../../04_center_periphery/core/01_define_centres.py`.
- `outputs/tables/covariate_summary.csv` — coverage and cross-tabulation
  checks on the merged covariates.
