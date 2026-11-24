# Morphological Features

Tabular input features for the student model and the indicator-based
ablation alternatives.

| File | Description |
|------|-------------|
| `features_analysis_set.csv` | Morphological indicators for the analysis-set grid cells |
| `features_val_set.csv` | Morphological indicators for the validation-pair grid cells |
| `col_name_mapping.csv` | Full name mapping for truncated column names |

One row per grid cell, identified by `GridID`. Feature columns cover land
cover, building height and volume, population, street-network topology, and
building/plot morphology indicators, matching
`../../../02_morphology_metrics/`'s output schema.

Covers 46 African cities across 20+ countries. See the manuscript's Methods
and Supplementary Information for missing-value handling.
