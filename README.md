# Africa Urban Morphology Pipeline

End-to-end pipeline for the continent-scale analysis of African urban
morphology: grid construction, morphology metric extraction, image
dataset construction, embedding model training, and the downstream
typology/regional/hierarchy/center-periphery analysis.

## Pipeline stages

1. [`01_hex_grid_workflow/`](01_hex_grid_workflow/) — builds the
   standardized 250 m hexagonal grid over African cities, with
   building height/volume, population, and land cover statistics.
2. [`02_morphology_metrics/`](02_morphology_metrics/) — extracts
   building footprints and road networks, and computes building-form,
   street-network, connectivity, and spatial-distribution morphology
   metrics onto that grid.
3. [`03_morphology_imagery/`](03_morphology_imagery/) — constructs the
   satellite/building-overlay image dataset and pairwise comparison
   set used to train and evaluate the embedding model.
4. [`04_morphology_embedding/`](04_morphology_embedding/) — trains the
   teacher-student embedding model (ablations M1-M6) and produces the
   continent-scale embedding table.
5. [`05_morphology_analysis/`](05_morphology_analysis/) — the
   downstream analysis: typology construction, regional structure,
   urban hierarchy, and center-periphery organisation, corresponding
   to the manuscript's main figures and supplementary analyses.

Each module is self-contained with its own environment, README, and
data-requirements section. See each module's README for its pipeline
order, usage, and outputs.

## Cross-module data

Modules read from each other's outputs using relative paths with
environment-variable overrides (e.g. `AFRICA_SHP_CLEANED_ROOT`,
`AFRICA_EMBEDDINGS_PATH`), assuming this repository's standard layout.
The one exception is the continent-scale embedding table
(`africa_embeddings.npz`), which is not produced by any script and
must be downloaded from `04_morphology_embedding/`'s release assets.

See the manuscript's Methods and Supplementary Information for the
full analysis and construction details.
