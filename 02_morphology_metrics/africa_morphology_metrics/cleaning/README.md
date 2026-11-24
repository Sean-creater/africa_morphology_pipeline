# Cleaning stage

The final QC/data-cleaning stage of the pipeline. Takes per-city hexagonal-
grid metric tables produced by the upstream grid-enrichment stage and turns
them into a verified, uniform-schema cleaned dataset, then rejoins the
cleaned attribute tables onto grid geometry to produce `SHP_cleaned/`.

Not every city's raw output reaches the full column schema by the same
route (some are missing height-derived metrics, road-network metrics, or
both). The scripts in this directory bring every city up to the common
schema and verify the result. All scripts take explicit CLI arguments for
input/output directories.

The scripts run in a fixed dependency order: a per-city quality inventory
first, then country-specific corrections, then column-completion steps
(height-derived metrics, then road-network/connectivity metrics), each
followed by a QC gate, ending with the final geometry join. See the
manuscript's Methods and Supplementary Information for the full run order,
each step's purpose, and the specific data-quality issues addressed.

## Reference/data files

Small, checked-in reference data used by these scripts lives in
`data/reference/` (relative to the repository root): a CSV-to-Shapefile
column-name mapping, a per-city tracking table, the list of cities excluded
for data-quality reasons, and the study-area boundary.
