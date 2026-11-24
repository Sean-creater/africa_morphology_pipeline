# Africa Urban Morphology Metrics

Extracts building footprints and road networks for African cities and
computes building-form, street-network, connectivity, and
spatial-distribution morphology metrics, aggregated onto a 250 m
hexagonal grid. Covers the ~8,393 city/town agglomerations in the
AFRICAPOLIS inventory across 54 African countries.

This pipeline builds on `../01_hex_grid_workflow/`, which builds the
250 m hexagonal grid and attaches building height/volume, population,
and land cover statistics. This pipeline consumes that grid as an
input and extends it with building morphology,
street-network, connectivity, and spatial-distribution metrics derived
from Google Open Buildings and OpenStreetMap, before running a
data-cleaning/QC process to produce one standardized attribute table +
shapefile per city.

## Pipeline stages

1. **Open Buildings extraction** (`gcs_buildings.py`) — building footprint
   polygons from Google Open Buildings v3.
2. **OSM network extraction** (`osm_network.py`) — road network from
   Geofabrik, clipped per city.
3. **Morphology metrics** (`morphology_metrics.py`) — per-building
   dimension, shape, complexity, and 3D metrics via momepy, aggregated to
   grid cells.
4. **Street network metrics** (`streetnetwork_metrics.py`) — street
   density, width, orientation, and building-to-street-alignment metrics.
5. **Connectivity metrics** (`connectivity/`) — graph-theoretic street
   network metrics (centrality, meshedness, intersection typology, COINS
   street continuity).
6. **Spatial distribution metrics** (`spatial_distribution_metrics.py`) —
   building-to-building alignment, inter-building distance, and
   neighbor-count statistics.
7. **Aggregation to grid** — every metric family above is joined onto the
   companion grid.
8. **Data cleaning / QC** (`cleaning/`) — a script sequence that brings
   every city's output to a common column schema and produces the final
   SHP output. See `cleaning/README.md` for the run order.

The orchestrator (`pipeline.py`) drives stages 1-7 per city, with
checkpointing and coverage-gap tracking across large batch runs.

## Installation

```bash
conda env create -f environment.yml
conda activate africa_morphology_metrics
# or: pip install -r requirements.txt
pip install -e .          # optional, for the console entry point
pip install -e geoworkflow/  # the extraction backend
```

Python 3.10+ is required. `python-igraph` is optional but recommended.

### The `geoworkflow` package

`geoworkflow/` is a general-purpose geospatial extraction/processing
library, vendored here as a checked-in copy and used for the Open
Buildings extraction stage. It is not published to PyPI, so it is
installed from this checkout; it carries its own MIT license, distinct
from the rest of this pipeline (see License below). This project's own
scripts call its processor classes directly.

## Data source requirements

None of the following are bundled in this repository.

| # | Source | Used for |
|---|--------|----------|
| 1 | Google Open Buildings v3 | Building footprint polygons |
| 2 | OpenStreetMap, via Geofabrik | Road/highway network |
| 3 | Companion 250 m hex grid (`africa_hex_grid_workflow`) | Base grid |
| 4 | AFRICAPOLIS and country/administrative boundaries | City AOI definition |

See the manuscript's Methods and Supplementary Information for exact
access patterns and caching behavior.

## Output schema

Every metric below is joined onto the companion grid (which already
carries `GridID`, land cover, WSF3D height/volume, and population
columns). Full column-name mapping is in `data/reference/col_name_mapping.csv`.

- **Building morphology** (`morphology_metrics.py`): dimension, shape,
  complexity, 3D (height/volume-derived), and intensity metrics per
  building, aggregated to grid cells.
- **Street network** (`streetnetwork_metrics.py`): density, width,
  orientation, and building-to-street-alignment metrics.
- **Connectivity** (`connectivity/`): node and edge centrality, clustering,
  graph topology, intersection typology, and COINS street-continuity
  metrics.
- **Spatial distribution** (`spatial_distribution_metrics.py`):
  building-to-building alignment, inter-building distance, and
  neighbor-count statistics.

See the manuscript's Methods and Supplementary Information for the full
metric definitions and construction details, including the rationale for
three interchangeable connectivity backends (`connectivity/`), city-specific
UTM projections, and other implementation choices.

## Data cleaning / QC stage

See `africa_morphology_metrics/cleaning/README.md` for the script run
order. `data/reference/excluded_cities.md` documents cities excluded from
building/plot-morphology-dependent analyses for data-quality reasons.

## Data layout

```
africa_morphology_metrics/   # package code (this repo)
geoworkflow/                  # extraction backend (own MIT license)
data/
    reference/                 # small, checked-in reference/lookup data
    example/                   # real Ndioum input bundle and historical output
```

No bulk data is bundled. A full pipeline run reads/writes a large working
cache under `data/{Country}/{City}/` that is not part of this repository.

## Example output

`data/example/Ndioum_Senegal/` contains a real (not synthetic) input bundle
and reference output for one small city. See `data/example/README.md`.

## License

No license file was found in the source project this pipeline was derived
from. Until a license is chosen, all rights are reserved by default; add a
`LICENSE` file before distributing this code publicly. `geoworkflow/` is
the one exception: it carries its own MIT license (Copyright 2025 Cornell
University — see `geoworkflow/LICENSE`).
