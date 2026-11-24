# Subgraph Construction

For each city, builds a spatial graph of its grid cells and a compact
17-node type-interaction graph summarising which morphological types are
spatially adjacent and how often. These per-city graphs are the shared
input to community detection.

## Method

1. `01_build_city_graphs.py` — for each city:
   - a cell graph: nodes are grid cells (queen adjacency edges), each
     labelled with its k = 17 morphological type;
   - a 17-node type graph: node weight is the city-wide share of cells of
     that type, edge weight is the log-enrichment of observed vs. expected
     adjacency between two types (only types actually present in the city
     appear as nodes).
2. `02_visualize_example.py` — illustrative rendering of an individual
   city's graphs (Fig. 2b), for figures and inspection.

## Data requirements

- Per-city cleaned SHP grid geometries (`SHP_ROOT`), produced by
  `../../../02_morphology_metrics/`.
- `grid_type_labels.csv` (`LABELS_PATH`), produced by
  `../../01_typology_construction/typology_construction/02_assign_labels.py`.

## Usage

```bash
# Single city (demo)
python 01_build_city_graphs.py --city Accra --country Ghana

# All cities
python 01_build_city_graphs.py --all
```

## Outputs

- `outputs/cache/<country>_<city>_cell_graph.pkl` /
  `_type_graph.pkl` — pickled `networkx.Graph` objects, one pair per city.
  Consumed by `../community_detection/01_build_city_features.py`.
