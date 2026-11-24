"""
01_build_city_graphs.py
=======================
For each city, build two representations:

1. Cell graph  G_c^cell
   Nodes  = grid cells (hex polygons)
   Edges  = queen spatial adjacency (shared boundary)
   Node attr: label_k17 (1-17)

2. 17-node type interaction graph  G_c^type
   Nodes  = morphological types 1-17
   Node weight p_i = share of city cells with that type
   Edge weight A_ij = log-enrichment of observed vs expected
                      adjacency between types i and j
                      A_ij = log((obs+eps) / (exp+eps))
   Only types present in the city appear as nodes.

Outputs (per city, in outputs/cache/):
   <country>_<city>_cell_graph.pkl      -- pickled networkx Graph
   <country>_<city>_type_graph.pkl      -- pickled networkx Graph

Example / demo run:  python 01_build_city_graphs.py --city Accra --country Ghana
Full run:            python 01_build_city_graphs.py --all
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from libpysal.weights import Queen

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
# SHP geometry + attributes per city, published by
# ../../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT   = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../02_morphology_metrics/outputs/SHP_cleaned",
))
LABELS_PATH = (ROOT.parent.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")
CACHE_DIR  = ROOT / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_TYPES = 17


# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_cell_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build queen-adjacency graph on hex grid cells.
    Node attribute: label_k17
    """
    w = Queen.from_dataframe(gdf, silence_warnings=True)
    G = nx.Graph()
    for idx, row in gdf.iterrows():
        G.add_node(row["GridID"], label=int(row["label_k17"]))
    for focal, neighbours in w.neighbors.items():
        grid_id_focal = gdf.iloc[focal]["GridID"]
        for nb in neighbours:
            grid_id_nb = gdf.iloc[nb]["GridID"]
            G.add_edge(grid_id_focal, grid_id_nb)
    return G


def build_type_graph(G_cell: nx.Graph) -> nx.Graph:
    """
    Compress cell graph into a 17-node type interaction graph.

    Node weight p_i = fraction of city cells with type i.
    Edge weight A_ij = log-enrichment:
        A_ij = log( (obs_ij + eps) / (exp_ij + eps) )
    where
        obs_ij = number of cell-edges between type i and type j
        exp_ij = n_i * n_j / (N*(N-1)/2)   [random expectation]
        n_i    = node count of type i
        N      = total nodes
    """
    labels = nx.get_node_attributes(G_cell, "label")
    label_arr = np.array(list(labels.values()))
    N = len(label_arr)

    # Node counts and shares
    types_present, counts = np.unique(label_arr, return_counts=True)
    p = {int(t): int(c) / N for t, c in zip(types_present, counts)}

    # Observed adjacency counts (symmetric, only upper triangle needed)
    obs = {}
    node_list = list(G_cell.nodes())
    label_map = {n: labels[n] for n in node_list}

    for u, v in G_cell.edges():
        ti, tj = label_map[u], label_map[v]
        key = (min(ti, tj), max(ti, tj))
        obs[key] = obs.get(key, 0) + 1

    total_edges = G_cell.number_of_edges()
    count_map = {int(t): int(c) for t, c in zip(types_present, counts)}
    # Total possible pairs (denominator for random expectation)
    total_pairs = N * (N - 1) / 2

    # Expected counts under random pairing:
    #   exp_ij = (n_i * n_j / total_pairs) * total_edges   for i != j
    #   exp_ii = (n_i*(n_i-1)/2 / total_pairs) * total_edges
    def expected(ti, tj):
        ni = count_map.get(ti, 0)
        nj = count_map.get(tj, 0)
        if ti == tj:
            pair_prob = ni * (ni - 1) / 2 / total_pairs
        else:
            pair_prob = ni * nj / total_pairs
        return pair_prob * total_edges

    # Build type graph
    G_type = nx.Graph()
    for t, share in p.items():
        G_type.add_node(t, p=share, count=int(share * N))

    for (ti, tj), obs_count in obs.items():
        exp_count = expected(ti, tj)
        if exp_count <= 0:
            continue
        enrichment = np.log(obs_count / exp_count)
        G_type.add_edge(ti, tj, weight=enrichment,
                        obs=obs_count, exp=exp_count)

    # Self-loops for same-type adjacency
    for ti in types_present:
        key = (int(ti), int(ti))
        if key in obs:
            exp_count = expected(int(ti), int(ti))
            if exp_count <= 0:
                continue
            enrichment = np.log(obs[key] / exp_count)
            G_type.add_edge(int(ti), int(ti), weight=enrichment,
                            obs=obs[key], exp=exp_count)

    return G_type


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_shp(country: str, city: str) -> Path | None:
    country_dir = SHP_ROOT / country / city
    if country_dir.exists():
        shps = list(country_dir.glob("*.shp"))
        if shps:
            return shps[0]
    return None


def cache_path(country: str, city: str, kind: str) -> Path:
    safe = f"{country}_{city}".replace(" ", "_")
    return CACHE_DIR / f"{safe}_{kind}_graph.pkl"


def save_graph(G: nx.Graph, path: Path):
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Process one city
# ---------------------------------------------------------------------------

def process_city(country: str, city: str,
                 labels: pd.DataFrame,
                 force: bool = False,
                 type_only: bool = False):
    cell_path = cache_path(country, city, "cell")
    type_path = cache_path(country, city, "type")

    cached = type_path.exists() and (type_only or cell_path.exists())
    if not force and cached:
        logging.debug("Cache hit: %s / %s", country, city)
        return "skip"

    shp = find_shp(country, city)
    if shp is None:
        logging.warning("SHP not found: %s / %s", country, city)
        return False

    try:
        gdf = gpd.read_file(shp)
    except Exception as e:
        logging.error("Failed to read SHP %s: %s", shp, e)
        return False

    # Join labels
    city_labels = labels[labels["city"] == city][["grid_id", "label_k17"]]
    if len(city_labels) == 0:
        logging.warning("No labels found: %s / %s", country, city)
        return False

    gdf = gdf.merge(city_labels, left_on="GridID", right_on="grid_id", how="inner")
    if len(gdf) < 10:
        logging.warning("Too few matched cells (%d): %s / %s", len(gdf), country, city)
        return False

    gdf = gdf.reset_index(drop=True)

    # Build graphs
    G_cell = build_cell_graph(gdf)
    G_type = build_type_graph(G_cell)

    if not type_only:
        save_graph(G_cell, cell_path)
    save_graph(G_type, type_path)

    logging.info("Built graphs for %s / %s  (cells=%d, edges=%d, types=%d)",
                 country, city, G_cell.number_of_nodes(),
                 G_cell.number_of_edges(), G_type.number_of_nodes())
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cell graph and 17-node type interaction graph per city.")
    parser.add_argument("--city",      default=None)
    parser.add_argument("--country",   default=None)
    parser.add_argument("--all",       action="store_true",
                        help="Process all cities in the labels table.")
    parser.add_argument("--force",     action="store_true",
                        help="Recompute even if cache exists.")
    parser.add_argument("--type-only", action="store_true",
                        help="Save only the type graph (skip large cell graph).")
    args = parser.parse_args()

    labels = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    if args.all:
        cities = labels[["country", "city"]].drop_duplicates()
        ok = fail = skip = 0
        for _, row in cities.iterrows():
            result = process_city(row["country"], row["city"], labels,
                                  args.force, type_only=args.type_only)
            if result is True:
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                fail += 1
        logging.info("Done. Success=%d  Skipped(cached)=%d  Failed=%d",
                     ok, skip, fail)
    elif args.city and args.country:
        process_city(args.country, args.city, labels,
                     args.force, type_only=args.type_only)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
