"""
01_build_city_features.py
=========================
Extract fixed-length feature vectors from per-city 17-node type
interaction graphs and save them for downstream community detection.

City representation
-------------------
Each city c is described by two vectors:

  p_c (17-dim)  — composition
      p_ci = n_ci / N_c  (share of cells of type i)

  A_c (≤153-dim, sparse) — spatial configuration
      A_cij = log(O_ij / E_ij)   only when O_ij > 0 and E_ij > 0
      undefined (NaN) otherwise

      O_ij : observed adjacency count between types i and j
      E_ij : expected count under random label permutation
               i ≠ j :  E_ij = (n_i * n_j / C(N,2)) * E
               i = j :  E_ii = (C(n_i,2)  / C(N,2)) * E
             where E = total edges in the cell graph

  A_ij > 0 : enriched adjacency   (more frequent than random)
  A_ij < 0 : depleted adjacency   (less frequent than random)
  A_ij = NaN: undefined           (O_ij = 0 or type absent)

Pairs where O_ij = 0 (both types present but never adjacent) are
treated as undefined — not as strong avoidance — to avoid
pseudocount arbitrariness.  Downstream distance computation uses
only the intersection of defined pairs across city pairs.

Country / city name authority
------------------------------
Names are resolved from the SHP_cleaned folder hierarchy
  <SHP_ROOT>/<Country>/<City>/
which is the single source of truth used throughout the pipeline.
Cache filenames (<Country>_<City>_type_graph.pkl, spaces→underscores)
are matched back to the authoritative folder names via a reverse lookup
table built at startup.  Unresolved files are logged and skipped.

Output
------
outputs/city_features.parquet
    Columns: country, city, n_cells, n_types, n_edges,
             p_1 … p_17,
             A_1_1  A_1_2 … A_17_17   (upper-triangle, NaN where undefined)

Usage
-----
    python 01_build_city_features.py
"""

import logging
import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT.parent / "subgraph_construction" / "outputs" / "cache"
# SHP geometry + attributes per city, published by
# ../../../02_morphology_metrics/ (cleaning stage output, "SHP_cleaned").
# Override with AFRICA_SHP_CLEANED_ROOT if your copy lives elsewhere.
SHP_ROOT  = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../02_morphology_metrics/outputs/SHP_cleaned",
))
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TYPES = 17
PAIRS   = [(i, j) for i in range(1, N_TYPES + 1)
                  for j in range(i, N_TYPES + 1)]   # 153 upper-triangle pairs
P_COLS  = [f"p_{i}"   for i in range(1, N_TYPES + 1)]
A_COLS  = [f"A_{i}_{j}" for i, j in PAIRS]


# ---------------------------------------------------------------------------
# Name resolution: SHP_cleaned folder → cache filename lookup
# ---------------------------------------------------------------------------

def build_name_lookup(shp_root: Path) -> dict[str, tuple[str, str]]:
    """
    Walk SHP_cleaned/<Country>/<City>/ and build a dict:
        stem_key  →  (country, city)
    where stem_key = f"{country}_{city}".replace(" ", "_")

    This allows exact reverse-lookup from cache filenames to authoritative
    country/city names, handling all multi-word names correctly.

    Collisions (same stem from different countries) are logged as warnings;
    the first entry wins (alphabetical order of countries).
    """
    lookup: dict[str, tuple[str, str]] = {}
    n_collision = 0
    for country_dir in sorted(shp_root.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        for city_dir in sorted(country_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city = city_dir.name
            key  = f"{country}_{city}".replace(" ", "_")
            if key in lookup:
                logging.warning(
                    "Stem collision: '%s' → (%s, %s) already mapped to (%s, %s); "
                    "keeping first entry.",
                    key, country, city, *lookup[key])
                n_collision += 1
            else:
                lookup[key] = (country, city)
    logging.info("Name lookup built: %d entries, %d collisions.",
                 len(lookup), n_collision)
    return lookup


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(G: nx.Graph) -> dict:
    """
    Extract p and A vectors from a 17-node type interaction graph.

    Returns a dict with keys p_1…p_17 and A_i_j for each pair.
    A_i_j is NaN when the pair is undefined (O_ij = 0 or type absent).
    """
    p_attr = nx.get_node_attributes(G, "p")
    w_attr = nx.get_edge_attributes(G, "weight")

    p_vec = {f"p_{i}": p_attr.get(i, 0.0) for i in range(1, N_TYPES + 1)}

    present = set(G.nodes())   # types that exist in this city

    a_vec = {}
    for i, j in PAIRS:
        if i not in present or j not in present:
            a_vec[f"A_{i}_{j}"] = np.nan   # type absent
            continue
        w = w_attr.get((i, j), w_attr.get((j, i), None))
        if w is None:
            # Both types present but O_ij = 0 → undefined
            a_vec[f"A_{i}_{j}"] = np.nan
        else:
            a_vec[f"A_{i}_{j}"] = float(w)

    return {**p_vec, **a_vec}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pkl_files = sorted(CACHE_DIR.glob("*_type_graph.pkl"))
    logging.info("Found %d type graph cache files.", len(pkl_files))

    # Build reverse-lookup from SHP_cleaned (authoritative names)
    name_lookup = build_name_lookup(SHP_ROOT)

    records     = []
    n_unresolved = 0

    for path in pkl_files:
        stem = path.stem.replace("_type_graph", "")

        resolved = name_lookup.get(stem)
        if resolved is None:
            logging.warning("Unresolved cache file (not in SHP_cleaned): %s",
                            path.name)
            n_unresolved += 1
            continue
        country, city = resolved

        try:
            with open(path, "rb") as f:
                G = pickle.load(f)
        except Exception as e:
            logging.warning("Failed to load %s: %s", path.name, e)
            continue

        counts  = nx.get_node_attributes(G, "count")
        n_cells = sum(counts.values())
        n_types = G.number_of_nodes()
        n_edges = sum(nx.get_edge_attributes(G, "obs").values())

        feats = extract_features(G)
        records.append({
            "country": country,
            "city":    city,
            "n_cells": n_cells,
            "n_types": n_types,
            "n_edges": int(n_edges),
            **feats,
        })

    if n_unresolved:
        logging.warning("%d cache files could not be resolved and were skipped.",
                        n_unresolved)

    df = pd.DataFrame(records)
    cols = ["country", "city", "n_cells", "n_types", "n_edges"] + P_COLS + A_COLS
    df = df[cols]

    out = OUT_DIR / "city_features.parquet"
    df.to_parquet(out, index=False)
    logging.info("Saved %d cities → %s", len(df), out)

    # Diagnostic: defined A pairs per city
    a_defined = df[A_COLS].notna().sum(axis=1)
    logging.info("Defined A pairs per city: median=%d  min=%d  max=%d",
                 int(a_defined.median()), int(a_defined.min()), int(a_defined.max()))
    logging.info("n_cells: median=%d  min=%d  max=%d",
                 int(df["n_cells"].median()), int(df["n_cells"].min()),
                 int(df["n_cells"].max()))
    logging.info("Countries: %d  Cities: %d",
                 df["country"].nunique(), len(df))


if __name__ == "__main__":
    main()
