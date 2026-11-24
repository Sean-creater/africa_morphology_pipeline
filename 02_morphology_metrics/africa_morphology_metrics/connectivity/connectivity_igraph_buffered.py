"""
Street network connectivity metrics - igraph-accelerated backend.

Design note - hybrid strategy:
Centrality calculations (betweenness, closeness, straightness, edge
betweenness) are the computational bottleneck of connectivity analysis and
are delegated to igraph, whose C implementation is substantially faster
than NetworkX's pure-Python algorithms for these operations. Simple
statistics (node degree, cyclomatic complexity, edge/node ratio, gamma
index, meshedness, node density, cul-de-sac length) remain on NetworkX /
momepy, since they are already O(V+E) and not worth the conversion
overhead. Subgraph caching (build once per cell, reuse for every metric)
is inherited from the NetworkX backend and extended to cache both the
NetworkX and igraph representations together.

Design note - buffer:
Like the NetworkX backend, each cell's subgraph includes nodes from
directly adjacent grid cells (a 6-neighbor buffer), giving shortest-path
metrics network context beyond the cell boundary. The no-buffer backend
in this package removes this; see its docstring for why.
"""

import logging
import pickle
import numpy as np
import networkx as nx
import igraph as ig
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

from .igraph_utils import (
    nx_to_igraph,
    get_target_vertices,
    extract_vertex_values
)

from .connectivity_networkx import (
    get_buffer_indices_fast,
    extract_node_ids_vectorized
)

logger = logging.getLogger(__name__)


# ============================================================================
# CORE CENTRALITY FUNCTIONS (igraph-powered)
# ============================================================================

def calculate_betweenness_centrality_ig(
    g: ig.Graph,
    target_vertices: List[int]
) -> Dict[int, float]:
    """
    Betweenness centrality via igraph: c_B(v) = sum_st sigma(s,t|v) / sigma(s,t)
    (fraction of shortest paths passing through node v).

    Parameters
    ----------
    g : ig.Graph
        igraph graph (subgraph with cell + neighbors).
    target_vertices : list of int
        Vertex indices for this cell only.

    Returns
    -------
    dict
        {vertex_idx: betweenness_value}
    """
    betweenness_values = g.betweenness(weights='weight', directed=False)
    result = {v_idx: betweenness_values[v_idx] for v_idx in target_vertices}
    return result


def calculate_closeness_centrality_ig(
    g: ig.Graph,
    target_vertices: List[int]
) -> Dict[int, float]:
    """
    Closeness centrality via igraph: c_C(v) = (n-1) / sum_u d(v,u)
    (inverse of average distance to all other nodes).

    Parameters
    ----------
    g : ig.Graph
        igraph graph.
    target_vertices : list of int
        Vertex indices for this cell only.

    Returns
    -------
    dict
        {vertex_idx: closeness_value}
    """
    # normalized=True matches momepy's default
    closeness_values = g.closeness(weights='weight', normalized=True)
    result = {v_idx: closeness_values[v_idx] for v_idx in target_vertices}
    return result


def calculate_straightness_centrality_ig(
    g: ig.Graph,
    target_vertices: List[int]
) -> Dict[int, float]:
    """
    Straightness centrality: c_S(v) = sum_u d_euclidean(v,u) / d_graph(v,u) / (n-1)
    (how "straight" the shortest paths from v are).

    Not a built-in igraph metric; computed here using igraph's fast
    weighted shortest-path distances as the graph-distance term.

    Parameters
    ----------
    g : ig.Graph
        igraph graph with 'x', 'y' vertex attributes.
    target_vertices : list of int
        Vertex indices for this cell only.

    Returns
    -------
    dict
        {vertex_idx: straightness_value}
    """
    result = {}
    n = g.vcount()

    x_coords = g.vs['x']
    y_coords = g.vs['y']

    for v_idx in target_vertices:
        # mode='out' is used for undirected graphs; weights='weight' applies edge weights
        distances = g.shortest_paths(
            source=v_idx,
            weights='weight',
            mode='out'
        )[0]  # Returns list of lists, take first row

        total = 0.0
        count = 0

        for u_idx in range(n):
            if u_idx == v_idx:
                continue

            dx = x_coords[v_idx] - x_coords[u_idx]
            dy = y_coords[v_idx] - y_coords[u_idx]
            d_euclidean = math.sqrt(dx*dx + dy*dy)

            d_graph = distances[u_idx]

            # Skip if unreachable (infinite distance)
            if d_graph > 0 and not math.isinf(d_graph):
                total += d_euclidean / d_graph
                count += 1

        result[v_idx] = total / count if count > 0 else 0.0

    return result


# ============================================================================
# SUBGRAPH CACHE BUILDING (NetworkX + igraph)
# ============================================================================

def build_subgraph_cache(
    grid: gpd.GeoDataFrame,
    adjacency: dict,
    nodes_gdf: gpd.GeoDataFrame,
    graph: nx.MultiGraph,
    cell_to_nodes: dict,
    grid_id_col: str = 'GridID',
    cache_file: Optional[Path] = None
) -> dict:
    """
    Build the subgraph cache, storing both NetworkX and igraph
    representations of each cell's subgraph.

    Cache structure:
    {
        cell_id: {
            'nx_subgraph': NetworkX MultiGraph (for simple stats),
            'ig_graph': igraph Graph (for centrality),
            'ig_mappings': {...},
            'target_node_ids': [...],  # NetworkX node IDs
            'target_vertices': [...]   # igraph vertex indices
        }
    }

    Parameters
    ----------
    grid : GeoDataFrame
        Grid cells.
    adjacency : dict
        Grid cell adjacency.
    nodes_gdf : GeoDataFrame
        Street network nodes.
    graph : nx.MultiGraph
        Street network graph.
    cell_to_nodes : dict
        Mapping from grid cell ID to node indices.
    grid_id_col : str
        Grid ID column name.
    cache_file : Path, optional
        Path to cache file for persistence.

    Returns
    -------
    dict
        Subgraph cache.
    """
    if cache_file and cache_file.exists():
        logger.info(f"  Loading cached subgraphs from {cache_file.name}...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"  Failed to load cache: {e}")
            logger.warning("  Rebuilding cache...")

    logger.info("  Building subgraph cache (NetworkX + igraph)...")
    logger.info(f"  Grid cells: {len(grid)}, Adjacency entries: {len(adjacency)}, cell_to_nodes entries: {len(cell_to_nodes)}")
    subgraph_cache = {}
    empty_count = 0
    conversion_fail_count = 0

    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="  Building subgraphs"):
        cell_id = row[grid_id_col]
        neighbors = adjacency.get(cell_id, [])
        buffer_cell_ids = [cell_id] + list(neighbors)

        # 1. Build NetworkX subgraph
        buffer_node_indices = get_buffer_indices_fast(buffer_cell_ids, cell_to_nodes)
        buffer_nodes = nodes_gdf.iloc[list(buffer_node_indices)]
        node_ids = extract_node_ids_vectorized(buffer_nodes)
        nx_subgraph = graph.subgraph(node_ids).copy()

        # Check if empty first
        if len(nx_subgraph.nodes) == 0:
            subgraph_cache[cell_id] = None
            empty_count += 1
            continue

        # Ensure connected (take largest component)
        if not nx.is_connected(nx_subgraph):
            largest_cc = max(nx.connected_components(nx_subgraph), key=len)
            nx_subgraph = nx_subgraph.subgraph(largest_cc).copy()

        # 2. Convert to igraph
        try:
            ig_graph, ig_mappings = nx_to_igraph(nx_subgraph)
        except Exception as e:
            logger.debug(f"  Cell {cell_id}: igraph conversion failed: {e}")
            subgraph_cache[cell_id] = None
            conversion_fail_count += 1
            continue

        # 3. Identify target nodes (only this cell)
        target_node_indices = cell_to_nodes.get(cell_id, [])
        target_nodes = nodes_gdf.iloc[target_node_indices]
        target_node_ids = extract_node_ids_vectorized(target_nodes)
        target_node_ids = [nid for nid in target_node_ids if nid in nx_subgraph.nodes()]

        # Map to igraph vertex indices
        target_vertices = get_target_vertices(target_node_ids, ig_mappings['node_to_vertex'])

        # 4. Store both NetworkX and igraph
        subgraph_cache[cell_id] = {
            'nx_subgraph': nx_subgraph,
            'ig_graph': ig_graph,
            'ig_mappings': ig_mappings,
            'target_node_ids': target_node_ids,
            'target_vertices': target_vertices
        }

    logger.info(f"  Empty subgraphs (no nodes): {empty_count}")
    logger.info(f"  Conversion failures: {conversion_fail_count}")

    if cache_file:
        logger.info(f"  Saving cache to {cache_file.name}...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(subgraph_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"  Cache saved: {cache_size_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"  Failed to save cache: {e}")

    return subgraph_cache


# ============================================================================
# UNIFIED METRIC CALCULATION (hybrid NetworkX + igraph)
# ============================================================================

# Global worker variables
_worker_nodes_gdf = None
_worker_subgraph_cache = None
_worker_grid_id_col = None


def _init_worker(nodes_gdf, subgraph_cache, grid_id_col):
    """Initialize worker process with shared data."""
    global _worker_nodes_gdf, _worker_subgraph_cache, _worker_grid_id_col
    _worker_nodes_gdf = nodes_gdf
    _worker_subgraph_cache = subgraph_cache
    _worker_grid_id_col = grid_id_col


def _calculate_all_metrics_batch(batch: List[Tuple]) -> List[Optional[Dict]]:
    """
    Calculate all metrics for a batch of cells.

    HYBRID STRATEGY:
    - Centrality (bottleneck) -> igraph
    - Simple stats -> NetworkX

    Parameters
    ----------
    batch : list of tuples
        List of (cell_id,) tuples.

    Returns
    -------
    list of dict
        Results for each cell.
    """
    global _worker_nodes_gdf, _worker_subgraph_cache, _worker_grid_id_col

    results = []

    for task in batch:
        cell_id = task[0]

        cache_entry = _worker_subgraph_cache.get(cell_id)
        if cache_entry is None:
            results.append(None)
            continue

        nx_subgraph = cache_entry['nx_subgraph']
        ig_graph = cache_entry['ig_graph']
        ig_mappings = cache_entry['ig_mappings']
        target_vertices = cache_entry['target_vertices']
        target_node_ids = cache_entry['target_node_ids']

        if len(target_node_ids) == 0:
            results.append(None)
            continue

        result = {'GridID': cell_id}

        try:
            # ================================================================
            # CENTRALITY METRICS - igraph
            # ================================================================

            # Betweenness Centrality
            try:
                btw_dict = calculate_betweenness_centrality_ig(ig_graph, target_vertices)
                btw_values = list(btw_dict.values())
                result['BtwCentM'] = np.mean(btw_values) if btw_values else np.nan
                result['BtwCentX'] = np.max(btw_values) if btw_values else np.nan
                result['BtwCentS'] = np.std(btw_values) if len(btw_values) > 1 else 0.0
            except Exception as e:
                logger.debug(f"Cell {cell_id} betweenness failed: {e}")
                result['BtwCentM'] = np.nan
                result['BtwCentX'] = np.nan
                result['BtwCentS'] = np.nan

            # Closeness Centrality
            try:
                cls_dict = calculate_closeness_centrality_ig(ig_graph, target_vertices)
                cls_values = list(cls_dict.values())
                result['ClsCentM'] = np.mean(cls_values) if cls_values else np.nan
                result['ClsCentX'] = np.max(cls_values) if cls_values else np.nan
                result['ClsCentS'] = np.std(cls_values) if len(cls_values) > 1 else 0.0
            except Exception as e:
                logger.debug(f"Cell {cell_id} closeness failed: {e}")
                result['ClsCentM'] = np.nan
                result['ClsCentX'] = np.nan
                result['ClsCentS'] = np.nan

            # Straightness Centrality
            try:
                str_dict = calculate_straightness_centrality_ig(ig_graph, target_vertices)
                str_values = list(str_dict.values())
                result['StrCentM'] = np.mean(str_values) if str_values else np.nan
                result['StrCentX'] = np.max(str_values) if str_values else np.nan
                result['StrCentS'] = np.std(str_values) if len(str_values) > 1 else 0.0
            except Exception as e:
                logger.debug(f"Cell {cell_id} straightness failed: {e}")
                result['StrCentM'] = np.nan
                result['StrCentX'] = np.nan
                result['StrCentS'] = np.nan

            # ================================================================
            # TOPOLOGY & GRAPH METRICS - NetworkX
            # ================================================================

            # Node Neighbor Density (Momepy, already fast)
            try:
                import momepy
                nx_with_degree = momepy.node_degree(nx_subgraph)
                node_dens_graph = momepy.node_density(nx_with_degree, radius=5, verbose=False)

                dens_vals = []
                for nid in target_node_ids:
                    if nid in node_dens_graph.nodes():
                        val = node_dens_graph.nodes[nid].get('node_density', np.nan)
                        if not np.isnan(val):
                            dens_vals.append(val)

                result['NdNbDenM'] = np.mean(dens_vals) if dens_vals else np.nan
                result['NdNbDenS'] = np.std(dens_vals) if len(dens_vals) > 1 else 0.0
            except:
                result['NdNbDenM'] = np.nan
                result['NdNbDenS'] = np.nan

            # Graph metrics (NetworkX, O(V+E))
            try:
                degrees = dict(nx_subgraph.degree())
                result['MeanDeg'] = np.mean(list(degrees.values()))

                # Mean Node Distance
                if nx.is_connected(nx_subgraph):
                    result['MeanNdDst'] = nx.average_shortest_path_length(nx_subgraph, weight='mm_len')
                else:
                    result['MeanNdDst'] = np.nan

                # Cyclomatic Complexity
                e = nx_subgraph.number_of_edges()
                v = nx_subgraph.number_of_nodes()
                p = nx.number_connected_components(nx_subgraph)
                result['Cyclomat'] = e - v + p

                # Edge-Node Ratio
                result['EdgeNdRat'] = e / v if v > 0 else np.nan

                # Gamma Index
                result['GammaIdx'] = e / (3 * (v - 2)) if v > 2 else np.nan

                # Meshedness
                result['Meshedness'] = (e - v + 1) / (2*v - 5) if v > 2 else np.nan
            except:
                result['MeanDeg'] = np.nan
                result['MeanNdDst'] = np.nan
                result['Cyclomat'] = np.nan
                result['EdgeNdRat'] = np.nan
                result['GammaIdx'] = np.nan
                result['Meshedness'] = np.nan

            # ================================================================
            # EDGE METRICS - igraph for edge betweenness
            # ================================================================

            try:
                # Edge Betweenness (igraph)
                edge_btw = ig_graph.edge_betweenness(weights='weight', directed=False)
                result['EdgeBtwM'] = np.mean(edge_btw) if edge_btw else np.nan

                # Edge Closeness & Straightness (NetworkX/momepy, not the bottleneck)
                import momepy
                node_cls_graph = momepy.closeness_centrality(nx_subgraph, weight='mm_len', name='closeness', verbose=False)
                momepy.mean_nodes(node_cls_graph, 'closeness')
                edge_cls_vals = [node_cls_graph.edges[edge].get('closeness', np.nan)
                                for edge in node_cls_graph.edges(keys=True)]
                edge_cls_vals = [v for v in edge_cls_vals if not np.isnan(v)]
                result['EdgeClsM'] = np.mean(edge_cls_vals) if edge_cls_vals else np.nan

                node_str_graph = momepy.straightness_centrality(nx_subgraph, weight='mm_len', normalized=False, name='straightness', verbose=False)
                momepy.mean_nodes(node_str_graph, 'straightness')
                edge_str_vals = [node_str_graph.edges[edge].get('straightness', np.nan)
                                for edge in node_str_graph.edges(keys=True)]
                edge_str_vals = [v for v in edge_str_vals if not np.isnan(v)]
                result['EdgeStrM'] = np.mean(edge_str_vals) if edge_str_vals else np.nan
            except:
                result['EdgeBtwM'] = np.nan
                result['EdgeClsM'] = np.nan
                result['EdgeStrM'] = np.nan

            # ================================================================
            # CUL-DE-SAC (NetworkX/momepy)
            # ================================================================

            try:
                import momepy
                nx_with_degree = momepy.node_degree(nx_subgraph)
                cds_len = momepy.cds_length(nx_with_degree, radius=None, mode='sum', verbose=False)
                result['CuldesacL'] = cds_len if cds_len is not None else np.nan
            except:
                result['CuldesacL'] = np.nan

            results.append(result)

        except Exception as e:
            logger.debug(f"Cell {cell_id} failed: {e}")
            results.append(None)

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def calculate_connectivity_metrics(
    grid: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    graph: nx.MultiGraph,
    spatial_graphs_pkl: Path,
    grid_id_col: str = 'GridID',
    skip_coins: bool = True,
    batch_size: int = 100
) -> gpd.GeoDataFrame:
    """
    Calculate connectivity metrics using the igraph-accelerated backend.

    API-compatible with the NetworkX and no-buffer backends; usable as a
    drop-in replacement.

    Parameters
    ----------
    grid : GeoDataFrame
        Grid cells.
    nodes_gdf : GeoDataFrame
        Street network nodes.
    edges_gdf : GeoDataFrame
        Street network edges.
    graph : nx.MultiGraph
        Street network graph.
    spatial_graphs_pkl : Path
        Path to spatial_graphs pkl file (contains adjacency).
    grid_id_col : str
        Grid ID column name.
    skip_coins : bool
        Not used by this backend (COINS is not computed here); present for
        signature compatibility with the other backends.
    batch_size : int
        Batch size for parallel processing.

    Returns
    -------
    GeoDataFrame
        Grid with connectivity metrics.
    """
    logger.info("=" * 80)
    logger.info("CONNECTIVITY METRICS - IGRAPH BACKEND")
    logger.info("=" * 80)

    import time
    start_time = time.time()

    from .connectivity_networkx import (
        precompute_spatial_mappings,
        build_grid_adjacency
    )

    logger.info(f"  Grid cells: {len(grid):,}")
    logger.info(f"  Network nodes: {len(nodes_gdf):,}")

    # Build spatial mappings
    logger.info("Building spatial mappings...")
    node_to_cells, edge_to_cells, cell_to_nodes, cell_to_edges = precompute_spatial_mappings(
        nodes_gdf, edges_gdf, grid, grid_id_col
    )

    # Build grid adjacency
    adjacency = build_grid_adjacency(grid, grid_id_col)

    # Build subgraph cache (NetworkX + igraph)
    cache_file = spatial_graphs_pkl.parent / "subgraph_cache_igraph.pkl"
    subgraph_cache = build_subgraph_cache(
        grid, adjacency, nodes_gdf, graph, cell_to_nodes,
        grid_id_col, cache_file
    )

    # Count valid vs invalid entries
    valid_count = sum(1 for entry in subgraph_cache.values() if entry is not None)
    logger.info(f"  Subgraph cache built: {len(subgraph_cache):,} total entries")
    logger.info(f"  Valid entries: {valid_count:,}")
    logger.info(f"  Invalid entries: {len(subgraph_cache) - valid_count:,}")

    # Prepare tasks
    valid_cells = [cell_id for cell_id, entry in subgraph_cache.items() if entry is not None]
    tasks = [(cell_id,) for cell_id in valid_cells]

    logger.info(f"  Valid cells to process: {len(tasks):,}")

    # Calculate metrics in parallel
    logger.info("Calculating connectivity metrics (igraph-powered)...")

    import multiprocessing as mp
    n_workers = min(14, mp.cpu_count())
    logger.info(f"  Using {n_workers} workers")

    all_results = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(nodes_gdf, subgraph_cache, grid_id_col)
    ) as executor:
        # Submit batches
        batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
        futures = [executor.submit(_calculate_all_metrics_batch, batch) for batch in batches]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing batches"):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch failed: {e}")

    # Merge results into grid
    logger.info("Merging results into grid...")

    results_df = gpd.GeoDataFrame([r for r in all_results if r is not None])

    if len(results_df) > 0:
        grid = grid.merge(results_df, on=grid_id_col, how='left')
        logger.info(f"  Merged {len(results_df):,} results")
    else:
        logger.warning("  No results to merge!")

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"igraph backend completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if len(valid_cells) > 0:
        logger.info(f"   Average time per cell: {elapsed/len(valid_cells):.3f}s")
    else:
        logger.warning("   No valid cells processed!")
    logger.info("=" * 80)

    return grid
