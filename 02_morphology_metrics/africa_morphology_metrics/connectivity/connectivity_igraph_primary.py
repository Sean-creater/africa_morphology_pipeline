"""
Street network connectivity metrics - no-buffer igraph backend (primary).

This is the primary, most fully-featured connectivity implementation. It
builds on the igraph-accelerated hybrid strategy (centrality via igraph,
simple statistics via NetworkX/momepy) but removes the 6-neighbor cell
buffer used by the earlier backends and adds clustering coefficient,
COINS street-continuity, and basic intersection-type metrics.

Design note - no buffer:
Earlier backends built each cell's subgraph from the cell's own nodes plus
the nodes of all directly adjacent grid cells (a 1-ring buffer), so that
shortest-path-based metrics had network context beyond the cell edge. This
backend uses only the nodes actually within the target cell
(cell_to_nodes[cell_id]), with no neighboring-cell nodes included. This
removes the adjacency-graph computation entirely, produces smaller
subgraphs (faster to build and to run centrality on), and avoids mixing
a cell's metrics with a variable, hard-to-reason-about amount of
surrounding context from neighbor cells - each cell's connectivity is now
purely a function of its own local network structure.

Design note - topology vs. geometry subgraphs:
Two different node/edge scopes are used depending on what a metric
measures:
- Topology metrics (Cyclomat, EdgeNdRat, GammaIdx, Meshedness) count ALL
  edges that touch a cell-internal node, including edges that extend
  outside the cell to a node in another cell. This reflects the true
  branching structure of the network at the cell's nodes.
- Geometry metrics (MeanNdDst, EdgeBtwM, EdgeClsM, EdgeStrM, CuldesacL,
  and the igraph-based centrality metrics) use only edges between two
  cell-internal nodes, so that very long edges reaching far outside the
  cell do not distort within-cell distances and centralities.
Similarly, MeanDeg/StdDeg are computed on the cell-internal subgraph,
while the dead-end/intersection/3-way/4-way counts used for
DeadendDen/InterDens/Ratio3way/Ratio4way/CuldesacR use each node's degree
in the *global* street network graph (not the subgraph), because taking a
node out of its subgraph can truncate its degree if some of its neighbors
fall in a different cell, which would otherwise undercount intersections.
For a MultiGraph, edges connecting the same node pair are also
deduplicated for the topology edge count, since parallel edges (e.g. at
roundabouts) should count as one connection, not one per key.

Design note - COINS:
COINS ("Continuity in Street Networks") is momepy's implementation of the
angle-based street continuity ("stroke") algorithm: street segments are
chained together into longer continuous strokes when they meet at another
segment within an angular deviation threshold, approximating how a human
would perceive a street as "going straight" rather than turning. Here,
`momepy.COINS` is run per-cell (`angle_threshold=30`, `flow_mode=False`,
i.e. undirected) on that cell's edges to derive stroke counts and mean
stroke length. It is skipped when the edge count for a cell exceeds 5000
(to bound worst-case per-cell runtime) or when the caller requests
`skip_coins=True`.
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
import time
from collections import defaultdict

from .igraph_utils import (
    nx_to_igraph,
    get_target_vertices,
    extract_vertex_values
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_node_ids_vectorized(nodes_gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    """
    Vectorized coordinate extraction (avoids slow iterrows).

    Returns node IDs as (x, y) coordinate tuples for NetworkX graph lookup.
    """
    if len(nodes_gdf) == 0:
        return []

    if 'x' in nodes_gdf.columns and 'y' in nodes_gdf.columns:
        return list(zip(nodes_gdf['x'].values, nodes_gdf['y'].values))
    else:
        return list(zip(
            nodes_gdf.geometry.x.values,
            nodes_gdf.geometry.y.values
        ))


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
    """
    result = {}
    n = g.vcount()

    x_coords = g.vs['x']
    y_coords = g.vs['y']

    for v_idx in target_vertices:
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
# SPATIAL MAPPING (no adjacency needed)
# ============================================================================

def precompute_spatial_mappings(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    grid_id_col: str = 'GridID'
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Compute cell -> nodes/edges spatial mappings (no grid adjacency).

    Only which nodes/edges belong to each cell is needed here; unlike the
    buffered backends, no neighbor-cell adjacency graph is built.

    Returns
    -------
    tuple of 4 dicts
        - node_to_cells: {node_idx: [cell_ids]} (kept for compatibility)
        - edge_to_cells: {edge_idx: [cell_ids]} (kept for compatibility)
        - cell_to_nodes: {cell_id: [node_indices]} (primary use)
        - cell_to_edges: {cell_id: [edge_indices]} (primary use)
    """
    logger.info("=" * 80)
    logger.info("SPATIAL MAPPING (NO BUFFER)")
    logger.info("=" * 80)

    overall_start = time.time()

    # Node -> Cells mapping (within predicate for nodes)
    logger.info("  [1/4] Computing node -> cells mapping...")
    start = time.time()

    nodes_with_cells = gpd.sjoin(
        nodes_gdf.reset_index(),
        grid[[grid_id_col, 'geometry']],
        how='left',
        predicate='within'  # Only nodes within the cell
    )

    node_to_cells = defaultdict(list)
    for idx, group in nodes_with_cells.groupby(level=0):
        cell_ids = group[grid_id_col].dropna().unique().tolist()
        if cell_ids:
            node_to_cells[idx] = cell_ids

    logger.info(f"      Mapped {len(node_to_cells):,} nodes in {time.time()-start:.1f}s")

    # Edge -> Cells mapping (intersects predicate for edges - partial overlap OK)
    logger.info("  [2/4] Computing edge -> cells mapping...")
    start = time.time()

    edges_with_cells = gpd.sjoin(
        edges_gdf.reset_index(),
        grid[[grid_id_col, 'geometry']],
        how='left',
        predicate='intersects'  # Edges partially in cell count
    )

    edge_to_cells = defaultdict(list)
    for idx, group in edges_with_cells.groupby(level=0):
        cell_ids = group[grid_id_col].dropna().unique().tolist()
        if cell_ids:
            edge_to_cells[idx] = cell_ids

    logger.info(f"      Mapped {len(edge_to_cells):,} edges in {time.time()-start:.1f}s")

    # Cell -> Nodes mapping (reverse)
    logger.info("  [3/4] Computing cell -> nodes mapping...")
    start = time.time()

    cell_to_nodes = defaultdict(list)
    for node_idx, cell_list in node_to_cells.items():
        for cell_id in cell_list:
            cell_to_nodes[cell_id].append(node_idx)

    logger.info(f"      Mapped {len(cell_to_nodes):,} cells in {time.time()-start:.1f}s")

    # Cell -> Edges mapping (reverse)
    logger.info("  [4/4] Computing cell -> edges mapping...")
    start = time.time()

    cell_to_edges = defaultdict(list)
    for edge_idx, cell_list in edge_to_cells.items():
        for cell_id in cell_list:
            cell_to_edges[cell_id].append(edge_idx)

    logger.info(f"      Mapped {len(cell_to_edges):,} cells in {time.time()-start:.1f}s")

    overall_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info(f"  SPATIAL MAPPINGS COMPLETED in {overall_time:.1f}s")
    logger.info("=" * 80)

    return dict(node_to_cells), dict(edge_to_cells), dict(cell_to_nodes), dict(cell_to_edges)


# ============================================================================
# SUBGRAPH CACHE BUILDING (no buffer; topology + geometry split)
# ============================================================================

def build_subgraph_cache(
    grid: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    graph: nx.MultiGraph,
    cell_to_nodes: dict,
    cell_to_edges: dict,
    grid_id_col: str = 'GridID',
    cache_file: Optional[Path] = None
) -> dict:
    """
    Build the per-cell subgraph cache using only cell-internal nodes.

    For each cell, two subgraphs are derived (see module docstring):
    1. Topology: all edges touching a cell-internal node, including edges
       that extend outside the cell (used for EdgeNdRat, GammaIdx,
       Cyclomat, Meshedness).
    2. Geometry: only edges between two cell-internal nodes (used for
       MeanNdDst, EdgeBtwM/ClsM/StrM, CuldesacL, and centrality).
    The geometry subgraph is converted to igraph for fast centrality, and
    cell edges are retained for COINS analysis.

    Cache structure:
    {
        cell_id: {
            'nx_subgraph_geom': NetworkX MultiGraph (geometry subgraph),
            'v_topo': int (cell-internal node count),
            'e_topo': int (unique edges touching cell nodes, incl. external),
            'p_topo': int (connected components in topology subgraph),
            'ig_graph': igraph Graph (based on geometry subgraph),
            'ig_mappings': {...},
            'target_node_ids': [...],
            'target_vertices': [...],
            'edge_indices': [...]  # For COINS analysis
        } or None
    }
    """
    if cache_file and cache_file.exists():
        logger.info(f"  Loading cached subgraphs from {cache_file.name}...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"  Failed to load cache: {e}")
            logger.warning("  Rebuilding cache...")

    logger.info("  Building subgraph cache (no buffer)...")
    logger.info(f"  Grid cells: {len(grid)}, cell_to_nodes entries: {len(cell_to_nodes)}")

    subgraph_cache = {}
    empty_count = 0
    conversion_fail_count = 0
    disconnected_count = 0

    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="  Building subgraphs"):
        cell_id = row[grid_id_col]

        # Only nodes in this cell (no neighboring cells)
        target_node_indices = cell_to_nodes.get(cell_id, [])

        if len(target_node_indices) == 0:
            subgraph_cache[cell_id] = None
            empty_count += 1
            continue

        target_nodes = nodes_gdf.iloc[target_node_indices]
        node_ids = extract_node_ids_vectorized(target_nodes)

        # ===================================================================
        # Topology subgraph (includes edges extending outside the cell)
        # vs. geometry subgraph (cell-internal edges only)
        # ===================================================================

        # Get all edges involving cell nodes (including to external nodes)
        edges_involving_target = set()
        for node in node_ids:
            edges_involving_target.update(graph.edges(node, keys=True))

        # Topology metrics (pre-computed for performance)
        v_topo = len(node_ids)  # Cell-internal nodes only

        # Count unique edge pairs (ignore parallel edges in the MultiGraph):
        # topology metrics (EdgeNdRat, GammaIdx) should count unique
        # connections between node pairs, not all parallel edges with
        # different keys.
        unique_edge_pairs = set()
        for u, v, key in edges_involving_target:
            edge_pair = tuple(sorted([u, v]))
            unique_edge_pairs.add(edge_pair)

        e_topo = len(unique_edge_pairs)  # Unique edge pairs (simple graph count)

        # Build topology subgraph (includes external nodes at edge endpoints)
        nx_subgraph_topo = graph.edge_subgraph(list(edges_involving_target)).copy()
        p_topo = nx.number_connected_components(nx_subgraph_topo)

        # Geometry subgraph: only edges between cell-internal nodes
        nx_subgraph_geom = graph.subgraph(node_ids).copy()

        if len(nx_subgraph_geom.nodes) == 0:
            subgraph_cache[cell_id] = None
            empty_count += 1
            continue

        # Take largest connected component if disconnected (geometry subgraph)
        if not nx.is_connected(nx_subgraph_geom):
            largest_cc = max(nx.connected_components(nx_subgraph_geom), key=len)
            nx_subgraph_geom = nx_subgraph_geom.subgraph(largest_cc).copy()
            disconnected_count += 1

        # Convert geometry subgraph to igraph (for centrality calculations)
        try:
            ig_graph, ig_mappings = nx_to_igraph(nx_subgraph_geom)
        except Exception as e:
            logger.debug(f"  Cell {cell_id}: igraph conversion failed: {e}")
            subgraph_cache[cell_id] = None
            conversion_fail_count += 1
            continue

        # Target nodes are all nodes in the cell (no filtering needed)
        target_node_ids = node_ids
        target_vertices = get_target_vertices(target_node_ids, ig_mappings['node_to_vertex'])

        # Edge indices for COINS analysis
        edge_indices = cell_to_edges.get(cell_id, [])

        subgraph_cache[cell_id] = {
            'nx_subgraph_geom': nx_subgraph_geom,
            'v_topo': v_topo,
            'e_topo': e_topo,
            'p_topo': p_topo,
            'ig_graph': ig_graph,
            'ig_mappings': ig_mappings,
            'target_node_ids': target_node_ids,
            'target_vertices': target_vertices,
            'edge_indices': edge_indices,
            'target_node_indices': target_node_indices  # For global degree lookup
        }

    valid_count = len(subgraph_cache) - empty_count - conversion_fail_count
    logger.info(f"  Valid subgraphs: {valid_count:,}")
    logger.info(f"  Empty cells (no nodes): {empty_count:,}")
    logger.info(f"  Conversion failures: {conversion_fail_count:,}")
    logger.info(f"  Disconnected (used largest CC): {disconnected_count:,}")

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
# UNIFIED METRIC CALCULATION
# ============================================================================

# Global worker variables
_worker_nodes_gdf = None
_worker_edges_gdf = None
_worker_subgraph_cache = None
_worker_grid_id_col = None
_worker_skip_coins = None


def _init_worker(nodes_gdf, edges_gdf, subgraph_cache, grid_id_col, skip_coins):
    """Initialize worker process with shared data."""
    global _worker_nodes_gdf, _worker_edges_gdf, _worker_subgraph_cache, _worker_grid_id_col, _worker_skip_coins
    _worker_nodes_gdf = nodes_gdf
    _worker_edges_gdf = edges_gdf
    _worker_subgraph_cache = subgraph_cache
    _worker_grid_id_col = grid_id_col
    _worker_skip_coins = skip_coins


def _calculate_all_metrics_batch(batch: List[Tuple]) -> List[Optional[Dict]]:
    """
    Calculate all metrics for a batch of cells.

    HYBRID STRATEGY (same as the igraph backend):
    - Centrality (bottleneck) -> igraph
    - Simple stats -> NetworkX
    """
    global _worker_nodes_gdf, _worker_subgraph_cache, _worker_grid_id_col

    results = []

    for task in batch:
        cell_id = task[0]

        cache_entry = _worker_subgraph_cache.get(cell_id)
        if cache_entry is None:
            results.append(None)
            continue

        # Geometry subgraph and pre-computed topology metrics
        nx_subgraph_geom = cache_entry.get('nx_subgraph_geom', cache_entry.get('nx_subgraph'))  # Fallback for old cache
        v_topo = cache_entry.get('v_topo')  # Cell-internal nodes count
        e_topo = cache_entry.get('e_topo')  # All edges involving cell nodes
        p_topo = cache_entry.get('p_topo')  # Connected components (topology)

        ig_graph = cache_entry['ig_graph']
        ig_mappings = cache_entry['ig_mappings']
        target_vertices = cache_entry['target_vertices']
        target_node_ids = cache_entry['target_node_ids']

        # For backward compatibility, use nx_subgraph_geom as nx_subgraph
        nx_subgraph = nx_subgraph_geom

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

            # Node Neighbor Density (momepy - geometry subgraph)
            try:
                import momepy
                nx_with_degree = momepy.node_degree(nx_subgraph_geom)
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

            # ================================================================
            # CLUSTERING COEFFICIENT (momepy/NetworkX)
            # ================================================================

            try:
                # Square clustering coefficient (momepy wrapper) - geometry subgraph.
                # Measures local clustering: how well a node's neighbors are
                # interconnected with each other.
                import momepy
                cluster_graph = momepy.clustering(nx_subgraph_geom, name='cluster')

                cluster_vals = []
                for nid in target_node_ids:
                    if nid in cluster_graph.nodes():
                        val = cluster_graph.nodes[nid].get('cluster', np.nan)
                        if not np.isnan(val):
                            cluster_vals.append(val)

                result['ClusterM'] = np.mean(cluster_vals) if cluster_vals else np.nan
                result['ClusterS'] = np.std(cluster_vals) if len(cluster_vals) > 1 else 0.0
            except Exception as e:
                logger.debug(f"Cell {cell_id} clustering failed: {e}")
                result['ClusterM'] = np.nan
                result['ClusterS'] = np.nan

            # Graph metrics (NetworkX, O(V+E))
            try:
                # MeanDeg/StdDeg use subgraph degrees: average connectivity
                # WITHIN the cell.
                degrees_subgraph = dict(nx_subgraph.degree())
                degree_vals_subgraph = list(degrees_subgraph.values())
                result['MeanDeg'] = np.mean(degree_vals_subgraph)
                result['StdDeg'] = np.std(degree_vals_subgraph) if len(degree_vals_subgraph) > 1 else 0.0

                # ================================================================
                # BASIC CONNECTIVITY METRICS
                # ================================================================
                # These use GLOBAL graph degrees, not subgraph degrees: when a
                # subgraph is built from only the nodes within a cell, a node's
                # degree can drop if some of its neighbors fall outside the
                # cell (e.g. degree 3 in the global graph but degree 1 in the
                # subgraph), which would undercount intersections. Global
                # degrees (from nodes_gdf) are used instead for the target
                # nodes in this cell.

                target_node_indices = cache_entry.get('target_node_indices', [])
                if target_node_indices and len(target_node_indices) > 0:
                    # Best case: use target_node_indices directly
                    target_nodes = _worker_nodes_gdf.iloc[target_node_indices]
                    global_degrees = target_nodes['degree'].values
                else:
                    # Fallback for old cache without target_node_indices:
                    # look up degrees from nodes_gdf using target_node_ids.
                    # Node IDs in the momepy graph are (x, y) coordinate tuples.
                    target_node_ids_set = set(target_node_ids)

                    node_id_to_degree = {}
                    for idx, row in _worker_nodes_gdf.iterrows():
                        node_coord = (row.geometry.x, row.geometry.y)
                        if node_coord in target_node_ids_set:
                            node_id_to_degree[node_coord] = row['degree']

                    global_degrees = [node_id_to_degree.get(nid, 0) for nid in target_node_ids]

                deadend_count = sum(1 for deg in global_degrees if deg == 1)
                junction_count = sum(1 for deg in global_degrees if deg >= 3)
                three_way_count = sum(1 for deg in global_degrees if deg == 3)
                four_way_count = sum(1 for deg in global_degrees if deg == 4)

                # Store raw counts (converted to densities after merging into the grid)
                result['DeadendCnt'] = deadend_count
                result['InterCnt'] = junction_count
                result['ThreeWayCnt'] = three_way_count
                result['FourWayCnt'] = four_way_count

                if junction_count > 0:
                    result['Ratio3way'] = three_way_count / junction_count
                    result['Ratio4way'] = four_way_count / junction_count
                    result['CuldesacR'] = deadend_count / junction_count
                else:
                    result['Ratio3way'] = np.nan
                    result['Ratio4way'] = np.nan
                    result['CuldesacR'] = np.nan

                # Mean Node Distance (geometry subgraph - cell-internal edges only)
                if nx.is_connected(nx_subgraph_geom):
                    result['MeanNdDst'] = nx.average_shortest_path_length(nx_subgraph_geom, weight='mm_len')
                else:
                    result['MeanNdDst'] = np.nan

                # ================================================================
                # TOPOLOGY METRICS - use pre-computed values (include extending edges)
                # ================================================================
                # These measure network connectivity structure and should
                # include ALL edges involving cell nodes, even if the edges
                # extend outside the cell boundary.
                #
                # v_topo: cell-internal nodes only
                # e_topo: all edges involving these nodes (including extending edges)
                # p_topo: connected components in the topology subgraph

                if v_topo is not None and e_topo is not None and p_topo is not None:
                    result['Cyclomat'] = e_topo - v_topo + p_topo
                    result['EdgeNdRat'] = e_topo / v_topo if v_topo > 0 else np.nan
                    result['GammaIdx'] = e_topo / (3 * (v_topo - 2)) if v_topo > 2 else np.nan
                    result['Meshedness'] = (e_topo - v_topo + 1) / (2*v_topo - 5) if v_topo > 2 else np.nan
                else:
                    # Fallback for old cache (backward compatibility)
                    e = nx_subgraph_geom.number_of_edges()
                    v = nx_subgraph_geom.number_of_nodes()
                    p = nx.number_connected_components(nx_subgraph_geom)
                    result['Cyclomat'] = e - v + p
                    result['EdgeNdRat'] = e / v if v > 0 else np.nan
                    result['GammaIdx'] = e / (3 * (v - 2)) if v > 2 else np.nan
                    result['Meshedness'] = (e - v + 1) / (2*v - 5) if v > 2 else np.nan
            except:
                result['MeanDeg'] = np.nan
                result['StdDeg'] = np.nan
                result['DeadendCnt'] = np.nan
                result['InterCnt'] = np.nan
                result['ThreeWayCnt'] = np.nan
                result['FourWayCnt'] = np.nan
                result['Ratio3way'] = np.nan
                result['Ratio4way'] = np.nan
                result['CuldesacR'] = np.nan
                result['MeanNdDst'] = np.nan
                result['Cyclomat'] = np.nan
                result['EdgeNdRat'] = np.nan
                result['GammaIdx'] = np.nan
                result['Meshedness'] = np.nan

            # ================================================================
            # EDGE METRICS - geometry subgraph (cell-internal edges only)
            # ================================================================
            # These measure edge importance and straightness, and use only
            # cell-internal edges to avoid distortion from very long edges
            # extending far outside the cell boundary.

            try:
                # Edge Betweenness (igraph - already based on geometry subgraph)
                edge_btw = ig_graph.edge_betweenness(weights='weight', directed=False)
                result['EdgeBtwM'] = np.mean(edge_btw) if edge_btw else np.nan

                # Edge Closeness & Straightness (geometry subgraph)
                import momepy
                node_cls_graph = momepy.closeness_centrality(nx_subgraph_geom, weight='mm_len', name='closeness', verbose=False)
                momepy.mean_nodes(node_cls_graph, 'closeness')
                edge_cls_vals = [node_cls_graph.edges[edge].get('closeness', np.nan)
                                for edge in node_cls_graph.edges(keys=True)]
                edge_cls_vals = [v for v in edge_cls_vals if not np.isnan(v)]
                result['EdgeClsM'] = np.mean(edge_cls_vals) if edge_cls_vals else np.nan

                node_str_graph = momepy.straightness_centrality(nx_subgraph_geom, weight='mm_len', normalized=False, name='straightness', verbose=False)
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
            # CUL-DE-SAC (NetworkX/momepy - geometry subgraph)
            # ================================================================
            # Cul-de-sac length is a local metric within the cell; the
            # geometry subgraph avoids counting extending edges.

            try:
                import momepy
                nx_with_degree = momepy.node_degree(nx_subgraph_geom)
                cds_len = momepy.cds_length(nx_with_degree, radius=None, mode='sum', verbose=False)
                result['CuldesacL'] = cds_len if cds_len is not None else np.nan
            except:
                result['CuldesacL'] = np.nan

            # ================================================================
            # COINS - Street Continuity Analysis (momepy)
            # ================================================================

            if not _worker_skip_coins:
                try:
                    import momepy
                    import warnings

                    # Get edges for this cell from the cache
                    edge_indices = cache_entry.get('edge_indices', [])

                    if len(edge_indices) > 0 and len(edge_indices) <= 5000:  # Bound worst-case COINS cost
                        # Extract the edges GeoDataFrame for this cell
                        cell_edges_gdf = _worker_edges_gdf.iloc[edge_indices].copy()

                        # Suppress COINS MultiGraph warnings: these occur when
                        # multiple edges connect the same node pair (e.g.
                        # roundabouts). COINS handles this by assuming a 0deg
                        # angle, which is correct for parallel edges.
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore',
                                                   message='.*identical.*overlapping.*',
                                                   category=UserWarning,
                                                   module='momepy.coins')

                            # angle_threshold=30: perceive streets as continuous
                            # when the angle deviation is under 30 degrees,
                            # matching human perception of "straight ahead" vs.
                            # "turning". flow_mode=False: undirected analysis,
                            # appropriate for bidirectional roads.
                            coins = momepy.COINS(cell_edges_gdf, angle_threshold=30, flow_mode=False)

                        stroke_gdf = coins.stroke_gdf()

                        result['COINSCnt'] = len(stroke_gdf)  # Number of stroke groups
                        result['COINSLen'] = stroke_gdf.geometry.length.mean() if len(stroke_gdf) > 0 else np.nan
                    else:
                        # Too many edges, or no edges - skip COINS
                        result['COINSCnt'] = np.nan
                        result['COINSLen'] = np.nan

                except Exception as e:
                    logger.debug(f"Cell {cell_id} COINS failed: {e}")
                    result['COINSCnt'] = np.nan
                    result['COINSLen'] = np.nan
            else:
                # COINS skipped by caller
                result['COINSCnt'] = np.nan
                result['COINSLen'] = np.nan

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
    Calculate connectivity metrics without a neighbor-cell buffer.

    Uses only nodes within the target cell (no adjacency graph needed),
    producing smaller, purely cell-local subgraphs, and additionally
    computes clustering coefficient, COINS street continuity, and basic
    intersection-type metrics (see module docstring for the topology vs.
    geometry subgraph distinction and the global-degree rationale).

    API-compatible with the NetworkX and igraph-buffered backends; usable
    as a drop-in replacement.

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
        Path to spatial_graphs pkl file (not used for adjacency by this
        backend; kept for signature compatibility, and its parent
        directory is used as the cache location).
    grid_id_col : str
        Grid ID column name.
    skip_coins : bool
        Skip COINS analysis (default True).
    batch_size : int
        Batch size for parallel processing.

    Returns
    -------
    GeoDataFrame
        Grid with connectivity metrics.
    """
    logger.info("=" * 80)
    logger.info("CONNECTIVITY METRICS - NO-BUFFER IGRAPH BACKEND")
    logger.info("=" * 80)
    logger.info("Features:")
    logger.info("  No neighbor cell buffer (cell-local subgraphs only)")
    logger.info("  No adjacency graph needed")
    logger.info("  igraph for centrality")
    logger.info("  Clustering coefficient analysis")
    logger.info(f"  COINS street continuity analysis ({'enabled' if not skip_coins else 'skipped'})")
    logger.info("=" * 80)

    start_time = time.time()

    logger.info(f"  Grid cells: {len(grid):,}")
    logger.info(f"  Network nodes: {len(nodes_gdf):,}")

    # Build simplified spatial mappings (no adjacency)
    logger.info("Building spatial mappings (no adjacency)...")
    node_to_cells, edge_to_cells, cell_to_nodes, cell_to_edges = precompute_spatial_mappings(
        nodes_gdf, edges_gdf, grid, grid_id_col
    )

    # Build subgraph cache (edge boundary / topology-geometry split)
    cache_file = spatial_graphs_pkl.parent / "subgraph_cache_no_buffer.pkl"
    subgraph_cache = build_subgraph_cache(
        grid, nodes_gdf, edges_gdf, graph, cell_to_nodes, cell_to_edges,
        grid_id_col, cache_file
    )

    # Count valid entries and check for topology fields (v_topo, e_topo, p_topo)
    valid_count = sum(1 for entry in subgraph_cache.values() if entry is not None)
    has_topo_metrics_count = sum(1 for entry in subgraph_cache.values()
                                 if entry is not None and 'v_topo' in entry and 'e_topo' in entry and 'p_topo' in entry)

    logger.info(f"  Subgraph cache built: {len(subgraph_cache):,} total entries")
    logger.info(f"  Valid entries: {valid_count:,}")
    logger.info(f"  Invalid entries: {len(subgraph_cache) - valid_count:,}")

    # Detect an older on-disk cache format missing the topology fields and
    # rebuild it: an older cache would silently fall back to the less
    # accurate per-subgraph edge/node counts for topology metrics.
    if valid_count > 0 and has_topo_metrics_count == 0:
        logger.warning("=" * 80)
        logger.warning("  Old cache format detected (missing v_topo, e_topo, p_topo fields).")
        logger.warning("  Deleting old cache and rebuilding with topology/geometry split...")
        logger.warning("=" * 80)

        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"  Deleted old cache: {cache_file.name}")

        logger.info("  Rebuilding cache...")
        subgraph_cache = build_subgraph_cache(
            grid, nodes_gdf, edges_gdf, graph, cell_to_nodes, cell_to_edges,
            grid_id_col, cache_file
        )

        valid_count = sum(1 for entry in subgraph_cache.values() if entry is not None)
        has_topo_metrics_count = sum(1 for entry in subgraph_cache.values()
                                     if entry is not None and 'v_topo' in entry)

        logger.info(f"  Cache rebuilt: {valid_count:,} valid entries, {has_topo_metrics_count:,} with topology fields")
        logger.info("=" * 80)

    # Prepare tasks
    valid_cells = [cell_id for cell_id, entry in subgraph_cache.items() if entry is not None]
    tasks = [(cell_id,) for cell_id in valid_cells]

    logger.info(f"  Valid cells to process: {len(tasks):,}")

    # Calculate metrics in parallel
    logger.info("Calculating connectivity metrics (no-buffer)...")

    import multiprocessing as mp
    n_workers = min(14, mp.cpu_count())
    logger.info(f"  Using {n_workers} workers")

    all_results = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(nodes_gdf, edges_gdf, subgraph_cache, grid_id_col, skip_coins)
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

        # ================================================================
        # Compute density metrics from counts
        # ================================================================
        logger.info("  Computing density metrics from counts...")

        if 'Area_m2' in grid.columns:
            area_m2 = grid['Area_m2'].replace(0, np.nan)

            grid['DeadendDen'] = grid['DeadendCnt'] / area_m2
            grid['InterDens'] = grid['InterCnt'] / area_m2

            logger.info("  Computed DeadendDen, InterDens from counts")

            # Ratio3way, Ratio4way, CuldesacR are already calculated as ratios
        else:
            logger.warning("  Area_m2 column not found, cannot compute densities")
            grid['DeadendDen'] = np.nan
            grid['InterDens'] = np.nan
    else:
        logger.warning("  No results to merge!")

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"No-buffer backend completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if len(valid_cells) > 0:
        logger.info(f"   Average time per cell: {elapsed/len(valid_cells):.3f}s")
    else:
        logger.warning("   No valid cells processed!")
    logger.info("=" * 80)

    return grid
