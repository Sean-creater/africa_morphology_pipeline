"""
Street network connectivity metrics - pure NetworkX backend.

This is the fallback implementation used when python-igraph is not
installed. All centrality and graph metrics are computed with NetworkX /
momepy directly (no C-accelerated igraph calls).

Design note - subgraph caching:
Each grid cell's local subgraph (the cell plus its adjacent cells, i.e. a
6-neighbor buffer) is built exactly once and cached (in-memory and
optionally on disk as a pickle), then reused for every metric computed on
that cell. This avoids rebuilding the same subgraph repeatedly across
separate centrality / topology / edge / cul-de-sac passes.

Design note - buffer:
Cells are analyzed using their own nodes plus the nodes of directly
adjacent grid cells (a 1-ring buffer), so that shortest-path-based metrics
(betweenness, closeness, straightness, mean node distance) have enough
surrounding network context near cell boundaries rather than being
artificially truncated at the cell edge. The no-buffer implementation in
this package removes this buffering; see its docstring for why.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import momepy
import networkx as nx
import logging
import warnings
import time
import pickle
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Tuple, List
from multiprocessing import cpu_count
from collections import defaultdict

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning, module='momepy')
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# GLOBAL WORKER STATE
# ============================================================================
_worker_nodes_gdf = None
_worker_subgraph_cache = None
_worker_grid_id_col = None


def _init_worker(nodes_gdf, subgraph_cache, grid_id_col):
    """Initialize worker process with pre-computed subgraphs."""
    global _worker_nodes_gdf, _worker_subgraph_cache, _worker_grid_id_col
    _worker_nodes_gdf = nodes_gdf
    _worker_subgraph_cache = subgraph_cache
    _worker_grid_id_col = grid_id_col


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_node_ids_vectorized(nodes_gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    """Vectorized coordinate extraction (avoids slow iterrows)."""
    if len(nodes_gdf) == 0:
        return []

    if 'x' in nodes_gdf.columns and 'y' in nodes_gdf.columns:
        return list(zip(nodes_gdf['x'].values, nodes_gdf['y'].values))
    else:
        return list(zip(
            nodes_gdf.geometry.x.values,
            nodes_gdf.geometry.y.values
        ))


def get_buffer_indices_fast(buffer_cell_ids: List, cell_to_items: Dict) -> set:
    """O(1)-per-cell lookup returning the union of items across buffer cells."""
    indices = set()
    for cell_id in buffer_cell_ids:
        indices.update(cell_to_items.get(cell_id, []))
    return indices


# ============================================================================
# SPATIAL MAPPING FUNCTIONS
# ============================================================================

def precompute_spatial_mappings(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    grid_id_col: str = 'GridID'
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Pre-compute node/edge <-> grid-cell spatial mappings once, up front."""
    logger.info("=" * 80)
    logger.info("PRE-COMPUTING SPATIAL MAPPINGS (ONE-TIME COST)")
    logger.info("=" * 80)

    overall_start = time.time()

    # Node -> Cells mapping
    logger.info("  [1/4] Computing node -> cells mapping...")
    start = time.time()

    nodes_with_cells = gpd.sjoin(
        nodes_gdf.reset_index(),
        grid[[grid_id_col, 'geometry']],
        how='left',
        predicate='within'
    )

    node_to_cells = defaultdict(list)
    for idx, group in nodes_with_cells.groupby(level=0):
        cell_ids = group[grid_id_col].dropna().unique().tolist()
        if cell_ids:
            node_to_cells[idx] = cell_ids

    logger.info(f"      Mapped {len(node_to_cells):,} nodes in {time.time()-start:.1f}s")

    # Edge -> Cells mapping
    logger.info("  [2/4] Computing edge -> cells mapping...")
    start = time.time()

    edges_with_cells = gpd.sjoin(
        edges_gdf.reset_index(),
        grid[[grid_id_col, 'geometry']],
        how='left',
        predicate='intersects'
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
    logger.info(f"  ALL SPATIAL MAPPINGS COMPUTED in {overall_time:.1f}s")
    logger.info("=" * 80)

    return dict(node_to_cells), dict(edge_to_cells), dict(cell_to_nodes), dict(cell_to_edges)


def build_grid_adjacency(grid: gpd.GeoDataFrame, grid_id_col: str = 'GridID') -> dict:
    """Build the grid cell adjacency map (cells sharing a boundary)."""
    logger.info(f"  Building grid cell adjacency for {len(grid):,} cells...")
    start_time = time.time()

    grid.sindex
    adjacency = {}

    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="    Grid adjacency",
                         disable=len(grid) < 100, mininterval=1.0):
        grid_id = row[grid_id_col]
        geom = row.geometry

        possible_matches_idx = list(grid.sindex.intersection(geom.bounds))
        possible_matches = grid.iloc[possible_matches_idx]

        neighbors = possible_matches[
            (possible_matches.geometry.touches(geom)) &
            (possible_matches[grid_id_col] != grid_id)
        ]

        adjacency[grid_id] = neighbors[grid_id_col].tolist()

    elapsed = time.time() - start_time
    logger.info(f"  Grid adjacency computed in {elapsed:.1f}s")

    return adjacency


# ============================================================================
# SUBGRAPH CACHING
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
    Build (or load) the per-cell subgraph cache, one subgraph per cell.

    This eliminates redundant subgraph construction across different
    metrics: each cell's subgraph is built once and reused for every
    calculation on that cell.
    """
    logger.info("=" * 80)
    logger.info("BUILDING SUBGRAPH CACHE (ONE-TIME COST)")
    logger.info("=" * 80)

    if cache_file and cache_file.exists():
        logger.info(f"  Loading cached subgraphs from {cache_file.name}...")
        start = time.time()
        with open(cache_file, 'rb') as f:
            subgraph_cache = pickle.load(f)
        logger.info(f"  Loaded {len(subgraph_cache):,} cached subgraphs in {time.time()-start:.1f}s")
        return subgraph_cache

    start_time = time.time()
    subgraph_cache = {}

    logger.info(f"  Building subgraphs for {len(grid):,} cells...")

    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="    Subgraphs", mininterval=1.0):
        cell_id = row[grid_id_col]

        # Get buffer cells (current + neighbors)
        neighbors = adjacency.get(cell_id, [])
        buffer_cell_ids = [cell_id] + list(neighbors)

        # Get nodes in buffer
        buffer_node_indices = get_buffer_indices_fast(buffer_cell_ids, cell_to_nodes)

        if len(buffer_node_indices) == 0:
            subgraph_cache[cell_id] = None
            continue

        # Build subgraph
        buffer_nodes = nodes_gdf.iloc[list(buffer_node_indices)]
        node_ids = extract_node_ids_vectorized(buffer_nodes)
        subgraph = graph.subgraph(node_ids).copy()

        # Extract largest connected component if needed
        if len(subgraph.nodes) > 0 and not nx.is_connected(subgraph):
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_cc).copy()

        subgraph_cache[cell_id] = subgraph if len(subgraph.nodes) > 0 else None

    elapsed = time.time() - start_time
    logger.info(f"  Built {len(subgraph_cache):,} subgraphs in {elapsed:.1f}s")

    if cache_file:
        logger.info(f"  Saving subgraph cache to {cache_file.name}...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(subgraph_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
        logger.info(f"  Cache saved ({cache_size_mb:.1f} MB)")

    logger.info("=" * 80)

    return subgraph_cache


# ============================================================================
# UNIFIED WORKER FUNCTION
# ============================================================================

def _calculate_all_metrics_batch(batch: List[Tuple]) -> List[Optional[Dict]]:
    """
    Calculate all metrics for a batch of cells in a single pass.

    For each cell:
    1. Get pre-built subgraph from cache
    2. Calculate centrality metrics
    3. Calculate topology/graph metrics
    4. Calculate edge metrics
    5. Calculate cul-de-sac metrics

    Returns all results in one dictionary per cell.
    """
    global _worker_nodes_gdf, _worker_subgraph_cache, _worker_grid_id_col

    results = []

    for task in batch:
        cell_id, target_node_indices = task

        # Get cached subgraph
        subgraph = _worker_subgraph_cache.get(cell_id)

        if subgraph is None or len(subgraph.nodes) == 0:
            results.append(None)
            continue

        result = {'GridID': cell_id}

        try:
            # Get target nodes for this cell
            target_nodes = _worker_nodes_gdf.iloc[target_node_indices]
            target_node_ids = extract_node_ids_vectorized(target_nodes)

            # Filter to nodes actually in subgraph
            target_node_ids = [nid for nid in target_node_ids if nid in subgraph.nodes()]

            if len(target_node_ids) == 0:
                results.append(None)
                continue

            # ================================================================
            # CENTRALITY METRICS
            # ================================================================
            try:
                btw_graph = momepy.betweenness_centrality(subgraph, weight='mm_len', name='betweenness', verbose=False)
                result['betweenness'] = [btw_graph.nodes[nid].get('betweenness', np.nan) for nid in target_node_ids]
            except:
                result['betweenness'] = [np.nan] * len(target_node_ids)

            try:
                cls_graph = momepy.closeness_centrality(subgraph, weight='mm_len', name='closeness', verbose=False)
                result['closeness'] = [cls_graph.nodes[nid].get('closeness', np.nan) for nid in target_node_ids]
            except:
                result['closeness'] = [np.nan] * len(target_node_ids)

            try:
                str_graph = momepy.straightness_centrality(subgraph, weight='mm_len', normalized=False, name='straightness', verbose=False)
                result['straightness'] = [str_graph.nodes[nid].get('straightness', np.nan) for nid in target_node_ids]
            except:
                result['straightness'] = [np.nan] * len(target_node_ids)

            # ================================================================
            # TOPOLOGY METRICS - Node Neighbor Density
            # ================================================================
            try:
                subgraph_with_degree = momepy.node_degree(subgraph)
                node_dens_graph = momepy.node_density(subgraph_with_degree, radius=5, verbose=False)

                dens_vals = []
                for nid in target_node_ids:
                    if nid in node_dens_graph.nodes():
                        val = node_dens_graph.nodes[nid].get('node_density', np.nan)
                        if not np.isnan(val):
                            dens_vals.append(val)

                if dens_vals:
                    result['NdNbDenM'] = np.mean(dens_vals)
                    result['NdNbDenS'] = np.std(dens_vals) if len(dens_vals) > 1 else 0.0
                else:
                    result['NdNbDenM'] = np.nan
                    result['NdNbDenS'] = np.nan
            except:
                result['NdNbDenM'] = np.nan
                result['NdNbDenS'] = np.nan

            # ================================================================
            # GRAPH METRICS
            # ================================================================
            try:
                degrees = [subgraph.degree(n) for n in subgraph.nodes()]
                result['MeanDeg'] = np.mean(degrees) if degrees else np.nan

                mean_dist_graph = momepy.mean_node_dist(subgraph, name='mean_dist', verbose=False)
                dist_vals = [mean_dist_graph.nodes[nid].get('mean_dist', np.nan) for nid in target_node_ids]
                dist_vals = [v for v in dist_vals if not np.isnan(v)]
                result['MeanNdDst'] = np.mean(dist_vals) if dist_vals else np.nan

                result['Cyclomat'] = momepy.cyclomatic(subgraph, radius=None, verbose=False)
                result['EdgeNdRat'] = momepy.edge_node_ratio(subgraph, radius=None, verbose=False)
                result['GammaIdx'] = momepy.gamma(subgraph, radius=None, verbose=False)
                result['Meshedness'] = momepy.meshedness(subgraph, radius=None, verbose=False)
            except:
                result['MeanDeg'] = np.nan
                result['MeanNdDst'] = np.nan
                result['Cyclomat'] = np.nan
                result['EdgeNdRat'] = np.nan
                result['GammaIdx'] = np.nan
                result['Meshedness'] = np.nan

            # ================================================================
            # EDGE METRICS
            # ================================================================
            try:
                edge_btw_graph = momepy.betweenness_centrality(subgraph, mode='edges', name='edge_betweenness', verbose=False)
                edge_btw_vals = [edge_btw_graph.edges[edge].get('edge_betweenness', np.nan) for edge in edge_btw_graph.edges(keys=True)]
                edge_btw_vals = [v for v in edge_btw_vals if not np.isnan(v)]
                result['EdgeBtwM'] = np.mean(edge_btw_vals) if edge_btw_vals else np.nan

                node_cls_graph = momepy.closeness_centrality(subgraph, name='closeness', verbose=False)
                momepy.mean_nodes(node_cls_graph, 'closeness')
                edge_cls_vals = [node_cls_graph.edges[edge].get('closeness', np.nan) for edge in node_cls_graph.edges(keys=True)]
                edge_cls_vals = [v for v in edge_cls_vals if not np.isnan(v)]
                result['EdgeClsM'] = np.mean(edge_cls_vals) if edge_cls_vals else np.nan

                node_str_graph = momepy.straightness_centrality(subgraph, name='straightness', verbose=False)
                momepy.mean_nodes(node_str_graph, 'straightness')
                edge_str_vals = [node_str_graph.edges[edge].get('straightness', np.nan) for edge in node_str_graph.edges(keys=True)]
                edge_str_vals = [v for v in edge_str_vals if not np.isnan(v)]
                result['EdgeStrM'] = np.mean(edge_str_vals) if edge_str_vals else np.nan
            except:
                result['EdgeBtwM'] = np.nan
                result['EdgeClsM'] = np.nan
                result['EdgeStrM'] = np.nan

            # ================================================================
            # CUL-DE-SAC METRICS
            # ================================================================
            try:
                subgraph_with_degree = momepy.node_degree(subgraph)
                cds_len = momepy.cds_length(subgraph_with_degree, radius=None, mode='sum', verbose=False)
                result['CuldesacL'] = cds_len if cds_len is not None else np.nan
            except:
                result['CuldesacL'] = np.nan

            results.append(result)

        except Exception as e:
            logger.debug(f"Cell {cell_id} failed: {e}")
            results.append(None)

    return results


# ============================================================================
# ORCHESTRATION FUNCTION
# ============================================================================

def calculate_all_metrics_unified(
    grid: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    subgraph_cache: dict,
    cell_to_nodes: dict,
    grid_id_col: str = 'GridID',
    batch_size: int = 50,
    max_workers: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Calculate all metrics for every grid cell using the cached subgraphs."""
    logger.info("=" * 80)
    logger.info("CALCULATING ALL METRICS (UNIFIED WITH SUBGRAPH CACHE)")
    logger.info("=" * 80)

    start_time = time.time()

    # Prepare tasks
    logger.info(f"  Preparing {len(grid):,} tasks...")
    tasks = []
    for idx, row in grid.iterrows():
        cell_id = row[grid_id_col]
        target_node_indices = cell_to_nodes.get(cell_id, [])
        tasks.append((cell_id, target_node_indices))

    # Batch tasks
    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
    logger.info(f"  Prepared {len(batches)} batches (size={batch_size})")

    if max_workers is None:
        max_workers = min(cpu_count(), len(batches))

    logger.info(f"  Processing with {max_workers} workers...")

    all_results = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(nodes_gdf, subgraph_cache, grid_id_col)
    ) as executor:
        futures = {executor.submit(_calculate_all_metrics_batch, batch): i for i, batch in enumerate(batches)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="  All metrics", mininterval=0.5):
            try:
                batch_results = future.result()
                all_results.extend([r for r in batch_results if r is not None])
            except Exception as e:
                logger.error(f"  Batch failed: {e}")

    logger.info(f"  Computed metrics for {len(all_results):,} cells")

    # ========================================================================
    # AGGREGATE RESULTS TO GRID
    # ========================================================================
    logger.info("  Aggregating results to grid...")

    # Initialize all metric columns
    metric_cols = [
        'BtwCentM', 'BtwCentX', 'BtwCentS',
        'ClsCentM', 'ClsCentX', 'ClsCentS',
        'StrCentM', 'StrCentX', 'StrCentS',
        'NdNbDenM', 'NdNbDenS',
        'MeanDeg', 'MeanNdDst', 'Cyclomat', 'EdgeNdRat', 'GammaIdx', 'Meshedness',
        'EdgeBtwM', 'EdgeClsM', 'EdgeStrM',
        'CuldesacL'
    ]

    for col in metric_cols:
        grid[col] = np.nan

    # Fill in results
    for result in all_results:
        cell_id = result['GridID']
        idx = grid[grid[grid_id_col] == cell_id].index
        if len(idx) == 0:
            continue
        idx = idx[0]

        # Centrality metrics
        btw_vals = [v for v in result.get('betweenness', []) if not np.isnan(v)]
        if btw_vals:
            grid.loc[idx, 'BtwCentM'] = np.mean(btw_vals)
            grid.loc[idx, 'BtwCentX'] = np.max(btw_vals)
            grid.loc[idx, 'BtwCentS'] = np.std(btw_vals) if len(btw_vals) > 1 else 0.0

        cls_vals = [v for v in result.get('closeness', []) if not np.isnan(v)]
        if cls_vals:
            grid.loc[idx, 'ClsCentM'] = np.mean(cls_vals)
            grid.loc[idx, 'ClsCentX'] = np.max(cls_vals)
            grid.loc[idx, 'ClsCentS'] = np.std(cls_vals) if len(cls_vals) > 1 else 0.0

        str_vals = [v for v in result.get('straightness', []) if not np.isnan(v)]
        if str_vals:
            grid.loc[idx, 'StrCentM'] = np.mean(str_vals)
            grid.loc[idx, 'StrCentX'] = np.max(str_vals)
            grid.loc[idx, 'StrCentS'] = np.std(str_vals) if len(str_vals) > 1 else 0.0

        # Other metrics (direct values)
        for col in ['NdNbDenM', 'NdNbDenS', 'MeanDeg', 'MeanNdDst', 'Cyclomat',
                    'EdgeNdRat', 'GammaIdx', 'Meshedness', 'EdgeBtwM', 'EdgeClsM',
                    'EdgeStrM', 'CuldesacL']:
            if col in result:
                grid.loc[idx, col] = result[col]

    elapsed = time.time() - start_time
    logger.info(f"  All metrics completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 80)

    return grid


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
    skip_coins: bool = False,  # Ignored: COINS is not computed by this backend
    batch_size: int = 50,
    max_workers: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Calculate connectivity metrics using the pure-NetworkX backend.

    Uses a 6-neighbor buffer around each cell for shortest-path-based
    metrics and a persistent per-cell subgraph cache so each subgraph is
    built only once and reused across all metric computations for that
    cell (see module docstring).

    Metrics (21 total):
    - Centrality: BtwCentM/X/S, ClsCentM/X/S, StrCentM/X/S (9)
    - Topology: NdNbDenM/S (2)
    - Graph: MeanDeg, MeanNdDst, Cyclomat, EdgeNdRat, GammaIdx, Meshedness (6)
    - Edge: EdgeBtwM, EdgeClsM, EdgeStrM (3)
    - Cul-de-sac: CuldesacL (1)
    """
    logger.info("=" * 80)
    logger.info("CONNECTIVITY METRICS - NETWORKX BACKEND")
    logger.info("=" * 80)

    overall_start = time.time()

    # Validate
    if len(grid) == 0 or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
        raise ValueError("Empty input data")
    if grid_id_col not in grid.columns:
        raise ValueError(f"Missing {grid_id_col} column")

    logger.info(f"  Grid: {len(grid):,} cells")
    logger.info(f"  Nodes: {len(nodes_gdf):,}")
    logger.info(f"  Edges: {len(edges_gdf):,}")
    logger.info(f"  Batch size: {batch_size}")

    # Pre-compute spatial mappings
    node_to_cells, edge_to_cells, cell_to_nodes, cell_to_edges = precompute_spatial_mappings(
        nodes_gdf, edges_gdf, grid, grid_id_col
    )

    # Build adjacency
    adjacency = build_grid_adjacency(grid, grid_id_col)

    # Build/load subgraph cache
    cache_file = spatial_graphs_pkl.parent / "subgraph_cache_networkx.pkl" if spatial_graphs_pkl else None
    subgraph_cache = build_subgraph_cache(
        grid, adjacency, nodes_gdf, graph, cell_to_nodes,
        grid_id_col, cache_file
    )

    grid = grid.copy()

    # Calculate all metrics in a unified pass
    grid = calculate_all_metrics_unified(
        grid, nodes_gdf, subgraph_cache, cell_to_nodes,
        grid_id_col, batch_size, max_workers
    )

    overall_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info(f"NetworkX backend completed in {overall_time:.1f}s ({overall_time/60:.1f} min)")
    logger.info("=" * 80)

    return grid


def get_connectivity_column_mappings() -> List[Tuple]:
    """
    Column mappings for connectivity metrics (31 total).

    This is the single source of truth for connectivity column metadata,
    used regardless of which backend (NetworkX, igraph, no-buffer) actually
    computed the values, since all backends share the same output schema.

    Returns
    -------
    list of tuple
        (Short_Name, Full_Name, Data_Source, Momepy_Category, Statistic_Type, Description)
    """
    return [
        # Basic connectivity density metrics (2 metrics)
        ('DeadendDen', 'Dead End Density', 'Momepy', 'Connectivity', 'Density',
         'Number of dead ends (degree-1 nodes) per unit area (count/m²)'),
        ('InterDens', 'Intersection Density', 'Momepy', 'Connectivity', 'Density',
         'Number of intersections (degree ≥ 3) per unit area (count/m²)'),

        # Intersection type ratios (3 metrics)
        ('Ratio3way', '3-Way Intersection Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of 3-way intersections to total intersections'),
        ('Ratio4way', '4-Way Intersection Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of 4-way intersections to total intersections'),
        ('CuldesacR', 'Cul-de-sac Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of dead ends to intersections'),

        # Node centrality metrics (9 metrics)
        ('BtwCentM', 'Betweenness Centrality Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean node betweenness centrality (measures nodes on shortest paths)'),
        ('BtwCentX', 'Betweenness Centrality Max', 'Momepy', 'Connectivity', 'Max',
         'Maximum node betweenness centrality in grid cell'),
        ('BtwCentS', 'Betweenness Centrality Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of node betweenness centrality (diversity)'),
        ('ClsCentM', 'Closeness Centrality Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean node closeness centrality (measures proximity to all nodes)'),
        ('ClsCentX', 'Closeness Centrality Max', 'Momepy', 'Connectivity', 'Max',
         'Maximum node closeness centrality in grid cell'),
        ('ClsCentS', 'Closeness Centrality Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of node closeness centrality (diversity)'),
        ('StrCentM', 'Straightness Centrality Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean node straightness centrality (ratio of euclidean to network distance)'),
        ('StrCentX', 'Straightness Centrality Max', 'Momepy', 'Connectivity', 'Max',
         'Maximum node straightness centrality in grid cell'),
        ('StrCentS', 'Straightness Centrality Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of node straightness centrality (diversity)'),

        # Clustering and local density (4 metrics)
        ('ClusterM', 'Clustering Coefficient Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean clustering coefficient (measures local connectivity density)'),
        ('ClusterS', 'Clustering Coefficient Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of clustering coefficient (diversity)'),
        ('NdNbDenM', 'Node Neighbour Density Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean local node density in spatial neighborhood'),
        ('NdNbDenS', 'Node Neighbour Density Std', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of local node density (diversity)'),

        # Global graph topology (7 metrics)
        ('MeanDeg', 'Mean Node Degree', 'Momepy', 'Connectivity', 'Mean',
         'Mean node degree (average number of edges per node)'),
        ('StdDeg', 'Std Node Degree', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of node degree (diversity indicator)'),
        ('MeanNdDst', 'Mean Node Distance', 'Momepy', 'Connectivity', 'Mean',
         'Mean shortest path distance between all node pairs (meters)'),
        ('Cyclomat', 'Cyclomatic Complexity', 'Momepy', 'Connectivity', 'Value',
         'Cyclomatic complexity (number of independent cycles in network)'),
        ('EdgeNdRat', 'Edge-Node Ratio', 'Momepy', 'Connectivity', 'Ratio',
         'Ratio of edges to nodes (measures connectivity density)'),
        ('GammaIdx', 'Gamma Index', 'Momepy', 'Connectivity', 'Index',
         'Gamma index (ratio of actual edges to maximum possible edges)'),
        ('Meshedness', 'Meshedness Coefficient', 'Momepy', 'Connectivity', 'Index',
         'Meshedness coefficient (measures network redundancy and connectivity)'),

        # Edge-level metrics (3 metrics)
        ('EdgeBtwM', 'Edge Betweenness Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean edge betweenness centrality (edges on shortest paths)'),
        ('EdgeClsM', 'Edge Closeness Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean edge closeness centrality'),
        ('EdgeStrM', 'Edge Straightness Mean', 'Momepy', 'Connectivity', 'Mean',
         'Mean edge straightness (linearity of street segments)'),

        # COINS (Continuity in Street Networks) (2 metrics)
        ('COINSCnt', 'COINS Street Count', 'Momepy', 'Connectivity', 'Count',
         'Number of continuous street segments identified by COINS algorithm'),
        ('COINSLen', 'COINS Mean Length', 'Momepy', 'Connectivity', 'Mean',
         'Mean length of continuous street segments (COINS, meters)'),

        # Cul-de-sac metrics (1 metric)
        ('CuldesacL', 'Cul-de-sac Length', 'Momepy', 'Connectivity', 'Sum',
         'Total length of cul-de-sac streets (dead-end streets, meters)')
    ]
