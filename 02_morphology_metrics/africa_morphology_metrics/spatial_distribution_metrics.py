"""
Spatial distribution metrics: grid-parallel computation and grid aggregation.

Computes five building-level spatial distribution metrics (Alignment,
MeanInterbuildingDistance, NeighborDistance, Neighbors, BuildingAdjacency)
per hex grid cell and aggregates them to the grid. Each cell is analyzed
independently using only the buildings whose centroids fall within that
cell (no neighboring-cell buffer), which keeps the per-cell spatial graphs
local and makes the cells embarrassingly parallel.

Performance design:
- A single upfront spatial join assigns every building to its grid cell,
  replacing one sjoin per grid cell with one sjoin total.
- The building GeoDataFrame is passed into the worker pool exactly once via
  the pool initializer, so each worker holds its own process-local
  reference (fork semantics on Unix) rather than having it serialized on
  every task. Each task then carries only a list of integer indices,
  reducing per-task payload from roughly 500 KB to roughly 100 bytes and
  worker-pool startup from minutes to seconds.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import momepy
import warnings
import logging
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning, module='momepy')
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# GLOBAL WORKER STATE (ZERO-COPY OPTIMIZATION)
# ============================================================================

# Set once per worker process by the pool initializer; holds the shared
# buildings GeoDataFrame and run parameters so per-task calls carry only
# building indices.
_WORKER_BUILDINGS = None
_WORKER_ORIENTATION_COL = None
_WORKER_K = None
_WORKER_TARGET_CRS = None


def _init_worker_pool(buildings_all, orientation_col, k, target_crs):
    """
    Initialize a worker process with data shared across all of its tasks.

    Called once per worker when the pool is created. The buildings
    GeoDataFrame is passed by reference (fork semantics on Unix), so it is
    not re-serialized for every task.

    Parameters
    ----------
    buildings_all : GeoDataFrame
        All buildings, transferred once at pool startup.
    orientation_col : str
        Column name for building orientation.
    k : int
        Number of nearest neighbors for the KNN graph.
    target_crs : str
        Target CRS for distance calculations.
    """
    global _WORKER_BUILDINGS, _WORKER_ORIENTATION_COL, _WORKER_K, _WORKER_TARGET_CRS

    _WORKER_BUILDINGS = buildings_all
    _WORKER_ORIENTATION_COL = orientation_col
    _WORKER_K = k
    _WORKER_TARGET_CRS = target_crs


# ============================================================================
# GRID ADJACENCY COMPUTATION
# ============================================================================

def build_grid_adjacency(grid, grid_id_col='grid_id'):
    """
    Build neighbor relationships for hexagonal grid cells.

    For each grid cell, finds neighboring cells (typically 6 for interior
    hexagons, fewer at the grid boundary) using the spatial ``touches``
    predicate.

    Parameters
    ----------
    grid : GeoDataFrame
        Grid cells (hexagonal).
    grid_id_col : str
        Column name for grid cell IDs.

    Returns
    -------
    dict
        Mapping grid_id -> list of neighbor grid_ids.
        Example: {1: [2, 3, 7, 8, 12, 13], 2: [1, 3, 4, ...], ...}

    Notes
    -----
    - Interior hexagons have 6 neighbors; boundary hexagons have fewer (3-5).
    - Uses the spatial index for efficient computation.
    - The result can be cached and reused for network analysis.
    """
    logger.info(f"Building grid adjacency graph for {len(grid):,} cells...")
    start_time = time.time()

    grid.sindex

    adjacency = {}

    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="  Computing neighbors", disable=len(grid) < 100):
        grid_id = row[grid_id_col]
        geom = row.geometry

        possible_matches_idx = list(grid.sindex.intersection(geom.bounds))
        possible_matches = grid.iloc[possible_matches_idx]

        neighbors = possible_matches[
            (possible_matches.geometry.touches(geom)) &
            (possible_matches[grid_id_col] != grid_id)
        ]

        neighbor_ids = neighbors[grid_id_col].tolist()
        adjacency[grid_id] = neighbor_ids

    elapsed = time.time() - start_time

    neighbor_counts = [len(neighbors) for neighbors in adjacency.values()]
    avg_neighbors = np.mean(neighbor_counts)

    logger.info(f"  ✓ Grid adjacency computed in {elapsed:.1f}s")
    logger.info(f"  Average neighbors per cell: {avg_neighbors:.1f}")
    logger.info(f"  Neighbor count distribution:")
    for n in range(7):
        count = sum(1 for nc in neighbor_counts if nc == n)
        if count > 0:
            logger.info(f"    {n} neighbors: {count:,} cells")

    return adjacency


def save_grid_adjacency(adjacency, output_path):
    """
    Save grid adjacency to CSV for later reuse.

    Format: grid_id, neighbor_1, neighbor_2, ..., neighbor_6
    """
    rows = []
    max_neighbors = max(len(neighbors) for neighbors in adjacency.values())

    for grid_id, neighbors in adjacency.items():
        row = [grid_id] + neighbors + [None] * (max_neighbors - len(neighbors))
        rows.append(row)

    columns = ['grid_id'] + [f'neighbor_{i+1}' for i in range(max_neighbors)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

    logger.info(f"  Grid adjacency saved: {output_path}")


def load_grid_adjacency(input_path):
    """Load grid adjacency from CSV."""
    df = pd.read_csv(input_path)

    adjacency = {}
    for _, row in df.iterrows():
        grid_id = row['grid_id']
        neighbors = [
            val for val in row[1:].values
            if pd.notna(val)
        ]
        adjacency[grid_id] = neighbors

    logger.info(f"  Grid adjacency loaded: {len(adjacency):,} cells")
    return adjacency


# ============================================================================
# ZERO-COPY WORKER FUNCTION
# ============================================================================

def _process_single_grid_cell(cell_id, target_indices):
    """
    Compute spatial distribution metrics for a single grid cell.

    Reads the shared buildings data from worker-global state initialized
    once at pool startup, so no building data is serialized per call, only
    the (tiny) list of building indices for this cell.

    Only buildings whose centroids fall within the cell are analyzed (no
    neighboring-cell buffer), consistent with the connectivity metrics.

    Parameters
    ----------
    cell_id : int/str
        Target grid cell ID.
    target_indices : list of int
        Building indices belonging to this cell.

    Returns
    -------
    tuple
        (cell_id, result_dict) where result_dict contains:
        {
            'indices': target_indices,
            'Alignment': [...],
            'MeanInterbuildingDistance': [...],
            'NeighborDistance': [...],
            'Neighbors': [...],
            'BuildingAdjacency': [...]
        }
        Returns (cell_id, None) if the cell has no buildings.
    """
    try:
        from libpysal import graph

        buildings_all = _WORKER_BUILDINGS
        orientation_col = _WORKER_ORIENTATION_COL
        k = _WORKER_K
        target_crs = _WORKER_TARGET_CRS

        if len(target_indices) == 0:
            return cell_id, None

        all_local_indices = target_indices

        local_buildings = buildings_all.iloc[all_local_indices].copy()

        # Delaunay/KNN graphs need at least k+1 points (KNN) and 3 points
        # (Delaunay) to be well-defined; degrade k or bail out to NaNs below
        # that threshold rather than erroring.
        min_buildings = max(k + 1, 3)
        if len(local_buildings) < min_buildings:
            if len(local_buildings) >= 3:
                k_local = min(k, len(local_buildings) - 1)
            else:
                return cell_id, {
                    'indices': target_indices,
                    'Alignment': [np.nan] * len(target_indices),
                    'MeanInterbuildingDistance': [np.nan] * len(target_indices),
                    'NeighborDistance': [np.nan] * len(target_indices),
                    'Neighbors': [np.nan] * len(target_indices),
                    'BuildingAdjacency': [np.nan] * len(target_indices)
                }
        else:
            k_local = k

        if local_buildings.crs.to_string() != target_crs:
            local_buildings = local_buildings.to_crs(target_crs)

        # Local spatial graphs, built from only this cell's buildings.
        delaunay_graph = graph.Graph.build_triangulation(
            local_buildings.centroid
        ).assign_self_weight()

        knn_graph = graph.Graph.build_knn(
            local_buildings.centroid,
            k=k_local
        ).assign_self_weight()

        # Queen contiguity (includes corner touches) for building_adjacency.
        contiguity_graph = graph.Graph.build_contiguity(
            local_buildings,
            rook=False
        )

        if orientation_col in local_buildings.columns:
            alignment_values = momepy.alignment(
                local_buildings[orientation_col],
                delaunay_graph
            )
        else:
            alignment_values = pd.Series([np.nan] * len(local_buildings), index=local_buildings.index)

        ibd_values = momepy.mean_interbuilding_distance(
            local_buildings,
            delaunay_graph,
            knn_graph
        )

        nd_values = momepy.neighbor_distance(
            local_buildings,
            delaunay_graph
        )

        neighbors_values = momepy.neighbors(
            local_buildings,
            delaunay_graph,
            weighted=False
        )

        # Ratio of joined built-up structures to buildings in the neighborhood.
        adjacency_values = momepy.building_adjacency(
            contiguity_graph,
            knn_graph
        )

        # Target-cell buildings were concatenated first, so their metrics
        # occupy the first target_count positions.
        target_count = len(target_indices)

        result = {
            'indices': target_indices,
            'Alignment': alignment_values.iloc[:target_count].tolist(),
            'MeanInterbuildingDistance': ibd_values.iloc[:target_count].tolist(),
            'NeighborDistance': nd_values.iloc[:target_count].tolist(),
            'Neighbors': neighbors_values.iloc[:target_count].tolist(),
            'BuildingAdjacency': adjacency_values.iloc[:target_count].tolist()
        }

        return cell_id, result

    except Exception as e:
        return cell_id, {
            'indices': target_indices,
            'Alignment': [np.nan] * len(target_indices),
            'MeanInterbuildingDistance': [np.nan] * len(target_indices),
            'NeighborDistance': [np.nan] * len(target_indices),
            'Neighbors': [np.nan] * len(target_indices),
            'BuildingAdjacency': [np.nan] * len(target_indices)
        }


# ============================================================================
# GRID-PARALLEL METRICS CALCULATION
# ============================================================================

def calculate_spatial_distribution_metrics_grid_parallel(
    buildings,
    grid,
    grid_id_col='grid_id',
    target_crs="ESRI:102022",
    k=5,
    orientation_col='Orientation',
    n_jobs=None,
    adjacency_cache=None
):
    """
    Calculate spatial distribution metrics with a shared-memory worker pool.

    Metrics computed (5 total):
    1. Alignment: orientation deviation from neighbors.
    2. MeanInterbuildingDistance: distance between adjacent buildings.
    3. NeighborDistance: distance to adjacent buildings.
    4. Neighbors: number of neighbors.
    5. BuildingAdjacency: ratio of joined structures to buildings in the neighborhood.

    Parameters
    ----------
    buildings : GeoDataFrame
        Building footprints with a stable index.
    grid : GeoDataFrame
        Grid cells for spatial partitioning.
    grid_id_col : str
        Column name for grid cell IDs (default: 'grid_id').
    target_crs : str
        Target CRS for distance calculations (default: "ESRI:102022" fallback).
        The main pipeline supplies city-specific UTM zones via the
        projection module for better accuracy (e.g. "EPSG:32631" for Lagos).
    k : int
        Number of nearest neighbors for the KNN graph.
    orientation_col : str
        Column name for building orientation.
    n_jobs : int
        Number of parallel workers (None = auto).
    adjacency_cache : str or dict
        Unused; retained for call-site compatibility.

    Returns
    -------
    GeoDataFrame
        Buildings with added metric columns and a grid_id column.

    Notes
    -----
    - Buildings must have a stable index (used for O(1) lookup).
    - Grid cells should be hexagonal with ~250m diameter.
    - Only buildings with centroids within each cell are analyzed (no
      neighboring-cell buffer).
    """
    logger.info("=" * 80)
    logger.info("SPATIAL DISTRIBUTION METRICS - GRID-PARALLEL")
    logger.info("=" * 80)
    logger.info(f"Buildings: {len(buildings):,}, Grid cells: {len(grid):,}")

    pipeline_start = time.time()

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    logger.info(f"Using {n_jobs} parallel workers")

    if not buildings.index.is_unique:
        logger.warning("  ⚠️  Buildings index is not unique, resetting index...")
        buildings = buildings.reset_index(drop=True)

    # Step 1: a single upfront spatial join assigns every building to its
    # grid cell, replacing one sjoin per grid cell with one sjoin total.
    logger.info("\n[1/5] Spatial join (buildings → grid)...")
    step_start = time.time()

    if grid_id_col not in grid.columns:
        logger.warning(f"  Grid missing '{grid_id_col}' column, using index")
        grid = grid.copy()
        grid[grid_id_col] = grid.index

    buildings_centroids = gpd.GeoDataFrame(
        buildings.drop(columns='geometry'),
        geometry=buildings.centroid,
        crs=buildings.crs
    )

    if grid.crs != buildings_centroids.crs:
        grid = grid.to_crs(buildings_centroids.crs)

    buildings_with_grid = gpd.sjoin(
        buildings_centroids,
        grid[[grid_id_col, 'geometry']],
        how='left',
        predicate='within'
    )

    if f'{grid_id_col}_right' in buildings_with_grid.columns:
        grid_join_col = f'{grid_id_col}_right'
    elif grid_id_col in buildings_with_grid.columns:
        grid_join_col = grid_id_col
    else:
        grid_join_col = 'index_right'
        logger.warning(f"  Using 'index_right' as grid ID column")

    buildings = buildings.copy()
    buildings[grid_id_col] = buildings_with_grid[grid_join_col]

    grid_to_indices = {}
    for building_idx, grid_id in buildings[grid_id_col].items():
        if pd.notna(grid_id):
            if grid_id not in grid_to_indices:
                grid_to_indices[grid_id] = []
            grid_to_indices[grid_id].append(building_idx)

    non_empty_cells = len(grid_to_indices)
    total_assigned = sum(len(indices) for indices in grid_to_indices.values())

    logger.info(f"  ✓ Spatial join completed ({time.time() - step_start:.1f}s)")
    logger.info(f"    Buildings assigned: {total_assigned:,} / {len(buildings):,}")
    logger.info(f"    Non-empty cells: {non_empty_cells:,} / {len(grid):,}")

    logger.info("\n[2/5] Grid adjacency graph (not required without a buffer zone)...")

    logger.info("\n[3/5] Preparing parallel tasks (index-only transfer)...")
    step_start = time.time()

    tasks = []
    for cell_id in grid_to_indices.keys():
        target_indices = grid_to_indices[cell_id]
        tasks.append((cell_id, target_indices))

    indices_size = sum(len(t[1]) for t in tasks)
    avg_indices_per_task = indices_size / len(tasks) if len(tasks) > 0 else 0

    logger.info(f"  ✓ {len(tasks):,} tasks prepared ({time.time() - step_start:.1f}s)")
    logger.info(f"    Average indices per task: {avg_indices_per_task:.1f}")
    logger.info(f"    Total indices to transfer: {indices_size:,}")
    logger.info(f"    Estimated task size: ~{avg_indices_per_task * 8:.0f} bytes")

    logger.info(f"\n[4/5] Parallel processing ({n_jobs} workers)...")
    logger.info(f"  Initializing worker pool with shared buildings data...")
    step_start = time.time()

    results = []
    with Pool(
        processes=n_jobs,
        initializer=_init_worker_pool,
        initargs=(buildings, orientation_col, k, target_crs)
    ) as pool:
        logger.info(f"  ✓ Worker pool initialized ({time.time() - step_start:.1f}s)")
        logger.info(f"  Processing {len(tasks):,} grid cells...")

        process_start = time.time()

        with tqdm(total=len(tasks), desc="  Processing grid cells", unit="cell") as pbar:
            # Process in batches so the progress bar updates incrementally.
            batch_size = max(1, len(tasks) // 20)  # ~5% updates
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = pool.starmap(_process_single_grid_cell, batch)
                results.extend(batch_results)
                pbar.update(len(batch))

    logger.info(f"  ✓ Parallel processing completed ({time.time() - process_start:.1f}s)")

    logger.info("\n[5/5] Merging results...")
    step_start = time.time()

    buildings['Alignment'] = np.nan
    buildings['MeanInterbuildingDistance'] = np.nan
    buildings['NeighborDistance'] = np.nan
    buildings['Neighbors'] = np.nan
    buildings['BuildingAdjacency'] = np.nan

    processed_count = 0
    empty_cells = 0

    for cell_id, result in results:
        if result is not None:
            indices = result['indices']
            buildings.loc[indices, 'Alignment'] = result['Alignment']
            buildings.loc[indices, 'MeanInterbuildingDistance'] = result['MeanInterbuildingDistance']
            buildings.loc[indices, 'NeighborDistance'] = result['NeighborDistance']
            buildings.loc[indices, 'Neighbors'] = result['Neighbors']
            buildings.loc[indices, 'BuildingAdjacency'] = result['BuildingAdjacency']
            processed_count += len(indices)
        else:
            empty_cells += 1

    logger.info(f"  ✓ Results merged: {processed_count:,} buildings ({time.time() - step_start:.1f}s)")
    logger.info(f"  Empty cells skipped: {empty_cells:,}")

    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ PIPELINE COMPLETED in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 80)

    logger.info("\n  Metric Summary:")
    for metric in ['Alignment', 'MeanInterbuildingDistance', 'NeighborDistance', 'Neighbors', 'BuildingAdjacency']:
        values = buildings[metric].dropna()
        if len(values) > 0:
            logger.info(f"    {metric:30s}: {values.mean():.2f} ± {values.std():.2f}")

    return buildings


# ============================================================================
# GRID AGGREGATION
# ============================================================================

def aggregate_spatial_metrics_to_grid(buildings, grid, grid_id_col='grid_id'):
    """
    Aggregate spatial distribution metrics from buildings to grid cells.

    Buildings already carry a grid_id column from the upfront spatial join
    performed by ``calculate_spatial_distribution_metrics_grid_parallel``,
    so no further spatial join is needed here.

    Parameters
    ----------
    buildings : GeoDataFrame
        Buildings with metrics and a grid_id column.
    grid : GeoDataFrame
        Grid cells.
    grid_id_col : str
        Column name for grid cell IDs.

    Returns
    -------
    GeoDataFrame
        Grid with aggregated spatial metrics.
    """
    logger.info(f"Aggregating spatial metrics from {len(buildings):,} buildings to {len(grid):,} grid cells...")

    if grid_id_col not in grid.columns:
        grid = grid.reset_index(names=grid_id_col)

    if grid_id_col not in buildings.columns:
        raise ValueError(
            f"Buildings missing '{grid_id_col}' column! "
            "Make sure to use calculate_spatial_distribution_metrics_grid_parallel() "
            "which performs the upfront spatial join."
        )

    logger.info(f"  ✓ Using existing grid_id column (no redundant spatial join)")

    buildings_per_cell = buildings.groupby(grid_id_col).size()
    logger.info(f"  Buildings assigned to {len(buildings_per_cell):,} grid cells")

    logger.info("  Aggregating metrics...")

    spatial_metrics = {
        'Alignment': ['mean', 'std'],
        'MeanInterbuildingDistance': ['mean', 'std'],
        'NeighborDistance': ['mean', 'std'],
        'Neighbors': ['mean', 'max', 'std'],
        'BuildingAdjacency': ['mean', 'std']
    }

    existing_metrics = {
        metric: aggs
        for metric, aggs in spatial_metrics.items()
        if metric in buildings.columns
    }

    grid_metrics = buildings.groupby(grid_id_col).agg(existing_metrics)

    column_mapping = {
        'Alignment': 'Align',
        'MeanInterbuildingDistance': 'IBDis',
        'NeighborDistance': 'NbDis',
        'Neighbors': 'Neigh',
        'BuildingAdjacency': 'BAdj'
    }

    grid_metrics.columns = [
        f"{agg.capitalize()}_{column_mapping.get(col, col)}"
        for col, agg in grid_metrics.columns
    ]
    grid_metrics = grid_metrics.reset_index()

    logger.info("  Merging results into grid...")
    grid = grid.merge(grid_metrics, on=grid_id_col, how="left", copy=False)

    non_null_cells = grid['Mean_Align'].notna().sum()
    logger.info(f"✓ Aggregation completed: {non_null_cells:,} cells with spatial metrics")

    logger.info("\n  Grid-level Metric Summary:")
    if 'Mean_Align' in grid.columns:
        logger.info(f"    Mean_Align:  {grid['Mean_Align'].mean():.2f}° ± {grid['Mean_Align'].std():.2f}°")
    if 'Mean_IBDis' in grid.columns:
        logger.info(f"    Mean_IBDis:  {grid['Mean_IBDis'].mean():.1f}m ± {grid['Mean_IBDis'].std():.1f}m")
    if 'Mean_Neigh' in grid.columns:
        logger.info(f"    Mean_Neigh:  {grid['Mean_Neigh'].mean():.1f} ± {grid['Mean_Neigh'].std():.1f}")

    return grid


# ============================================================================
# HIGH-LEVEL WRAPPER FUNCTION
# ============================================================================

def calculate_and_aggregate_spatial_metrics(
    buildings,
    grid,
    grid_id_col='grid_id',
    target_crs="ESRI:102022",
    k=5,
    orientation_col='Orientation',
    n_jobs=None,
    adjacency_cache=None
):
    """
    Compute per-building spatial distribution metrics and aggregate them to the grid.

    Combines ``calculate_spatial_distribution_metrics_grid_parallel`` and
    ``aggregate_spatial_metrics_to_grid`` into a single call.

    Parameters
    ----------
    buildings : GeoDataFrame
        Building footprints with an Orientation column.
    grid : GeoDataFrame
        Grid cells for spatial aggregation.
    grid_id_col : str
        Column name for grid cell IDs (default: 'grid_id').
    target_crs : str
        Target CRS for distance calculations (default: "ESRI:102022" fallback).
        The main pipeline supplies city-specific UTM zones via the
        projection module for better accuracy (e.g. "EPSG:32631" for Lagos).
    k : int
        Number of nearest neighbors for the KNN graph.
    orientation_col : str
        Column name for building orientation.
    n_jobs : int
        Number of parallel workers (None = auto).
    adjacency_cache : str or dict
        Unused; retained for call-site compatibility.

    Returns
    -------
    tuple
        (buildings_with_metrics, grid_with_metrics). buildings_with_metrics
        includes the grid_id column.
    """
    logger.info("=" * 80)
    logger.info("SPATIAL DISTRIBUTION METRICS PIPELINE - GRID-PARALLEL")
    logger.info("=" * 80)

    pipeline_start = time.time()

    buildings_with_metrics = calculate_spatial_distribution_metrics_grid_parallel(
        buildings,
        grid,
        grid_id_col=grid_id_col,
        target_crs=target_crs,
        k=k,
        orientation_col=orientation_col,
        n_jobs=n_jobs,
        adjacency_cache=adjacency_cache
    )

    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING TO GRID")
    logger.info("=" * 80)

    grid_with_metrics = aggregate_spatial_metrics_to_grid(
        buildings_with_metrics,
        grid,
        grid_id_col=grid_id_col
    )

    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ PIPELINE FINISHED in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 80)

    return buildings_with_metrics, grid_with_metrics


# ============================================================================
# COLUMN NAME MAPPING
# ============================================================================

def get_spatial_distribution_column_mappings():
    """Get column name mappings for the spatial distribution metrics (5 metrics)."""
    mappings = [
        # Alignment
        ('Mean_Align', 'Mean Building Alignment', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean deviation of building orientation from adjacent buildings (degrees)'),
        ('Std_Align', 'Std Building Alignment', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of building alignment deviations (diversity)'),

        # MeanInterbuildingDistance
        ('Mean_IBDis', 'Mean Interbuilding Distance', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean distance between adjacent buildings within neighborhood (meters)'),
        ('Std_IBDist', 'Std Interbuilding Distance', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of interbuilding distances (diversity)'),

        # NeighborDistance
        ('Mean_NbDis', 'Mean Neighbor Distance', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean distance to adjacent buildings based on spatial graph (meters)'),
        ('Std_NbDist', 'Std Neighbor Distance', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of neighbor distances (diversity)'),

        # Neighbors
        ('Mean_Neigh', 'Mean Neighbor Count', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean number of neighbors captured by spatial graph'),
        ('Max_Neigh', 'Max Neighbor Count', 'Momepy', 'Spatial Distribution', 'Max',
         'Maximum number of neighbors in grid cell'),
        ('Std_Neigh', 'Std Neighbor Count', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of neighbor counts (diversity)'),

        # BuildingAdjacency
        ('Mean_BAdj', 'Mean Building Adjacency', 'Momepy', 'Spatial Distribution', 'Mean',
         'Mean ratio of joined built-up structures to buildings in neighborhood (0-1)'),
        ('Std_BAdj', 'Std Building Adjacency', 'Momepy', 'Diversity', 'Std',
         'Standard deviation of building adjacency ratios (diversity)'),
    ]

    return mappings
