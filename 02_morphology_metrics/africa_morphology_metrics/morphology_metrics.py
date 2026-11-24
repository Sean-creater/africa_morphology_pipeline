"""
Morphology Metrics Calculation Module
Contains functions for calculating building morphological metrics using momepy.

This module provides functions to:
1. Calculate individual building morphological metrics
2. Aggregate building metrics to grid cells
3. Filter and process grid data

Metrics calculated:
- Area, Perimeter, CourtyardArea
- CircularCompactness, Squareness, Rectangularity
- Corners, Orientation, Elongation
- EquivalentRectangularIndex, CompactnessWeightedAxis
- LongestAxisLength, Convexity, SquareCompactness
- CentroidCornersMean, CentroidCornersStd
- SharedWalls
- CourtyardIndex, FacadeRatio, FormFactor, FractalDimension, ShapeIndex
"""

import geopandas as gpd
import pandas as pd
import momepy
import warnings
import logging
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

# Suppress warnings to reduce log clutter
warnings.filterwarnings('ignore', category=FutureWarning, module='momepy')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*swapaxes.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in arccos.*')


# ============================================================================
# RASTER SAMPLING FUNCTIONS
# ============================================================================

def extract_raster_values_from_centroids(buildings, raster_path, band=1, default_value=np.nan):
    """
    Extract raster values at building centroid locations.

    This function samples a raster (e.g., building height or volume) at the
    centroid of each building polygon. Buildings outside the raster extent
    will receive the default_value.

    Parameters:
    -----------
    buildings : GeoDataFrame
        Building footprints with geometry column
    raster_path : str or Path
        Path to the raster file (e.g., building_height.tif)
    band : int, optional
        Raster band to sample (default: 1)
    default_value : float, optional
        Value to assign when sampling fails or is outside raster (default: np.nan)

    Returns:
    --------
    Series
        Raster values at each building centroid location

    Examples:
    ---------
    >>> heights = extract_raster_values_from_centroids(buildings, 'height.tif')
    >>> buildings['height'] = heights
    """
    try:
        import rasterio
        from rasterio.sample import sample_gen
    except ImportError:
        logger.error("rasterio not installed. Install with: pip install rasterio")
        return pd.Series([default_value] * len(buildings), index=buildings.index)

    # Check if raster file exists
    if not Path(raster_path).exists():
        logger.warning(f"Raster file not found: {raster_path}. Using default value {default_value}")
        return pd.Series([default_value] * len(buildings), index=buildings.index)

    try:
        # Read raster and sample values
        with rasterio.open(raster_path) as src:
            # Get building centroids in the same CRS as raster
            centroids = buildings.geometry.centroid

            # Check CRS compatibility
            if buildings.crs is not None and src.crs is not None:
                # Transform centroids to raster CRS if needed
                if buildings.crs != src.crs:
                    logger.info(f"Transforming centroids from {buildings.crs} to raster CRS {src.crs}")
                    centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=buildings.crs)
                    centroids_gdf = centroids_gdf.to_crs(src.crs)
                    centroids = centroids_gdf.geometry

            # Extract coordinates and sample raster
            coords = [(geom.x, geom.y) for geom in centroids]

            # Sample raster values in one batch call for performance
            raw_samples = list(sample_gen(src, coords, indexes=band))

            # Process sampled values
            sampled_values = []
            for val in raw_samples:
                if val is None or (hasattr(val, '__len__') and len(val) == 0):
                    sampled_values.append(default_value)
                else:
                    # Handle array or scalar
                    v = val[0] if hasattr(val, '__len__') else val
                    # Check for nodata values
                    if src.nodata is not None and v == src.nodata:
                        sampled_values.append(default_value)
                    else:
                        sampled_values.append(v)

            return pd.Series(sampled_values, index=buildings.index, dtype='float64')

    except Exception as e:
        logger.error(f"Error sampling raster {raster_path}: {str(e)}")
        return pd.Series([default_value] * len(buildings), index=buildings.index)


def calculate_floors_from_height(height, floor_height=3.0, min_height=2.0):
    """
    Calculate estimated number of floors from building height.

    Rules:
    - Any building with height > 0: at least 1 floor (ground floor always exists)
    - Every full floor_height (3m): +1 floor
    - Remainder >= min_height (2m): +1 additional floor
    - Remainder < min_height (2m): no additional floor

    Parameters:
    -----------
    height : array-like
        Building heights in meters
    floor_height : float, optional
        Typical floor height in meters (default: 3.0)
    min_height : float, optional
        Minimum remainder height to count as additional floor (default: 2.0)

    Returns:
    --------
    array
        Estimated number of floors for each building

    Examples:
    ---------
    >>> heights = pd.Series([1.5, 2.8, 3.1, 5.5, 9.0, 9.5])
    >>> floors = calculate_floors_from_height(heights)
    >>> print(floors)
    [1, 1, 2, 2, 3, 3]
    """
    height = np.asarray(height, dtype='float64')

    # Calculate base floors and remainder
    base_floors = np.floor(height / floor_height)  # Complete floors
    remainder = height % floor_height              # Remaining height

    # Add extra floor if remainder >= min_height
    extra_floor = np.where(remainder >= min_height, 1, 0)
    floors = base_floors + extra_floor

    # Any building with height > 0 has at least 1 floor
    floors = np.maximum(floors, 1)

    # Handle invalid heights (<=0 or NaN)
    floors = np.where((height > 0) & np.isfinite(height), floors, 0)

    return floors


# ============================================================================
# OPTIMIZED HELPER FUNCTIONS
# ============================================================================

def _safe_centroid_corner_distance(geometry, eps=10):
    """
    Compute the mean and standard deviation of centroid-to-corner distances
    for each building, via direct per-geometry iteration.

    Iterating explicitly (rather than momepy's groupby().apply() approach)
    keeps geometry indices aligned one-to-one with the input GeoSeries, which
    matters for buildings with degenerate or irregular exterior rings.

    Parameters:
    -----------
    geometry : GeoSeries
        Building geometries
    eps : float
        Deviation from 180 degrees to consider a corner (default: 10)

    Returns:
    --------
    tuple of (mean_series, std_series)
    """
    means = []
    stds = []

    for geom in geometry:
        try:
            # Get exterior coordinates and centroid
            coords = np.array(geom.exterior.coords[:-1])  # Exclude last duplicate point
            centroid = np.array([geom.centroid.x, geom.centroid.y])

            if len(coords) < 3:
                means.append(np.nan)
                stds.append(np.nan)
                continue

            # Calculate angles using vectorized operations
            a = np.roll(coords, 1, axis=0)
            b = coords
            c = np.roll(coords, -1, axis=0)

            ba = a - b
            bc = c - b

            norms_ba = np.linalg.norm(ba, axis=1)
            norms_bc = np.linalg.norm(bc, axis=1)

            valid = (norms_ba > 0) & (norms_bc > 0)

            if not valid.any():
                means.append(np.nan)
                stds.append(np.nan)
                continue

            cosine_angle = np.zeros(len(coords))
            cosine_angle[valid] = np.sum(ba[valid] * bc[valid], axis=1) / (norms_ba[valid] * norms_bc[valid])
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

            angles = np.arccos(cosine_angle)
            degrees = np.degrees(angles)

            # Find corners
            true_angles = np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps)

            # Calculate distances from centroid to corners
            corner_coords = coords[true_angles]
            if len(corner_coords) > 0:
                distances = np.linalg.norm(corner_coords - centroid, axis=1)
                means.append(np.nanmean(distances))
                stds.append(np.nanstd(distances))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        except Exception:
            means.append(np.nan)
            stds.append(np.nan)

    return (
        pd.Series(means, index=geometry.index, dtype='float64'),
        pd.Series(stds, index=geometry.index, dtype='float64')
    )


def _vectorized_corners_squareness(geometry, eps=10):
    """
    Vectorized calculation of corners and squareness for all buildings at once.

    This is much faster than momepy's groupby().apply() approach because it:
    1. Extracts all coordinates in one batch operation
    2. Processes all buildings in vectorized loops
    3. Avoids Python-level iteration overhead
    4. Handles invalid geometries gracefully

    Parameters:
    -----------
    geometry : GeoSeries
        Building geometries
    eps : float
        Deviation from 180 degrees to consider a corner (default: 10)

    Returns:
    --------
    tuple of (corners_series, squareness_series)
    """
    corners_dict = {}
    squareness_dict = {}

    # Process each building individually to handle errors gracefully
    for idx, geom in geometry.items():
        try:
            # Get coordinates
            coords = np.array(geom.exterior.coords[:-1])  # Exclude last duplicate point

            if len(coords) < 3:
                corners_dict[idx] = 0
                squareness_dict[idx] = np.nan
                continue

            # Calculate angles using vectorized operations
            a = np.roll(coords, 1, axis=0)
            b = coords
            c = np.roll(coords, -1, axis=0)

            ba = a - b
            bc = c - b

            # Compute angles
            norms_ba = np.linalg.norm(ba, axis=1)
            norms_bc = np.linalg.norm(bc, axis=1)

            # Avoid division by zero
            valid = (norms_ba > 0) & (norms_bc > 0)

            if not valid.any():
                corners_dict[idx] = 0
                squareness_dict[idx] = np.nan
                continue

            cosine_angle = np.zeros(len(coords))
            cosine_angle[valid] = np.sum(ba[valid] * bc[valid], axis=1) / (norms_ba[valid] * norms_bc[valid])

            # Clip to avoid numerical errors in arccos
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

            angles = np.arccos(cosine_angle)
            degrees = np.degrees(angles)

            # Find corners (angles deviating from 180 by more than eps)
            true_angles = np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps)

            # Count corners
            corner_count = np.count_nonzero(true_angles)
            corners_dict[idx] = corner_count

            # Calculate squareness (mean deviation from 90 degrees at corners)
            if corner_count > 0:
                squareness_val = np.nanmean(np.abs(90 - degrees[true_angles]))
                squareness_dict[idx] = squareness_val
            else:
                squareness_dict[idx] = np.nan

        except Exception:
            # Skip problematic geometries
            corners_dict[idx] = 0
            squareness_dict[idx] = np.nan

    # Create series with matching indices
    corners_series = pd.Series(corners_dict, dtype='int64')
    squareness_series = pd.Series(squareness_dict, dtype='float64')

    return corners_series, squareness_series


def _optimized_shared_walls(buildings, grid=None, grid_id_col='grid_id', buffer_distance=0.1):
    """
    Optimized SharedWalls calculation using grid-based spatial partitioning.

    This is much faster than a global calculation because it:
    1. Groups buildings by grid cell first
    2. Only calculates SharedWalls within each grid cell
    3. Buildings in different grid cells cannot share walls
    4. Reduces complexity from O(n^2) to O(k*m^2) where k=grids, m=buildings/grid

    Parameters:
    -----------
    buildings : GeoDataFrame
        Building geometries with calculated metrics
    grid : GeoDataFrame, optional
        Grid cells. If provided, buildings will be grouped by grid cell.
        If None, falls back to global calculation (slow).
    grid_id_col : str
        Column name for grid cell IDs (default: 'grid_id')
    buffer_distance : float
        Distance to buffer buildings for adjacency detection (default: 0.1 meters)

    Returns:
    --------
    Series
        Length of shared walls for each building
    """
    if grid is not None:
        logger.info("  Calculating SharedWalls using grid-based optimization...")
        return _shared_walls_by_grid(buildings, grid, grid_id_col, buffer_distance)
    else:
        logger.info("  Calculating SharedWalls globally (slow, no grid provided)...")
        return _shared_walls_global(buildings, buffer_distance)


def _shared_walls_by_grid(buildings, grid, grid_id_col, buffer_distance=0.1):
    """
    Calculate SharedWalls grouped by grid cells using vectorized shapely operations.

    This is the recommended method for large datasets.
    """
    from tqdm import tqdm
    import shapely
    from shapely import STRtree

    # Step 1: Spatial join to assign buildings to grid cells
    logger.info("  Assigning buildings to grid cells...")
    building_centroids = gpd.GeoDataFrame(
        buildings.drop(columns='geometry'),
        geometry=buildings.centroid,
        crs=buildings.crs
    )

    # Ensure grid has grid_id_col
    if grid_id_col not in grid.columns:
        grid = grid.reset_index(names=grid_id_col)

    # Spatial join
    bldg_with_grid = gpd.sjoin(
        building_centroids,
        grid[[grid_id_col, 'geometry']],
        predicate='within',
        how='left'
    )

    # Initialize result dictionary
    shared_walls_dict = {}

    # Step 2: Group buildings by grid cell and calculate SharedWalls within each group
    grouped = bldg_with_grid.groupby(grid_id_col)
    total_grids = len(grouped)

    logger.info(f"  Processing {total_grids:,} grid cells...")

    for grid_id, group_indices in tqdm(grouped.groups.items(),
                                        total=total_grids,
                                        desc="    Grid cells",
                                        unit="grid"):
        # Get buildings in this grid cell
        cell_buildings = buildings.loc[group_indices]

        if len(cell_buildings) < 2:
            # No shared walls possible with single building
            for idx in group_indices:
                shared_walls_dict[idx] = 0.0
            continue

        # Convert to shapely array for vectorized operations
        geoms = cell_buildings.geometry.values
        indices = cell_buildings.index.values
        n_buildings = len(geoms)

        # Build spatial index (STRtree) for fast neighbor queries
        # This is faster than computing all pairwise distances
        tree = STRtree(geoms)

        # Process each building using spatial index
        for i in range(n_buildings):
            building_geom = geoms[i]
            building_idx = indices[i]

            # Use STRtree to find candidates within buffer distance
            # This is O(log n) instead of O(n)
            buffered = building_geom.buffer(buffer_distance)
            candidate_indices = tree.query(buffered, predicate='intersects')

            # Remove self from candidates
            candidate_indices = candidate_indices[candidate_indices != i]

            if len(candidate_indices) == 0:
                shared_walls_dict[building_idx] = 0.0
                continue

            # Get candidate geometries
            candidate_geoms = geoms[candidate_indices]

            # Batch distance calculation (vectorized) - verify they're truly adjacent
            distances = shapely.distance(building_geom, candidate_geoms)
            adjacent_mask = distances <= buffer_distance

            if not adjacent_mask.any():
                shared_walls_dict[building_idx] = 0.0
                continue

            # Get only adjacent geometries
            adjacent_geoms = candidate_geoms[adjacent_mask]

            # Batch intersection calculation (vectorized)
            intersections = shapely.intersection(building_geom, adjacent_geoms)

            # Vectorized length calculation using shapely.length()
            # This handles LineString, MultiLineString, and other geometry types
            lengths = shapely.length(intersections)
            total_shared = np.sum(lengths)

            shared_walls_dict[building_idx] = total_shared

    # Convert to Series with original index
    return pd.Series(shared_walls_dict, dtype='float64')


def _shared_walls_global(buildings, buffer_distance=0.1):
    """
    Calculate SharedWalls globally (slow - only use if no grid available).

    This is the fallback method for when a grid is not provided.
    """
    from tqdm import tqdm

    shared_walls = np.zeros(len(buildings))
    buildings.sindex

    idx_to_pos = {idx: pos for pos, idx in enumerate(buildings.index)}

    for idx, building in tqdm(buildings.iterrows(),
                               total=len(buildings),
                               desc="    SharedWalls (global)"):
        possible_matches_idx = list(buildings.sindex.intersection(building.geometry.bounds))
        neighbor_positions = [i for i in possible_matches_idx if buildings.index[i] != idx]

        if not neighbor_positions:
            continue

        candidates = buildings.iloc[neighbor_positions]

        total_shared = 0.0
        for _, neighbor in candidates.iterrows():
            distance = building.geometry.distance(neighbor.geometry)
            if distance <= buffer_distance:
                intersection = building.geometry.intersection(neighbor.geometry)
                if intersection.geom_type == 'LineString':
                    total_shared += intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    total_shared += sum(line.length for line in intersection.geoms)

        pos = idx_to_pos[idx]
        shared_walls[pos] = total_shared

    return pd.Series(shared_walls, index=buildings.index, dtype='float64')


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def _calculate_metrics_chunk(args):
    """
    Calculate metrics for a chunk of buildings (worker function for multiprocessing).

    Supports 3D metrics calculation when height/volume rasters are provided.

    Parameters:
    -----------
    args : tuple
        Tuple of (buildings_chunk, height_raster_path, volume_raster_path)
        - buildings_chunk: GeoDataFrame chunk to process
        - height_raster_path: Path to height raster or None
        - volume_raster_path: Path to volume raster or None

    Returns:
    --------
    GeoDataFrame
        Buildings chunk with calculated metrics (including 3D metrics if rasters provided)
    """
    # Unpack arguments
    buildings_chunk, height_raster_path, volume_raster_path = args

    # Make a copy to avoid modifying original
    buildings_chunk = buildings_chunk.copy()

    # Filter out invalid geometries that can cause issues
    valid_geom = buildings_chunk.geometry.is_valid & ~buildings_chunk.geometry.is_empty
    if not valid_geom.all():
        logger.warning(f"Found {(~valid_geom).sum()} invalid geometries in chunk, attempting to fix...")
        buildings_chunk.loc[~valid_geom, 'geometry'] = buildings_chunk.loc[~valid_geom, 'geometry'].buffer(0)
        # Recheck
        valid_geom = buildings_chunk.geometry.is_valid & ~buildings_chunk.geometry.is_empty
        if not valid_geom.all():
            logger.warning(f"Still have {(~valid_geom).sum()} invalid geometries after fix, removing them...")
            buildings_chunk = buildings_chunk[valid_geom].copy()

    # Basic geometric properties - fastest operations
    buildings_chunk['Area'] = buildings_chunk.geometry.area
    buildings_chunk['Perimeter'] = buildings_chunk.geometry.length

    # Fast metrics that don't require complex geometry analysis
    buildings_chunk['CourtyardArea'] = momepy.courtyard_area(buildings_chunk)
    buildings_chunk['CircularCompactness'] = momepy.circular_compactness(buildings_chunk)
    buildings_chunk['LongestAxisLength'] = momepy.longest_axis_length(buildings_chunk)
    buildings_chunk['SquareCompactness'] = momepy.square_compactness(buildings_chunk)
    buildings_chunk['CompactnessWeightedAxis'] = momepy.compactness_weighted_axis(buildings_chunk)
    buildings_chunk['EquivalentRectangularIndex'] = momepy.equivalent_rectangular_index(buildings_chunk)
    buildings_chunk['Rectangularity'] = momepy.rectangularity(buildings_chunk)
    buildings_chunk['Convexity'] = momepy.convexity(buildings_chunk)
    buildings_chunk['Elongation'] = momepy.elongation(buildings_chunk)
    buildings_chunk['Orientation'] = momepy.orientation(buildings_chunk)

    # Metrics derived from already-calculated values where possible
    buildings_chunk['CourtyardIndex'] = momepy.courtyard_index(buildings_chunk, courtyard_area=buildings_chunk['CourtyardArea'])
    buildings_chunk['FacadeRatio'] = momepy.facade_ratio(buildings_chunk)
    buildings_chunk['FractalDimension'] = momepy.fractal_dimension(buildings_chunk)
    buildings_chunk['ShapeIndex'] = momepy.shape_index(buildings_chunk, longest_axis_length=buildings_chunk['LongestAxisLength'])

    # Centroid metrics - use safe implementation to avoid momepy's index mismatch bug
    centroid_mean, centroid_std = _safe_centroid_corner_distance(buildings_chunk.geometry)
    buildings_chunk['CentroidCornersMean'] = centroid_mean
    buildings_chunk['CentroidCornersStd'] = centroid_std

    # Optimized corners and squareness calculation (vectorized)
    corners, squareness = _vectorized_corners_squareness(buildings_chunk.geometry)
    buildings_chunk['Corners'] = corners
    buildings_chunk['Squareness'] = squareness

    # ========================================================================
    # 3D METRICS - Height-dependent calculations
    # ========================================================================
    # Check if height data is available
    has_height_data = height_raster_path is not None and Path(height_raster_path).exists()
    if has_height_data:
        # Extract heights from raster at building centroids
        buildings_chunk['Height_90m'] = extract_raster_values_from_centroids(
            buildings_chunk, height_raster_path, default_value=3.0  # Default 1-floor if missing
        )
        # Clean height data
        buildings_chunk['Height_90m'] = buildings_chunk['Height_90m'].clip(lower=0, upper=200)
        buildings_chunk['Height_90m'] = buildings_chunk['Height_90m'].fillna(3.0)

        # FormFactor: 3D shape compactness (surface area to volume ratio)
        buildings_chunk['FormFactor'] = momepy.form_factor(buildings_chunk, buildings_chunk['Height_90m'])

        # H2F_Ratio: Height-to-footprint ratio
        buildings_chunk['H2F_Ratio'] = buildings_chunk['Height_90m'] / np.sqrt(buildings_chunk['Area'])

        # FloorArea: Estimated total floor area (for Floor Area Ratio calculation)
        floors = calculate_floors_from_height(buildings_chunk['Height_90m'])
        buildings_chunk['FloorArea'] = buildings_chunk['Area'] * floors

    # Volume data extraction (if volume raster available)
    has_volume_data = volume_raster_path is not None and Path(volume_raster_path).exists()
    if has_volume_data:
        buildings_chunk['Volume_90m'] = extract_raster_values_from_centroids(
            buildings_chunk, volume_raster_path, default_value=0.0  # Default 0 if missing
        )
        # Clean volume data - remove negative values and extreme outliers
        buildings_chunk['Volume_90m'] = buildings_chunk['Volume_90m'].clip(lower=0, upper=1e6)  # Cap at 1M m^3
        buildings_chunk['Volume_90m'] = buildings_chunk['Volume_90m'].fillna(0.0)

        # Calculate building-level volume density (for aggregation statistics)
        # VolumeDens at building level = Volume / FootprintArea
        buildings_chunk['VolumeDens_bldg'] = buildings_chunk['Volume_90m'] / buildings_chunk['Area']

    # SharedWalls requires grid context to be accurate; it is calculated globally
    # after merging chunks back together, inside aggregate_metrics_to_grid().
    buildings_chunk['SharedWalls'] = 0.0

    return buildings_chunk


def calculate_building_metrics(buildings, n_jobs=None, chunk_size=None, height_raster_path=None, volume_raster_path=None):
    """
    Calculate primary morphological metrics for buildings with optional parallel processing.

    This function calculates morphological metrics for each building using the momepy library.
    All metrics are added as new columns to the input GeoDataFrame. Supports multiprocessing
    for large datasets.

    If height_raster_path is provided, additional 3D metrics will be calculated:
    - FormFactor: 3D shape compactness
    - H2F_Ratio: Height-to-footprint ratio
    - FloorArea: Estimated total floor area (used for Floor Area Ratio calculation)

    If volume_raster_path is provided, volume data will be extracted and aggregated:
    - Volume_90m: Building volume from WSF3D raster (~90m resolution)
    - Volume metrics will be aggregated to grid (mean, max, std, VolumeDens)

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building polygons with valid geometries
    n_jobs : int, optional
        Number of parallel jobs. If None, uses CPU count - 1 (default: None)
        Set to 1 to disable parallel processing
    chunk_size : int, optional
        Number of buildings per chunk for parallel processing
        If None, automatically calculated based on total buildings and n_jobs
    height_raster_path : str or Path, optional
        Path to building height raster file (e.g., WSF3D height TIF).
        If provided, heights will be extracted and 3D metrics calculated.
        If None, height-dependent metrics will be skipped (default: None)
    volume_raster_path : str or Path, optional
        Path to building volume raster file (e.g., WSF3D volume TIF, ~90m resolution).
        If provided, volumes will be extracted at building centroids.
        If None, volume metrics will be skipped (default: None)

    Returns:
    --------
    GeoDataFrame
        Input GeoDataFrame with added metric columns

    Raises:
    -------
    ValueError
        If buildings GeoDataFrame is empty or has invalid geometries
    TypeError
        If input is not a GeoDataFrame

    Examples:
    ---------
    >>> buildings = gpd.read_file("buildings.geojson")
    >>> # Single-threaded
    >>> buildings_with_metrics = calculate_building_metrics(buildings, n_jobs=1)
    >>> # Multi-threaded (auto)
    >>> buildings_with_metrics = calculate_building_metrics(buildings)
    >>> # Custom parallel settings
    >>> buildings_with_metrics = calculate_building_metrics(buildings, n_jobs=4, chunk_size=50000)
    """
    # Input validation
    if not isinstance(buildings, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")

    if len(buildings) == 0:
        raise ValueError("Buildings GeoDataFrame is empty")

    if buildings.geometry.isna().any():
        logger.warning(f"Found {buildings.geometry.isna().sum()} null geometries, removing...")
        buildings = buildings[buildings.geometry.notna()].copy()

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
    elif n_jobs <= 0:
        n_jobs = max(1, cpu_count() + n_jobs)  # Allow negative indexing like -1

    # Determine chunk size
    n_buildings = len(buildings)
    if chunk_size is None:
        # Larger chunks reduce overhead; aim for ~2-3 chunks per worker
        chunk_size = max(10000, n_buildings // (n_jobs * 2))

    # Use single-threaded for small datasets or n_jobs=1
    use_parallel = n_jobs > 1 and n_buildings > 20000

    logger.info(f"Calculating metrics for {n_buildings:,} buildings...")
    if use_parallel:
        logger.info(f"Using {n_jobs} parallel workers with chunk size {chunk_size:,}")
    else:
        logger.info("Using single-threaded processing")

    # Check if height data is available
    has_height_data = height_raster_path is not None and Path(height_raster_path).exists()
    if has_height_data:
        logger.info(f"Height raster found: {height_raster_path}")
        logger.info("  Will calculate 3D metrics: FormFactor, H2F_Ratio, FloorArea")
    else:
        if height_raster_path is not None:
            logger.warning(f"Height raster not found: {height_raster_path}")
        logger.info("  Skipping 3D metrics (no height data)")

    # Check if volume data is available
    has_volume_data = volume_raster_path is not None and Path(volume_raster_path).exists()
    if has_volume_data:
        logger.info(f"Volume raster found: {volume_raster_path}")
        logger.info("  Will extract volume data at ~90m resolution")
    else:
        if volume_raster_path is not None:
            logger.warning(f"Volume raster not found: {volume_raster_path}")
        logger.info("  Skipping volume metrics (no volume data)")

    try:
        if use_parallel:
            # Parallel processing
            # Split buildings into chunks
            n_chunks = max(1, n_buildings // chunk_size)
            chunks = np.array_split(buildings, n_chunks)

            logger.info(f"Processing {len(chunks)} chunks in parallel...")

            # Prepare arguments for each chunk (chunk, height_raster_path, volume_raster_path)
            chunk_args = [
                (chunk, height_raster_path, volume_raster_path)
                for chunk in chunks
            ]

            # Process chunks in parallel
            with Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(_calculate_metrics_chunk, chunk_args),
                    total=len(chunks),
                    desc="  Calculating building metrics",
                    unit="chunk"
                ))

            # Combine results
            buildings = pd.concat(results, ignore_index=False)

            # SharedWalls is calculated later with grid information inside
            # aggregate_metrics_to_grid(); chunks only carry a 0.0 placeholder.

            return buildings

        else:
            # Single-threaded processing with optimizations
            # Define metrics count for progress tracking
            total_metrics = 19  # Base metrics
            if has_height_data:
                total_metrics += 4  # Height extraction + FormFactor + H2F_Ratio + FloorArea
            if has_volume_data:
                total_metrics += 1  # Volume extraction

            with tqdm(total=total_metrics, desc="  Calculating building metrics", unit="metric") as pbar:
                # Basic geometric properties (fast operations)
                pbar.set_postfix_str("Area & Perimeter")
                buildings['Area'] = buildings.geometry.area
                buildings['Perimeter'] = buildings.geometry.length
                pbar.update(1)

                # Fast momepy metrics
                pbar.set_postfix_str("CourtyardArea")
                buildings['CourtyardArea'] = momepy.courtyard_area(buildings)
                pbar.update(1)

                pbar.set_postfix_str("CircularCompactness")
                buildings['CircularCompactness'] = momepy.circular_compactness(buildings)
                pbar.update(1)

                pbar.set_postfix_str("LongestAxisLength")
                buildings['LongestAxisLength'] = momepy.longest_axis_length(buildings)
                pbar.update(1)

                pbar.set_postfix_str("SquareCompactness")
                buildings['SquareCompactness'] = momepy.square_compactness(buildings)
                pbar.update(1)

                pbar.set_postfix_str("CompactnessWeightedAxis")
                buildings['CompactnessWeightedAxis'] = momepy.compactness_weighted_axis(buildings)
                pbar.update(1)

                pbar.set_postfix_str("EquivalentRectangularIndex")
                buildings['EquivalentRectangularIndex'] = momepy.equivalent_rectangular_index(buildings)
                pbar.update(1)

                pbar.set_postfix_str("Rectangularity")
                buildings['Rectangularity'] = momepy.rectangularity(buildings)
                pbar.update(1)

                pbar.set_postfix_str("Convexity")
                buildings['Convexity'] = momepy.convexity(buildings)
                pbar.update(1)

                pbar.set_postfix_str("Elongation")
                buildings['Elongation'] = momepy.elongation(buildings)
                pbar.update(1)

                pbar.set_postfix_str("Orientation")
                buildings['Orientation'] = momepy.orientation(buildings)
                pbar.update(1)

                # Centroid metrics - use safe implementation
                pbar.set_postfix_str("CentroidCorners")
                centroid_mean, centroid_std = _safe_centroid_corner_distance(buildings.geometry)
                buildings['CentroidCornersMean'] = centroid_mean
                buildings['CentroidCornersStd'] = centroid_std
                pbar.update(1)

                # Optimized corners and squareness (vectorized)
                pbar.set_postfix_str("Corners & Squareness")
                corners, squareness = _vectorized_corners_squareness(buildings.geometry)
                buildings['Corners'] = corners
                buildings['Squareness'] = squareness
                pbar.update(2)  # Count as 2 steps

                # Metrics derived from already-calculated values where possible
                pbar.set_postfix_str("CourtyardIndex")
                buildings['CourtyardIndex'] = momepy.courtyard_index(buildings, courtyard_area=buildings['CourtyardArea'])
                pbar.update(1)

                pbar.set_postfix_str("FacadeRatio")
                buildings['FacadeRatio'] = momepy.facade_ratio(buildings)
                pbar.update(1)

                pbar.set_postfix_str("FractalDimension")
                buildings['FractalDimension'] = momepy.fractal_dimension(buildings)
                pbar.update(1)

                pbar.set_postfix_str("ShapeIndex")
                buildings['ShapeIndex'] = momepy.shape_index(buildings, longest_axis_length=buildings['LongestAxisLength'])
                pbar.update(1)

                # Height-dependent 3D metrics (if height data available)
                if has_height_data:
                    pbar.set_postfix_str("Extracting heights")
                    buildings['Height_90m'] = extract_raster_values_from_centroids(
                        buildings, height_raster_path, default_value=3.0  # Default 1-floor if missing
                    )
                    # Clean height data
                    buildings['Height_90m'] = buildings['Height_90m'].clip(lower=0, upper=200)
                    buildings['Height_90m'] = buildings['Height_90m'].fillna(3.0)
                    pbar.update(1)

                    pbar.set_postfix_str("FormFactor")
                    buildings['FormFactor'] = momepy.form_factor(buildings, buildings['Height_90m'])
                    pbar.update(1)

                    pbar.set_postfix_str("H2F_Ratio")
                    buildings['H2F_Ratio'] = buildings['Height_90m'] / np.sqrt(buildings['Area'])
                    pbar.update(1)

                    pbar.set_postfix_str("FloorArea")
                    floors = calculate_floors_from_height(buildings['Height_90m'])
                    buildings['FloorArea'] = buildings['Area'] * floors
                    pbar.update(1)

                # Volume data extraction (if volume raster available)
                if has_volume_data:
                    pbar.set_postfix_str("Extracting volumes")
                    buildings['Volume_90m'] = extract_raster_values_from_centroids(
                        buildings, volume_raster_path, default_value=0.0  # Default 0 if missing
                    )
                    # Clean volume data - remove negative values and extreme outliers
                    buildings['Volume_90m'] = buildings['Volume_90m'].clip(lower=0, upper=1e6)  # Cap at 1M m^3
                    buildings['Volume_90m'] = buildings['Volume_90m'].fillna(0.0)

                    # Calculate building-level volume density (for aggregation statistics)
                    # VolumeDens at building level = Volume / FootprintArea
                    buildings['VolumeDens_bldg'] = buildings['Volume_90m'] / buildings['Area']
                    pbar.update(1)

                # SharedWalls is calculated later with grid information, inside
                # aggregate_metrics_to_grid(); set to 0.0 for now.
                buildings['SharedWalls'] = 0.0

            return buildings

    except Exception as e:
        logger.error(f"Error calculating building metrics: {str(e)}")
        raise


def calculate_individual_metric(buildings, metric_name, **kwargs):
    """
    Calculate a single morphological metric for buildings.

    This is useful when you only need specific metrics instead of all of them.

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building polygons
    metric_name : str
        Name of the metric to calculate. Options:
        'Area', 'Perimeter', 'CourtyardArea', 'CircularCompactness',
        'Corners', 'Squareness', 'EquivalentRectangularIndex',
        'Elongation', 'CentroidCorners', 'Orientation',
        'LongestAxisLength', 'CompactnessWeightedAxis',
        'Convexity', 'Rectangularity', 'SquareCompactness', 'SharedWalls'
    **kwargs : dict
        Additional arguments specific to the metric

    Returns:
    --------
    pandas.Series or tuple
        Series with calculated metric values (or tuple for CentroidCorners)

    Examples:
    ---------
    >>> area = calculate_individual_metric(buildings, 'Area')
    >>> elongation = calculate_individual_metric(buildings, 'Elongation')
    """
    metric_functions = {
        'Area': lambda b, k: b.geometry.area,
        'Perimeter': lambda b, k: b.geometry.length,
        'CourtyardArea': lambda b, k: momepy.courtyard_area(b),
        'CircularCompactness': lambda b, k: momepy.circular_compactness(b),
        'Corners': lambda b, k: momepy.corners(b),
        'Squareness': lambda b, k: momepy.squareness(b),
        'EquivalentRectangularIndex': lambda b, k: momepy.equivalent_rectangular_index(b),
        'Elongation': lambda b, k: momepy.elongation(b),
        'CentroidCorners': lambda b, k: momepy.centroid_corner_distance(b),
        'Orientation': lambda b, k: momepy.orientation(b),
        'LongestAxisLength': lambda b, k: momepy.longest_axis_length(b),
        'CompactnessWeightedAxis': lambda b, k: momepy.compactness_weighted_axis(b),
        'Convexity': lambda b, k: momepy.convexity(b),
        'Rectangularity': lambda b, k: momepy.rectangularity(b),
        'SquareCompactness': lambda b, k: momepy.square_compactness(b),
        'SharedWalls': lambda b, k: momepy.shared_walls(b),
    }

    if metric_name not in metric_functions:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(metric_functions.keys())}")

    return metric_functions[metric_name](buildings, kwargs)


# ============================================================================
# GRID AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_metrics_to_grid(buildings, grid, grid_id_col='grid_id', aggregation_methods=None):
    """
    Aggregate building metrics to grid cells with metric-specific aggregation rules.

    This function performs a spatial join between buildings and grid cells,
    then aggregates building metrics using meaningful statistical methods for each metric type.

    Since all grid cells have the same area (Area_m2), sum metrics are converted to
    density metrics (per unit area) to avoid redundancy and provide more meaningful indicators.

    Aggregation rules follow momepy categories:
    - Dimension (Area, Perimeter, etc.): mean, max, std only (no sum)
    - Shape (compactness, etc.): mean, std only
    - Spatial Distribution (Orientation): circular_mean, circular_std
    - Intensity (density metrics - all sum metrics converted to densities):
        * BldgDens: building_count / cell_area (buildings per m^2)
        * CoverRatio: building_area_sum / cell_area (0-1 ratio)
        * OpenRatio: 1 - CoverRatio (0-1 ratio)
        * PerimDens: perimeter_sum / cell_area (m/m^2)
        * CourtDens: courtyard_area_sum / cell_area (m^2/m^2)
        * VolumeDens: V_sum / cell_area (m^3/m^2) - if V_sum from WSF3D exists
        * ShareDens: shared_walls_sum / cell_area (m/m^2)
    - Diversity: ALL std statistics (variation measures)

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame with building polygons and calculated metrics
    grid : GeoDataFrame
        GeoDataFrame with grid cells
    grid_id_col : str
        Column name for grid cell IDs (default: 'grid_id')
    aggregation_methods : dict or None
        Dictionary mapping metrics to their aggregation methods
        If None, uses predefined meaningful aggregations per metric

    Returns:
    --------
    GeoDataFrame
        Grid GeoDataFrame with aggregated metrics

    Raises:
    -------
    ValueError
        If grid_id_col not found after reset_index
    KeyError
        If metric columns not found in buildings

    Examples:
    ---------
    >>> grid = aggregate_metrics_to_grid(buildings, grid)
    >>> # Custom aggregations
    >>> custom_agg = {'Area': ['mean', 'sum'], 'Corners': ['mean', 'max']}
    >>> grid = aggregate_metrics_to_grid(buildings, grid, aggregation_methods=custom_agg)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Define meaningful aggregations for each metric if not provided
    if aggregation_methods is None:
        aggregation_methods = {
            # Dimension - mean, max, std only (sum temporarily included for density calculation)
            'Area': ['mean', 'max', 'std', 'sum'],  # sum -> CoverRatio, then removed
            'Perimeter': ['mean', 'max', 'std', 'sum'],  # sum -> PerimDens, then removed
            'CourtyardArea': ['mean', 'std', 'sum'],  # sum -> CourtDens, then removed
            'LongestAxisLength': ['mean', 'max', 'std'],

            # Shape - mean and std only (ratios/indices)
            'CircularCompactness': ['mean', 'std'],
            'SquareCompactness': ['mean', 'std'],
            'CompactnessWeightedAxis': ['mean', 'std'],
            'EquivalentRectangularIndex': ['mean', 'std'],
            'Rectangularity': ['mean', 'std'],
            'Convexity': ['mean', 'std'],
            'Elongation': ['mean', 'std'],
            'Squareness': ['mean', 'std'],
            'CentroidCornersMean': ['mean', 'std'],
            'CentroidCornersStd': ['mean', 'std'],

            # Additional shape metrics
            'CourtyardIndex': ['mean', 'std'],
            'FacadeRatio': ['mean', 'std'],
            'FractalDimension': ['mean', 'std'],
            'ShapeIndex': ['mean', 'std'],

            # 3D metrics (height-dependent)
            'FormFactor': ['mean', 'std'],
            'H2F_Ratio': ['mean', 'std'],
            'FloorArea': ['sum'],  # sum -> FloorARat, then removed

            # Volume density at building level (Volume / FootprintArea)
            # This allows mean and std of volume density to be computed across buildings
            'VolumeDens_bldg': ['mean', 'std'],  # Mean and std of building-level volume density

            # Complexity
            'Corners': ['mean', 'max', 'std'],

            # Spatial Distribution - Orientation needs special handling
            'Orientation': ['mean', 'std'],  # Will be replaced with circular stats

            # SharedWalls (Intensity)
            'SharedWalls': ['mean', 'max', 'std', 'sum']  # sum -> ShareDens, then removed
        }

    logger.info(f"Aggregating {len(buildings):,} buildings to {len(grid):,} grid cells...")

    try:
        # Define aggregation steps count
        total_steps = 7  # SharedWalls, centroids, grid prep, sjoin, agg, merge, density

        with tqdm(total=total_steps, desc="  Aggregating to grid", unit="step") as pbar:
            # Step 0: Calculate SharedWalls using grid-based optimization
            pbar.set_postfix_str("Calculating SharedWalls")
            buildings['SharedWalls'] = _optimized_shared_walls(
                buildings,
                grid=grid,
                grid_id_col=grid_id_col
            )
            pbar.update(1)
            # Step 1: Create building centroids for spatial join (optimized)
            pbar.set_postfix_str("Creating centroids")
            # Create temporary GeoDataFrame with centroids (avoid modifying original)
            building_centroids = gpd.GeoDataFrame(
                buildings.drop(columns='geometry'),
                geometry=buildings.centroid,
                crs=buildings.crs
            )
            pbar.update(1)

            # Step 2: Reset grid index to ensure grid_id_col exists
            pbar.set_postfix_str("Preparing grid")
            if grid_id_col not in grid.columns:
                grid = grid.reset_index(names=grid_id_col)
            pbar.update(1)

            # Step 3: Spatial join: buildings to grid (use centroids)
            pbar.set_postfix_str("Spatial join")
            bldg_in_grid = gpd.sjoin(
                building_centroids,
                grid[[grid_id_col, "geometry"]],
                predicate="within",
                how="left"
            )

            # Count buildings per grid cell
            buildings_per_cell = bldg_in_grid.groupby(grid_id_col).size()
            pbar.update(1)

            # Step 4: Compute aggregations per grid cell with metric-specific rules
            pbar.set_postfix_str("Computing aggregations")

            # Filter aggregation_methods to only include metrics that exist in buildings
            # Use set for faster lookup
            bldg_columns_set = set(bldg_in_grid.columns)
            existing_agg_methods = {
                metric: methods
                for metric, methods in aggregation_methods.items()
                if metric in bldg_columns_set
            }

            # Compute aggregations directly (geometry already excluded from building_centroids)
            grid_metrics = bldg_in_grid.groupby(grid_id_col).agg(existing_agg_methods)

            # Flatten MultiIndex columns with shorter names for shapefile compatibility (10 char limit)
            # These mappings match exactly with column_name_mapping.csv short names
            column_mapping = {
                'Area': 'Area',
                'Perimeter': 'Perim',
                'CourtyardArea': 'Court',
                'LongestAxisLength': 'Longe',  # Mean_Longe (10 chars max)
                'CircularCompactness': 'Circu',
                'Squareness': 'Squar',
                'EquivalentRectangularIndex': 'Equiv',
                'Elongation': 'Elong',
                'CompactnessWeightedAxis': 'Compa',
                'Convexity': 'Conve',
                'Rectangularity': 'Recta',
                'SquareCompactness': 'Squ_1',
                'CentroidCornersMean': 'Centr',
                'CentroidCornersStd': 'CentS',
                'Corners': 'Corne',  # Mean_Corne (10 chars max)
                'Orientation': 'Orien',  # Mean_Orien (10 chars max)
                'SharedWalls': 'Share',
                # Additional shape metrics
                'CourtyardIndex': 'CrtYd',  # Mean_CrtYd (10 chars max)
                'FacadeRatio': 'Facad',  # Mean_Facad (10 chars max)
                'FractalDimension': 'Fract',  # Mean_Fract (10 chars max)
                'ShapeIndex': 'ShpId',  # Mean_ShpId (10 chars max)
                # 3D metrics (height-dependent)
                'FormFactor': 'FormF',  # Mean_FormF (10 chars exactly) - truncated from FormFa
                'H2F_Ratio': 'H2F',  # Mean_H2F (8 chars)
                'FloorArea': 'FlrAr',  # Sum_FlrAr -> FloorARat (10 chars max)
                # Volume density (building-level)
                'VolumeDens_bldg': 'VolDe'  # Mean_VolDe, Std_VolDen (10 chars exactly) - truncated from VolDens
            }

            grid_metrics.columns = [
                f"{agg.capitalize()}_{column_mapping.get(col, col)}"
                for col, agg in grid_metrics.columns
            ]
            grid_metrics = grid_metrics.reset_index()

            # Special handling for Orientation using circular statistics, since
            # arithmetic mean/std are meaningless for angular (0-180 degree) data
            from scipy.stats import circmean, circstd

            if 'Orientation' in bldg_columns_set:

                if 'Orientation' not in bldg_in_grid.columns:
                    logger.warning("Orientation column not found in bldg_in_grid after spatial join!")
                    logger.warning(f"Available columns: {bldg_in_grid.columns.tolist()}")
                else:
                    logger.info(f"  Orientation column found. Non-null values: {bldg_in_grid['Orientation'].notna().sum()}/{len(bldg_in_grid)}")

                # Compute circular statistics efficiently
                def circular_stats(x):
                    if len(x) == 0 or x.isna().all():
                        return pd.Series({'circular_mean': np.nan, 'circular_std': np.nan})
                    # Filter valid angles (should be 0-180 for building orientation)
                    valid_angles = x.dropna()
                    if len(valid_angles) == 0:
                        return pd.Series({'circular_mean': np.nan, 'circular_std': np.nan})
                    return pd.Series({
                        'circular_mean': circmean(valid_angles, high=180, low=0),
                        'circular_std': circstd(valid_angles, high=180, low=0)
                    })

                # Only calculate if Orientation column exists
                if 'Orientation' in bldg_in_grid.columns:
                    try:
                        orientation_stats = bldg_in_grid.groupby(grid_id_col)['Orientation'].apply(circular_stats)

                        # Handle case where result might not be MultiIndex (when all grids have same stats)
                        if not isinstance(orientation_stats.index, pd.MultiIndex):
                            # Result is already unstacked or single-level
                            logger.warning(f"orientation_stats is not MultiIndex (type: {type(orientation_stats.index)}), converting...")
                            # If it's a Series of Series, convert to DataFrame
                            if len(orientation_stats) > 0 and isinstance(orientation_stats.iloc[0], pd.Series):
                                orientation_stats = pd.DataFrame(list(orientation_stats.values), index=orientation_stats.index)
                            else:
                                # It's already in the wrong format, create empty DataFrame
                                logger.error(f"Unexpected orientation_stats format. Length: {len(orientation_stats)}, Type of first element: {type(orientation_stats.iloc[0]) if len(orientation_stats) > 0 else 'N/A'}")
                                orientation_stats = pd.DataFrame({'circular_mean': [], 'circular_std': []})
                        else:
                            orientation_stats = orientation_stats.unstack()

                        logger.info(f"  Circular stats calculated. Non-null circular_mean: {orientation_stats['circular_mean'].notna().sum()}/{len(orientation_stats)}")
                    except Exception as e:
                        logger.error(f"Error calculating circular stats: {e}")
                        logger.error(f"  orientation_stats type: {type(orientation_stats)}")
                        logger.error(f"  orientation_stats index type: {type(orientation_stats.index)}")
                        # Skip circular stats if error
                        orientation_stats = None

                    # Replace regular mean/std with circular statistics
                    # Use .values to ignore index and perform positional assignment,
                    # since orientation_stats and grid_metrics can have different row order
                    if orientation_stats is not None:
                        if 'Mean_Orien' in grid_metrics.columns:
                            grid_metrics['Mean_Orien'] = orientation_stats['circular_mean'].values
                            logger.info(f"  Replaced Mean_Orien with circular_mean")
                        else:
                            logger.warning(f"Mean_Orien column not found in grid_metrics. Available: {[c for c in grid_metrics.columns if 'Orien' in c]}")
                        if 'Std_Orien' in grid_metrics.columns:
                            grid_metrics['Std_Orien'] = orientation_stats['circular_std'].values
                            logger.info(f"  Replaced Std_Orien with circular_std")
                        else:
                            logger.warning(f"Std_Orien column not found in grid_metrics. Available: {[c for c in grid_metrics.columns if 'Orien' in c]}")
                    else:
                        logger.warning("Skipping circular stats replacement due to earlier error")
                else:
                    logger.error("Cannot calculate circular stats: Orientation column missing from bldg_in_grid!")

            # Add building count
            grid_metrics['BldgCount'] = buildings_per_cell.values
            pbar.update(1)

            # Step 5: Merge metrics into grid
            pbar.set_postfix_str("Merging results")
            # Use copy=False to avoid unnecessary data duplication
            grid = grid.merge(grid_metrics, on=grid_id_col, how="left", copy=False)

            # Fill NaN values with 0 for count
            grid['BldgCount'] = grid['BldgCount'].fillna(0).astype(int)

            # Step 6: Calculate density metrics (Intensity type)
            # All grid cells have the same area, so replace sum metrics with density metrics
            pbar.set_postfix_str("Calculating density metrics")

            # Check if Area_m2 column exists (grid cell area)
            if 'Area_m2' in grid.columns:
                cell_area = grid['Area_m2']

                # Building density: building_count / cell_area
                grid['BldgDens'] = grid['BldgCount'] / cell_area

                # Coverage ratio: building_area_sum / cell_area
                if 'Sum_Area' in grid.columns:
                    grid['CoverRatio'] = grid['Sum_Area'] / cell_area
                    # Open space ratio: 1 - coverage_ratio
                    grid['OpenRatio'] = 1.0 - grid['CoverRatio']
                    # Remove Sum_Area (redundant)
                    grid = grid.drop(columns=['Sum_Area'])

                # Perimeter density: perimeter_sum / cell_area
                if 'Sum_Perim' in grid.columns:
                    grid['PerimDens'] = grid['Sum_Perim'] / cell_area
                    # Remove Sum_Perim (redundant)
                    grid = grid.drop(columns=['Sum_Perim'])

                # Courtyard density: courtyard_area_sum / cell_area
                if 'Sum_Court' in grid.columns:
                    grid['CourtDens'] = grid['Sum_Court'] / cell_area
                    # Remove Sum_Court (redundant)
                    grid = grid.drop(columns=['Sum_Court'])

                # Note: VolumeDens (Mean_VolDens, Std_VolDens) is calculated from
                # building-level VolumeDens_bldg during aggregation, not from sum/area

                # SharedWalls density: sum_shared_walls / cell_area (keep max/mean/std)
                if 'Sum_Share' in grid.columns:
                    grid['ShareDens'] = grid['Sum_Share'] / cell_area
                    # Remove Sum_Share (redundant)
                    grid = grid.drop(columns=['Sum_Share'])

                # Floor area ratio: sum_floor_area / cell_area (if FloorArea exists)
                if 'Sum_FlrAr' in grid.columns:
                    grid['FloorARat'] = grid['Sum_FlrAr'] / cell_area
                    # Remove Sum_FlrAr (redundant)
                    grid = grid.drop(columns=['Sum_FlrAr'])

            pbar.update(1)

        return grid

    except Exception as e:
        logger.error(f"Error aggregating metrics to grid: {str(e)}")
        raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def filter_grid_with_values(grid, column_name='Mean_Area'):
    """
    Filter grid cells that have valid metric values.

    Parameters:
    -----------
    grid : GeoDataFrame
        Grid GeoDataFrame with metrics
    column_name : str
        Column name to check for non-null values (default: 'Mean_Area')

    Returns:
    --------
    GeoDataFrame
        Filtered grid with only cells containing valid values

    Examples:
    ---------
    >>> grid_filtered = filter_grid_with_values(grid, 'Mean_Area')
    >>> grid_high_density = filter_grid_with_values(grid, 'BuildingCount')
    """
    if column_name not in grid.columns:
        return grid

    filtered = grid[grid[column_name].notna()].copy()
    return filtered


def get_metric_statistics(buildings, metric_columns=None):
    """
    Calculate summary statistics for building metrics.

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame with calculated metrics
    metric_columns : list or None
        List of metric column names to analyze (default: all metric columns)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with statistics (mean, median, std, min, max) for each metric

    Examples:
    ---------
    >>> stats = get_metric_statistics(buildings)
    >>> print(stats)
    """
    if metric_columns is None:
        metric_columns = [
            'Area', 'Perimeter', 'CourtyardArea', 'CircularCompactness', 'Corners', 'Squareness',
            'EquivalentRectangularIndex', 'Elongation', 'CentroidCornersMean', 'CentroidCornersStd',
            'Orientation', 'LongestAxisLength', 'CompactnessWeightedAxis', 'Convexity',
            'Rectangularity', 'SquareCompactness', 'SharedWalls'
        ]

    # Filter to existing columns (use set for faster lookup)
    building_cols = set(buildings.columns)
    existing_cols = [col for col in metric_columns if col in building_cols]

    # Select specific percentiles to reduce computation
    stats = buildings[existing_cols].describe(percentiles=[0.5]).T
    stats = stats[['mean', '50%', 'std', 'min', 'max']]
    stats.columns = ['Mean', 'Median', 'Std', 'Min', 'Max']

    return stats


def export_metrics_to_csv(grid, output_path, metric_prefix='Mean_'):
    """
    Export grid metrics to CSV file.

    Parameters:
    -----------
    grid : GeoDataFrame
        Grid with aggregated metrics
    output_path : str or Path
        Output CSV file path
    metric_prefix : str
        Prefix to filter metric columns (default: 'Mean_')

    Examples:
    ---------
    >>> export_metrics_to_csv(grid, 'grid_metrics.csv')
    >>> export_metrics_to_csv(grid, 'grid_metrics.csv', metric_prefix='Sum_')
    """
    # Select metric columns (exclude geometry)
    cols_to_export = [col for col in grid.columns if col.startswith(metric_prefix) and col != 'geometry']

    # Include index if it has a name
    if grid.index.name:
        cols_to_export.insert(0, grid.index.name)

    # Export to CSV (avoid creating intermediate dataframe)
    grid[cols_to_export].to_csv(output_path, index=False)
    logger.info(f"Exported {len(cols_to_export)} metric columns to {output_path}")


def generate_column_mapping_with_categories(output_path='column_name_mapping.csv'):
    """
    Generate column name mapping CSV with comprehensive category classification.

    This function creates a comprehensive mapping of all grid columns following
    momepy's 6-category framework (Grid System + 5 morphology categories):

    - Short names (shapefile-compatible, <=10 chars)
    - Full descriptive names
    - Data sources (Momepy, WSF3D, Copernicus, WorldPop, etc.)
    - Categories:
        * Grid System: Grid identifiers and metadata
        * Dimension: Physical measurements (area, perimeter, height, volume) - mean/max only
        * Shape: Form ratios and indices - mean only
        * Spatial Distribution: Orientation and positioning
        * Intensity: ALL density/coverage measures (buildings, land cover, population, sum statistics)
        * Diversity: ALL variation measures (std statistics, Shannon entropy)
    - Statistic types (Mean, Max, Sum, Std, Count, Class, Fraction, etc.)
    - Descriptions

    Parameters:
    -----------
    output_path : str or Path
        Output CSV file path (default: 'column_name_mapping.csv')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with column mappings

    Examples:
    ---------
    >>> mapping_df = generate_column_mapping_with_categories()
    >>> mapping_df = generate_column_mapping_with_categories('my_mapping.csv')
    """
    import pandas as pd

    # Define mappings with categories
    mappings = []

    # ============================================================================
    # GRID SYSTEM COLUMNS
    # ============================================================================
    mappings.extend([
        ('GridID', 'Grid Cell ID', 'Generated', 'Grid System', 'Identifier', 'Unique identifier for each hexagonal grid cell'),
        ('City', 'City Name', 'Generated', 'Grid System', 'Metadata', 'Name of the city or area'),
        ('Area_m2', 'Grid Cell Area', 'Generated', 'Grid System', 'Geometry', 'Area of hexagonal grid cell (square meters)'),
        ('Center_X', 'Grid Center X', 'Generated', 'Grid System', 'Geometry', 'X coordinate of grid cell center'),
        ('Center_Y', 'Grid Center Y', 'Generated', 'Grid System', 'Geometry', 'Y coordinate of grid cell center'),
        ('BldgCount', 'Building Count', 'Momepy', 'Intensity', 'Count', 'Number of buildings in grid cell'),
    ])

    # ============================================================================
    # DIMENSION METRICS - Physical measurements (area, perimeter, height, volume)
    # ============================================================================
    # Building footprint dimensions (from Momepy)
    dimension_metrics = [
        # Building Area (mean and max only - sum goes to Intensity, std goes to Diversity)
        ('Mean_Area', 'Mean Building Area', 'Dimension', 'Mean', 'Mean area of buildings in grid cell (square meters)'),
        ('Max_Area', 'Max Building Area', 'Dimension', 'Max', 'Maximum building area in grid cell (square meters)'),
        ('Std_Area', 'Std Building Area', 'Diversity', 'Std', 'Standard deviation of building areas (diversity indicator)'),

        # Building Perimeter
        ('Mean_Perim', 'Mean Building Perimeter', 'Dimension', 'Mean', 'Mean perimeter of buildings (meters)'),
        ('Max_Perim', 'Max Building Perimeter', 'Dimension', 'Max', 'Maximum building perimeter (meters)'),
        ('Std_Perim', 'Std Building Perimeter', 'Diversity', 'Std', 'Standard deviation of building perimeters (diversity)'),

        # Courtyard Area
        ('Mean_Court', 'Mean Courtyard Area', 'Dimension', 'Mean', 'Mean courtyard area within buildings (square meters)'),
        ('Std_Court', 'Std Courtyard Area', 'Diversity', 'Std', 'Standard deviation of courtyard areas (diversity)'),

        # Longest Axis Length - use 'Longe' abbreviation to fit 10-char limit (truncated to 10 chars in shp)
        ('Mean_Longe', 'Mean Longest Axis Length', 'Dimension', 'Mean', 'Mean longest axis length of buildings (meters)'),
        ('Max_Longe', 'Max Longest Axis Length', 'Dimension', 'Max', 'Maximum longest axis length (meters)'),
        ('Std_Longe', 'Std Longest Axis Length', 'Diversity', 'Std', 'Standard deviation of longest axis lengths (diversity)'),

        # Building Height (from WSF3D)
        ('H_mean', 'Mean Building Height', 'Dimension', 'Mean', 'Mean building height from WSF3D (meters)'),
        ('H_max', 'Max Building Height', 'Dimension', 'Max', 'Maximum building height from WSF3D (meters)'),
        ('H_std', 'Std Building Height', 'Diversity', 'Std', 'Standard deviation of building heights (diversity)'),

        # Building Volume (from existing grid shp - inherited columns, not calculated by this script)
        ('V_mean', 'Mean Building Volume', 'Dimension', 'Mean', 'Mean building volume inherited from previous grid (cubic meters)'),
        ('V_max', 'Max Building Volume', 'Dimension', 'Max', 'Maximum building volume inherited from previous grid (cubic meters)'),
        ('V_std', 'Std Building Volume', 'Diversity', 'Std', 'Standard deviation of building volumes inherited from previous grid (diversity)'),
        ('V_sum', 'Sum Building Volume', 'Intensity', 'Sum', 'Total building volume inherited from previous grid (cubic meters)'),
        # Note: Volume_90m is internal only, used to calculate VolumeDens statistics
    ]

    # ============================================================================
    # SHAPE METRICS - Form ratios and indices
    # ============================================================================
    shape_metrics = [
        # Circular Compactness
        ('Mean_Circu', 'Mean Circular Compactness', 'Shape', 'Mean', 'Mean circular compactness (ratio of building area to circumscribed circle)'),
        ('Std_Circu', 'Std Circular Compactness', 'Diversity', 'Std', 'Standard deviation of circular compactness (diversity)'),

        # Squareness
        ('Mean_Squar', 'Mean Squareness', 'Shape', 'Mean', 'Mean squareness index (similarity to square shape)'),
        ('Std_Squar', 'Std Squareness', 'Diversity', 'Std', 'Standard deviation of squareness (diversity)'),

        # Equivalent Rectangular Index
        ('Mean_Equiv', 'Mean Equivalent Rectangular Index', 'Shape', 'Mean', 'Mean equivalent rectangular index (ERI)'),
        ('Std_Equiv', 'Std Equivalent Rectangular Index', 'Diversity', 'Std', 'Standard deviation of ERI (diversity)'),

        # Elongation
        ('Mean_Elong', 'Mean Elongation', 'Shape', 'Mean', 'Mean elongation ratio of buildings'),
        ('Std_Elong', 'Std Elongation', 'Diversity', 'Std', 'Standard deviation of elongation (diversity)'),

        # Compactness Weighted Axis
        ('Mean_Compa', 'Mean Compactness Weighted Axis', 'Shape', 'Mean', 'Mean compactness weighted by axis length'),
        ('Std_Compa', 'Std Compactness Weighted Axis', 'Diversity', 'Std', 'Standard deviation of compactness weighted axis (diversity)'),

        # Convexity
        ('Mean_Conve', 'Mean Convexity', 'Shape', 'Mean', 'Mean convexity (ratio of convex hull area to building area)'),
        ('Std_Conve', 'Std Convexity', 'Diversity', 'Std', 'Standard deviation of convexity (diversity)'),

        # Rectangularity
        ('Mean_Recta', 'Mean Rectangularity', 'Shape', 'Mean', 'Mean rectangularity (similarity to rectangular shape)'),
        ('Std_Recta', 'Std Rectangularity', 'Diversity', 'Std', 'Standard deviation of rectangularity (diversity)'),

        # Square Compactness
        ('Mean_Squ_1', 'Mean Square Compactness', 'Shape', 'Mean', 'Mean square compactness index'),
        ('Std_Squ_1', 'Std Square Compactness', 'Diversity', 'Std', 'Standard deviation of square compactness (diversity)'),

        # Centroid-Corners Distance
        ('Mean_Centr', 'Mean Centroid-Corners Distance (Mean)', 'Shape', 'Mean', 'Mean of distances from building centroid to corners'),
        ('Std_Centr', 'Std Centroid-Corners Distance (Mean)', 'Diversity', 'Std', 'Standard deviation of mean centroid-corners distances (diversity)'),

        ('Mean_CentS', 'Mean Centroid-Corners Distance (Std)', 'Shape', 'Mean', 'Mean of standard deviations of centroid-corners distances'),
        ('Std_CentS', 'Std Centroid-Corners Distance (Std)', 'Diversity', 'Std', 'Standard deviation of std centroid-corners distances (diversity)'),

        # Additional shape metrics
        ('Mean_CrtYd', 'Mean Courtyard Index', 'Shape', 'Mean', 'Mean courtyard index (ratio of courtyard area to building area)'),
        ('Std_CrtYd', 'Std Courtyard Index', 'Diversity', 'Std', 'Standard deviation of courtyard index (diversity)'),

        ('Mean_Facad', 'Mean Facade Ratio', 'Shape', 'Mean', 'Mean facade ratio (perimeter to area ratio)'),
        ('Std_Facad', 'Std Facade Ratio', 'Diversity', 'Std', 'Standard deviation of facade ratio (diversity)'),

        ('Mean_Fract', 'Mean Fractal Dimension', 'Shape', 'Mean', 'Mean fractal dimension (complexity measure based on area and perimeter)'),
        ('Std_Fract', 'Std Fractal Dimension', 'Diversity', 'Std', 'Standard deviation of fractal dimension (diversity)'),

        ('Mean_ShpId', 'Mean Shape Index', 'Shape', 'Mean', 'Mean shape index (ratio of area to area of circle with same perimeter)'),
        ('Std_ShpId', 'Std Shape Index', 'Diversity', 'Std', 'Standard deviation of shape index (diversity)'),

        # 3D metrics (height-dependent, from WSF3D raster)
        ('Mean_FormF', 'Mean Form Factor', 'Shape', 'Mean', 'Mean form factor (surface area to volume ratio, 3D compactness indicator)'),
        ('Std_FormF', 'Std Form Factor', 'Diversity', 'Std', 'Standard deviation of form factor (diversity)'),

        ('Mean_H2F', 'Mean Height-to-Footprint Ratio', 'Shape', 'Mean', 'Mean height-to-footprint ratio (height / sqrt(footprint_area))'),
        ('Std_H2F', 'Std Height-to-Footprint Ratio', 'Diversity', 'Std', 'Standard deviation of height-to-footprint ratio (diversity)'),
    ]

    # ============================================================================
    # COMPLEXITY METRICS - Building geometry complexity
    # ============================================================================
    complexity_metrics = [
        ('Mean_Corne', 'Mean Corners', 'Shape', 'Mean', 'Mean number of corners per building'),
        ('Max_Corne', 'Max Corners', 'Shape', 'Max', 'Maximum number of corners in grid cell'),
        ('Std_Corne', 'Std Corners', 'Diversity', 'Std', 'Standard deviation of corner counts (diversity)'),
    ]

    # ============================================================================
    # SPATIAL DISTRIBUTION METRICS - Orientation and positioning
    # ============================================================================
    spatial_metrics = [
        # Building Orientation
        ('Mean_Orien', 'Mean Orientation', 'Spatial Distribution', 'Mean', 'Mean building orientation angle (degrees, computed using circular mean)'),
        ('Std_Orien', 'Std Orientation', 'Diversity', 'Std', 'Standard deviation of building orientation (diversity, computed using circular std)'),

        # Spatial Distribution Metrics (building alignment / interbuilding / neighbor graph)
        ('Mean_Align', 'Mean Building Alignment', 'Spatial Distribution', 'Mean', 'Mean deviation of building orientation from adjacent buildings (degrees)'),
        ('Std_Align', 'Std Building Alignment', 'Diversity', 'Std', 'Standard deviation of building alignment deviations (diversity)'),

        ('Mean_IBDis', 'Mean Interbuilding Distance', 'Spatial Distribution', 'Mean', 'Mean distance between adjacent buildings within neighborhood (meters)'),
        ('Std_IBDis', 'Std Interbuilding Distance', 'Diversity', 'Std', 'Standard deviation of interbuilding distances (diversity)'),

        ('Mean_NbDis', 'Mean Neighbor Distance', 'Spatial Distribution', 'Mean', 'Mean distance to adjacent buildings based on spatial graph (meters)'),
        ('Std_NbDis', 'Std Neighbor Distance', 'Diversity', 'Std', 'Standard deviation of neighbor distances (diversity)'),

        ('Mean_Neigh', 'Mean Neighbor Count', 'Spatial Distribution', 'Mean', 'Mean number of neighbors captured by spatial graph'),
        ('Max_Neigh', 'Max Neighbor Count', 'Spatial Distribution', 'Max', 'Maximum number of neighbors in grid cell'),
        ('Std_Neigh', 'Std Neighbor Count', 'Diversity', 'Std', 'Standard deviation of neighbor counts (diversity)'),

        # BuildingAdjacency, using a local contiguity graph for performance
        ('Mean_BAdj', 'Mean Building Adjacency', 'Spatial Distribution', 'Mean', 'Mean ratio of joined built-up structures to buildings in neighborhood (0-1)'),
        ('Std_BAdj', 'Std Building Adjacency', 'Diversity', 'Std', 'Standard deviation of building adjacency ratios (diversity)'),
    ]

    # ============================================================================
    # INTENSITY METRICS - ALL density/coverage/ratio measures
    # ============================================================================
    # Since all grid cells have the same area, sum metrics are converted to
    # density metrics to avoid redundancy and provide more meaningful indicators.
    intensity_metrics = [
        # ---- Building Intensity Measures (Density-based) ----
        # Building density: building_count / cell_area
        ('BldgDens', 'Building Density', 'Intensity', 'Density', 'Number of buildings per unit area (count/m²)'),

        # Coverage ratio: building_area_sum / cell_area
        ('CoverRatio', 'Building Coverage Ratio', 'Intensity', 'Ratio', 'Ratio of building footprint area to grid cell area (0-1)'),

        # Open space ratio: 1 - coverage_ratio
        ('OpenRatio', 'Open Space Ratio', 'Intensity', 'Ratio', 'Ratio of open space to grid cell area (0-1)'),

        # Perimeter density: perimeter_sum / cell_area
        ('PerimDens', 'Perimeter Density', 'Intensity', 'Density', 'Total building perimeter per unit area (m/m²)'),

        # Courtyard density: courtyard_area_sum / cell_area
        ('CourtDens', 'Courtyard Density', 'Intensity', 'Density', 'Total courtyard area per unit area (m²/m²)'),

        # Volume density statistics (from WSF3D ~90m raster, aggregated from building-level densities)
        ('Mean_VolDe', 'Mean Volume Density', 'Intensity', 'Mean', 'Mean building-level volume density from WSF3D ~90m raster (m³/m² footprint)'),
        ('Std_VolDe', 'Std Volume Density', 'Diversity', 'Std', 'Standard deviation of building-level volume densities (diversity indicator)'),

        # Shared walls density: shared_walls_sum / cell_area
        ('ShareDens', 'Shared Walls Density', 'Intensity', 'Density', 'Total shared wall length per unit area (m/sq.m)'),

        # Shared Walls statistics (mean, max, std)
        ('Mean_Share', 'Mean Shared Walls', 'Intensity', 'Mean', 'Mean length of shared walls per building (meters)'),
        ('Max_Share', 'Max Shared Walls', 'Intensity', 'Max', 'Maximum shared wall length (meters)'),
        ('Std_Share', 'Std Shared Walls', 'Diversity', 'Std', 'Standard deviation of shared wall lengths (diversity)'),

        # Floor Area Ratio (FAR) - 3D intensity metric
        ('FloorARat', 'Floor Area Ratio', 'Intensity', 'Ratio', 'Floor area ratio (total floor area / grid cell area, indicates development intensity)'),

        # Building Height sum (from WSF3D) - kept as sum for reference
        ('H_sum', 'Sum Building Height', 'Intensity', 'Sum', 'Total building height from WSF3D (meters, cumulative)'),

        # ---- Land Cover Intensity Measures (Discrete Classes) ----
        # Basic land cover types
        ('LC_Shrub', 'Shrubs Cover', 'Intensity', 'Class', 'Percentage of shrubland in grid cell (0-100%)'),
        ('LC_Herb', 'Herbaceous Cover', 'Intensity', 'Class', 'Percentage of herbaceous vegetation in grid cell (0-100%)'),
        ('LC_Crop', 'Cropland Cover', 'Intensity', 'Class', 'Percentage of cropland in grid cell (0-100%)'),
        ('LC_Built', 'Built-up Cover', 'Intensity', 'Class', 'Percentage of built-up area in grid cell (0-100%)'),
        ('LC_Bare', 'Bare/Sparse Vegetation Cover', 'Intensity', 'Class', 'Percentage of bare/sparse vegetation in grid cell (0-100%)'),
        ('LC_Snow', 'Snow and Ice Cover', 'Intensity', 'Class', 'Percentage of snow/ice in grid cell (0-100%)'),
        ('LC_Water', 'Permanent Water Cover', 'Intensity', 'Class', 'Percentage of permanent water in grid cell (0-100%)'),
        ('LC_Wetlnd', 'Wetland Cover', 'Intensity', 'Class', 'Percentage of wetland in grid cell (0-100%)'),
        ('LC_Moss', 'Moss and Lichen Cover', 'Intensity', 'Class', 'Percentage of moss/lichen in grid cell (0-100%)'),

        # Forest types - Closed forest (FC)
        ('LC_FCEvNd', 'Closed Forest Evergreen Needleleaf', 'Intensity', 'Class', 'Percentage of closed evergreen needleleaf forest (0-100%)'),
        ('LC_FCEvBr', 'Closed Forest Evergreen Broadleaf', 'Intensity', 'Class', 'Percentage of closed evergreen broadleaf forest (0-100%)'),
        ('LC_FCDcNd', 'Closed Forest Deciduous Needleleaf', 'Intensity', 'Class', 'Percentage of closed deciduous needleleaf forest (0-100%)'),
        ('LC_FCDcBr', 'Closed Forest Deciduous Broadleaf', 'Intensity', 'Class', 'Percentage of closed deciduous broadleaf forest (0-100%)'),
        ('LC_FCMix', 'Closed Forest Mixed', 'Intensity', 'Class', 'Percentage of closed mixed forest (0-100%)'),
        ('LC_FCUnd', 'Closed Forest Unknown Type', 'Intensity', 'Class', 'Percentage of closed forest unknown type (0-100%)'),

        # Forest types - Open forest (FO)
        ('LC_FOEvNd', 'Open Forest Evergreen Needleleaf', 'Intensity', 'Class', 'Percentage of open evergreen needleleaf forest (0-100%)'),
        ('LC_FOEvBr', 'Open Forest Evergreen Broadleaf', 'Intensity', 'Class', 'Percentage of open evergreen broadleaf forest (0-100%)'),
        ('LC_FODcNd', 'Open Forest Deciduous Needleleaf', 'Intensity', 'Class', 'Percentage of open deciduous needleleaf forest (0-100%)'),
        ('LC_FODcBr', 'Open Forest Deciduous Broadleaf', 'Intensity', 'Class', 'Percentage of open deciduous broadleaf forest (0-100%)'),
        ('LC_FOMix', 'Open Forest Mixed', 'Intensity', 'Class', 'Percentage of open mixed forest (0-100%)'),
        ('LC_FOUnd', 'Open Forest Unknown Type', 'Intensity', 'Class', 'Percentage of open forest unknown type (0-100%)'),

        # Ocean
        ('LC_Ocean', 'Ocean', 'Intensity', 'Class', 'Percentage of ocean in grid cell (0-100%)'),

        # ---- Land Cover Intensity Measures (Continuous Fractions) ----
        ('LCF_Bare', 'Bare/Sparse Vegetation Fraction', 'Intensity', 'Fraction', 'Continuous bare/sparse vegetation cover fraction (0-100%)'),
        ('LCF_Crop', 'Cropland Fraction', 'Intensity', 'Fraction', 'Continuous cropland cover fraction (0-100%)'),
        ('LCF_Grass', 'Grassland Fraction', 'Intensity', 'Fraction', 'Continuous grassland cover fraction (0-100%)'),
        ('LCF_Moss', 'Moss and Lichen Fraction', 'Intensity', 'Fraction', 'Continuous moss/lichen cover fraction (0-100%)'),
        ('LCF_Water', 'Permanent Water Fraction', 'Intensity', 'Fraction', 'Continuous permanent water cover fraction (0-100%)'),
        ('LCF_SeaWtr', 'Seasonal Water Fraction', 'Intensity', 'Fraction', 'Continuous seasonal water cover fraction (0-100%)'),
        ('LCF_Shrub', 'Shrubland Fraction', 'Intensity', 'Fraction', 'Continuous shrubland cover fraction (0-100%)'),
        ('LCF_Snow', 'Snow and Ice Fraction', 'Intensity', 'Fraction', 'Continuous snow/ice cover fraction (0-100%)'),
        ('LCF_Tree', 'Tree Cover Fraction', 'Intensity', 'Fraction', 'Continuous tree cover fraction (0-100%)'),
        ('LCF_Built', 'Built-up Cover Fraction', 'Intensity', 'Fraction', 'Continuous built-up cover fraction (0-100%)'),

        # ---- Population Intensity Measures ----
        ('Pop2022', 'Population 2022', 'Intensity', 'Count', 'Population count from WorldPop 2022 (people per grid cell)'),
    ]

    # ============================================================================
    # DIVERSITY METRICS - ALL variation measures (std statistics, Shannon entropy)
    # ============================================================================
    diversity_metrics = [
        ('LC_Shannon', 'Land Cover Shannon Diversity', 'Diversity', 'Index', 'Shannon entropy index of land cover diversity (higher = more diverse)'),
    ]

    # ============================================================================
    # STREET NETWORK METRICS
    # ============================================================================
    street_network_metrics = [
        # ---- Intensity Metrics (Network Density) ----
        ('StreetDens', 'Street Density', 'Intensity', 'Density', 'Total street length per unit area (m/m²)'),
        ('NodeDens', 'Node Density', 'Intensity', 'Density', 'Number of network nodes per unit area (count/m²)'),
        ('StreetsPN', 'Streets Per Node', 'Intensity', 'Ratio', 'Average number of street segments per node'),
        ('SegmentDen', 'Segment Density', 'Intensity', 'Density', 'Number of street segments per unit area (count/m²)'),
        ('PlotDens', 'Plot Density', 'Intensity', 'Density', 'Number of plots (polygonized areas) per unit area (count/m²)'),
        ('BldgDensN', 'Building Density (Network)', 'Intensity', 'Density', 'Building footprint area per unit area from network analysis (ratio)'),
        ('PlotFrag', 'Plot Fragmentation', 'Intensity', 'Ratio', 'Ratio of plot density to building density (fragmentation indicator)'),

        # ---- Connectivity Metrics (Network Topology) ----
        ('DeadendDen', 'Dead End Density', 'Connectivity', 'Density', 'Number of dead ends (degree-1 nodes) per unit area (count/m²)'),
        ('InterDens', 'Intersection Density', 'Connectivity', 'Density', 'Number of intersections (degree ≥ 3) per unit area (count/m²)'),
        ('Ratio3way', '3-Way Intersection Ratio', 'Connectivity', 'Ratio', 'Ratio of 3-way intersections to total intersections'),
        ('Ratio4way', '4-Way Intersection Ratio', 'Connectivity', 'Ratio', 'Ratio of 4-way intersections to total intersections'),
        ('CuldesacR', 'Cul-de-sac Ratio', 'Connectivity', 'Ratio', 'Ratio of dead ends to intersections'),

        # ---- Dimension Metrics ----
        ('StrWidth', 'Street Width', 'Dimension', 'Mean', 'Mean street width measured perpendicular to street centerline (meters)'),
        ('StrOpen', 'Street Openness', 'Dimension', 'Mean', 'Mean street openness ratio (proportion of ticks not reaching buildings)'),

        # ---- Shape Metrics ----
        ('StrLinM', 'Street Linearity Mean', 'Shape', 'Mean', 'Mean street linearity (ratio of euclidean distance to actual length)'),

        # ---- Spatial Distribution Metrics ----
        ('StrOrientM', 'Street Orientation Mean', 'Spatial Distribution', 'Mean', 'Mean street orientation angle (degrees, computed using circular mean)'),
        ('BldgStrAM', 'Building-Street Alignment Mean', 'Spatial Distribution', 'Mean', 'Mean deviation of building orientation from nearest street orientation (degrees)'),

        # ---- Diversity Metrics ----
        ('StrWidthD', 'Street Width Deviation', 'Diversity', 'Std', 'Mean standard deviation of street width (diversity indicator)'),
        ('StrLinS', 'Street Linearity Std', 'Diversity', 'Std', 'Standard deviation of street linearity (diversity)'),
        ('StrOrientS', 'Street Orientation Std', 'Diversity', 'Std', 'Standard deviation of street orientation (diversity, computed using circular std)'),
        ('BldgStrAS', 'Building-Street Alignment Std', 'Diversity', 'Std', 'Standard deviation of building-street alignment (diversity)'),
    ]

    # ============================================================================
    # COMBINE ALL METRICS
    # ============================================================================
    all_metrics = (
        dimension_metrics + shape_metrics + complexity_metrics +
        spatial_metrics + intensity_metrics + diversity_metrics +
        street_network_metrics
    )

    # Add all metrics to mappings with automatic data source detection
    for short_name, full_name, category, stat_type, description in all_metrics:
        # Determine data source based on column name pattern
        if short_name.startswith(('H_', 'V_')) or 'VolDens' in short_name:
            # WSF3D building height and volume (including VolumeDens derivatives)
            source = 'WSF3D'
        elif short_name == 'LC_Shannon':
            # Shannon entropy is derived from LC columns but computed by our code
            source = 'Momepy'
        elif short_name.startswith(('LC_', 'LCF_')):
            # Copernicus land cover (discrete classes and continuous fractions)
            source = 'Copernicus'
        elif short_name == 'Pop2022':
            # WorldPop population
            source = 'WorldPop'
        else:
            # Default to Momepy for building morphology metrics
            source = 'Momepy'

        mappings.append((
            short_name,
            full_name,
            source,
            category,
            stat_type,
            description
        ))

    # ============================================================================
    # CONNECTIVITY METRICS
    # ============================================================================
    try:
        from .connectivity import get_connectivity_column_mappings

        connectivity_mappings = get_connectivity_column_mappings()
        mappings.extend(connectivity_mappings)
        logger.info(f"  Added {len(connectivity_mappings)} connectivity metric mappings")
    except ImportError as e:
        logger.warning(f"  connectivity module not available: {e}")
    except Exception as e:
        logger.warning(f"  Failed to load connectivity mappings: {e}")

    # Create DataFrame
    df = pd.DataFrame(
        mappings,
        columns=['Short_Name', 'Full_Name', 'Data_Source', 'Momepy_Category', 'Statistic_Type', 'Description']
    )

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Generated column mapping with {len(df)} entries: {output_path}")

    return df
