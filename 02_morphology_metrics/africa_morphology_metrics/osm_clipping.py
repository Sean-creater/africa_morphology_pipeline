"""
Highway geometry clipping to an area of interest (AOI).

Clipping a country-scale highway network (often 1M+ segments) down to a
single city boundary is dominated by the cost of the spatial join and
geometric intersection, so this module provides both a single-threaded
implementation (used for typical city-sized inputs) and a tiled,
thread-parallel implementation that is switched to automatically once the
input is large enough that tiling pays for its own overhead.
"""

from typing import List, Optional
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Above this many input highway segments, the tiled thread-parallel
# implementation is used instead of a single sjoin, since tiling and
# scheduling overhead only pays off once the sjoin itself would be the
# bottleneck.
PARALLEL_CLIP_THRESHOLD = 500_000


def clip_highways_to_aoi(
    highways_gdf: gpd.GeoDataFrame,
    aoi_geometry,
    buffer_meters: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Clip highway geometries to an AOI boundary, using the spatial index to
    filter candidates before the more expensive intersection step.

    Args:
        highways_gdf: GeoDataFrame of highways.
        aoi_geometry: Shapely geometry or GeoDataFrame.
        buffer_meters: Buffer the AOI by this many meters before clipping.

    Returns:
        Clipped GeoDataFrame.
    """
    # Handle both a bare shapely geometry and a GeoDataFrame as input,
    # normalizing to a single AOI geometry plus its CRS.
    if isinstance(aoi_geometry, gpd.GeoDataFrame):
        aoi_geom = aoi_geometry.union_all()
        aoi_crs = aoi_geometry.crs
    else:
        aoi_geom = aoi_geometry
        aoi_crs = None

    if not isinstance(aoi_geometry, gpd.GeoDataFrame):
        if aoi_crs is None:
            aoi_crs = highways_gdf.crs
        temp_aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs=aoi_crs)
    else:
        temp_aoi_gdf = aoi_geometry.copy()

    if temp_aoi_gdf.crs != highways_gdf.crs:
        logger.info(f"Reprojecting AOI from {temp_aoi_gdf.crs} to {highways_gdf.crs}")
        temp_aoi_gdf = temp_aoi_gdf.to_crs(highways_gdf.crs)
        aoi_geom = temp_aoi_gdf.union_all()

    if buffer_meters > 0:
        if highways_gdf.crs.is_geographic:
            # Buffering in a geographic CRS distorts distances, so
            # reproject to an appropriate UTM zone for the buffer and
            # reproject back afterward.
            utm_crs = temp_aoi_gdf.estimate_utm_crs()

            temp_gdf_utm = temp_aoi_gdf.to_crs(utm_crs)
            temp_gdf_utm.loc[:, 'geometry'] = temp_gdf_utm.geometry.buffer(buffer_meters)
            temp_gdf_buffered = temp_gdf_utm.to_crs(highways_gdf.crs)

            aoi_geom = temp_gdf_buffered.union_all()
        else:
            aoi_geom = aoi_geom.buffer(buffer_meters)

    # Large country-scale networks are routed through the tiled
    # thread-parallel path; smaller (typically already-prefiltered,
    # city-scale) inputs use a plain sjoin.
    if len(highways_gdf) > PARALLEL_CLIP_THRESHOLD:
        logger.info(f"  Large dataset ({len(highways_gdf):,} highways) - using PARALLEL processing...")
        return clip_highways_to_aoi_parallel(
            highways_gdf,
            temp_aoi_gdf,  # AOI already reprojected/buffered above
            buffer_meters=0,
            num_workers=min(16, max(4, int(len(highways_gdf) / 100000)))
        )

    logger.info(f"  Using spatial index to filter {len(highways_gdf):,} highways...")

    try:
        if len(highways_gdf) == 0:
            logger.warning(f"  No highways to filter (input is empty - city likely outside OSM coverage)")
            return gpd.GeoDataFrame(columns=['geometry'], crs=aoi_crs or 'EPSG:4326')

        aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs=highways_gdf.crs)

        intersecting_highways = gpd.sjoin(
            highways_gdf,
            aoi_gdf,
            how='inner',
            predicate='intersects'
        )

        if 'index_right' in intersecting_highways.columns:
            intersecting_highways = intersecting_highways.drop(columns=['index_right'])

        if len(highways_gdf) > 0:
            percentage = len(intersecting_highways) / len(highways_gdf) * 100
        else:
            percentage = 0.0

        logger.info(f"  Spatial filtered: {len(highways_gdf):,} → {len(intersecting_highways):,} ({percentage:.1f}%)")

        if len(intersecting_highways) == 0:
            logger.info(f"  No highways intersect AOI")
            return gpd.GeoDataFrame(columns=highways_gdf.columns, crs=highways_gdf.crs)

        clipped = intersecting_highways.copy()
        clipped.loc[:, 'geometry'] = clipped.geometry.intersection(aoi_geom)
        clipped = clipped[~clipped.geometry.is_empty].copy()

        logger.info(f"  Final clipped: {len(clipped):,} segments")
        return clipped

    except Exception as e:
        logger.error(f"  Error during spatial filtering: {str(e)}")
        # Fall back to a direct spatial-index query if the sjoin path fails.
        possible_matches_index = list(highways_gdf.sindex.query(aoi_geom, predicate='intersects'))
        if len(possible_matches_index) == 0:
            return gpd.GeoDataFrame(columns=highways_gdf.columns, crs=highways_gdf.crs)
        candidates = highways_gdf.iloc[possible_matches_index].copy()
        candidates.loc[:, 'geometry'] = candidates.geometry.intersection(aoi_geom)
        return candidates[~candidates.geometry.is_empty].copy()


def clip_highways_to_aoi_parallel(
    highways_gdf: gpd.GeoDataFrame,
    aoi_geometry,
    buffer_meters: float = 0.0,
    num_workers: int = 8
) -> gpd.GeoDataFrame:
    """
    Clip highway geometries to an AOI using tiled, thread-parallel processing.

    For country-scale networks (1M+ segments), splits the highways into a
    grid of spatial tiles, discards tiles that don't intersect the AOI, and
    filters/intersects the remaining tiles concurrently. This is
    substantially faster than a single sjoin over the whole network.

    Args:
        highways_gdf: GeoDataFrame of highways.
        aoi_geometry: Shapely geometry or GeoDataFrame.
        buffer_meters: Buffer the AOI by this many meters before clipping.
        num_workers: Number of parallel workers.

    Returns:
        Clipped GeoDataFrame.
    """
    if isinstance(aoi_geometry, gpd.GeoDataFrame):
        aoi_geom = aoi_geometry.union_all()
        aoi_crs = aoi_geometry.crs
    else:
        aoi_geom = aoi_geometry
        aoi_crs = None

    if not isinstance(aoi_geometry, gpd.GeoDataFrame):
        if aoi_crs is None:
            aoi_crs = highways_gdf.crs
        temp_aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_geom]}, crs=aoi_crs)
    else:
        temp_aoi_gdf = aoi_geometry.copy()

    if temp_aoi_gdf.crs != highways_gdf.crs:
        logger.info(f"  Reprojecting AOI from {temp_aoi_gdf.crs} to {highways_gdf.crs}")
        temp_aoi_gdf = temp_aoi_gdf.to_crs(highways_gdf.crs)
        aoi_geom = temp_aoi_gdf.union_all()

    if buffer_meters > 0:
        if highways_gdf.crs.is_geographic:
            utm_crs = temp_aoi_gdf.estimate_utm_crs()
            temp_gdf_utm = temp_aoi_gdf.to_crs(utm_crs)
            temp_gdf_utm.loc[:, 'geometry'] = temp_gdf_utm.geometry.buffer(buffer_meters)
            temp_gdf_buffered = temp_gdf_utm.to_crs(highways_gdf.crs)
            aoi_geom = temp_gdf_buffered.union_all()
        else:
            aoi_geom = aoi_geom.buffer(buffer_meters)

    aoi_bounds = aoi_geom.bounds  # (minx, miny, maxx, maxy)

    logger.info(f"  Parallel clipping {len(highways_gdf):,} highways using {num_workers} workers...")

    # Adaptive tiling: aim for roughly 50k highways per tile.
    num_tiles_per_side = max(4, int(np.sqrt(len(highways_gdf) / 50000)))
    chunk_results = _parallel_spatial_filter(
        highways_gdf,
        aoi_geom,
        aoi_bounds,
        num_tiles_per_side,
        num_workers
    )

    if len(chunk_results) == 0:
        logger.info(f"  No highways intersect AOI boundary")
        return gpd.GeoDataFrame(columns=highways_gdf.columns, crs=highways_gdf.crs)

    logger.info(f"  Combining {len(chunk_results)} chunk results...")
    filtered_highways = pd.concat(chunk_results, ignore_index=True)
    logger.info(f"  Spatially filtered: {len(highways_gdf):,} → {len(filtered_highways):,} candidates ({len(filtered_highways)/len(highways_gdf)*100:.1f}%)")

    logger.info(f"  Clipping {len(filtered_highways):,} highways to exact boundary...")
    start_intersection = pd.Timestamp.now()

    clipped_geoms = _parallel_intersection(filtered_highways.geometry, aoi_geom, num_workers)

    result = filtered_highways.copy()
    result['geometry'] = clipped_geoms

    result = result[~result.geometry.is_empty].copy()

    intersection_time = (pd.Timestamp.now() - start_intersection).total_seconds()
    logger.info(f"  Final clipped highways: {len(result):,} segments (intersection took {intersection_time:.1f}s)")

    return result


def _parallel_spatial_filter(
    highways_gdf: gpd.GeoDataFrame,
    aoi_geom,
    aoi_bounds: tuple,
    num_tiles_per_side: int,
    num_workers: int
) -> List[gpd.GeoDataFrame]:
    """
    Spatially filter highways in parallel using a grid of tiles.

    Returns a list of GeoDataFrames, each containing the highways from one
    tile that intersect the AOI.
    """
    full_bounds = highways_gdf.total_bounds  # (minx, miny, maxx, maxy)

    x_step = (full_bounds[2] - full_bounds[0]) / num_tiles_per_side
    y_step = (full_bounds[3] - full_bounds[1]) / num_tiles_per_side

    tiles_to_process = []
    for i in range(num_tiles_per_side):
        for j in range(num_tiles_per_side):
            tile_minx = full_bounds[0] + i * x_step
            tile_miny = full_bounds[1] + j * y_step
            tile_maxx = tile_minx + x_step
            tile_maxy = tile_miny + y_step

            tile_box = box(tile_minx, tile_miny, tile_maxx, tile_maxy)

            if tile_box.intersects(aoi_geom):
                tiles_to_process.append((i, j, tile_box))

    logger.info(f"  Processing {len(tiles_to_process)}/{num_tiles_per_side**2} tiles that intersect AOI...")

    chunk_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for tile_idx, (i, j, tile_box) in enumerate(tiles_to_process):
            future = executor.submit(
                _process_tile,
                highways_gdf,
                tile_box,
                aoi_geom,
                tile_idx,
                len(tiles_to_process)
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    chunk_results.append(result)
            except Exception as e:
                logger.error(f"  Tile processing error: {str(e)}")

    return chunk_results


def _process_tile(
    highways_gdf: gpd.GeoDataFrame,
    tile_box,
    aoi_geom,
    tile_idx: int,
    total_tiles: int
) -> Optional[gpd.GeoDataFrame]:
    """Filter the highways within a single tile that intersect the AOI."""
    try:
        possible_matches_idx = list(highways_gdf.sindex.query(tile_box, predicate='intersects'))

        if len(possible_matches_idx) == 0:
            return None

        tile_highways = highways_gdf.iloc[possible_matches_idx]

        # Vectorized intersects() is 10-50x faster here than iterating
        # geometries one at a time.
        aoi_mask = tile_highways.geometry.intersects(aoi_geom)
        result = tile_highways[aoi_mask]

        if len(result) == 0:
            return None

        if tile_idx % 10 == 0:
            logger.info(f"    Tile {tile_idx+1}/{total_tiles}: {len(result)} highways")

        return result

    except Exception as e:
        logger.error(f"  Error processing tile {tile_idx}: {str(e)}")
        return None


def _parallel_intersection(
    geometries: gpd.GeoSeries,
    aoi_geom,
    num_workers: int
) -> gpd.GeoSeries:
    """Compute geometry intersection in parallel by chunking the input."""
    chunk_size = max(1000, len(geometries) // (num_workers * 4))
    chunks = [geometries.iloc[i:i+chunk_size] for i in range(0, len(geometries), chunk_size)]

    logger.info(f"  Processing intersection in {len(chunks)} chunks with {num_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk_idx, chunk in enumerate(chunks):
            future = executor.submit(_intersect_chunk, chunk, aoi_geom, chunk_idx, len(chunks))
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"  Intersection chunk error: {str(e)}")

    return pd.concat(results, ignore_index=False)


def _intersect_chunk(geom_series: gpd.GeoSeries, aoi_geom, chunk_idx: int, total_chunks: int) -> gpd.GeoSeries:
    """Intersect one chunk of geometries with the AOI."""
    result = geom_series.intersection(aoi_geom)
    if chunk_idx % 20 == 0:
        logger.info(f"    Intersection chunk {chunk_idx+1}/{total_chunks} done")
    return result
