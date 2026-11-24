"""Zonal statistics extraction: attach per-hexagon raster values to a grid.

Three extraction paths are used:

- `extract_land_cover_percentages`: for the discrete land cover
  classification raster, computes the percentage of each class within a
  hexagon (single geometry at a time, run in-process since land cover
  produces 23 output columns per raster and isn't worth parallelizing the
  same way as the single-column sources).
- `_process_grid_cell_stable` + `attach_rasters_batch`: for every
  single-column source using `rasterio.mask.mask` (building height/volume,
  land cover cover-fraction layers), computed in parallel across worker
  processes via `multiprocessing.Pool`. Output columns are grouped by the
  raster file that backs them, so a single raster backing several output
  columns (e.g. building height's max/mean/std/sum) is only masked once
  per grid cell.
- `_area_weighted_sum_batch`: for population (`resampling="area_weighted_sum"`
  sources), apportions each pixel's value across intersecting hexagons in
  proportion to the pixel's covered area, via `exactextract`. This differs
  from the `rasterio.mask.mask`-based sum above, which counts every pixel
  that merely touches a hexagon at full weight (`all_touched=True`) — for
  a raster whose pixel size is a non-trivial fraction of the hexagon's
  area, that double-counts pixels shared between adjacent hexagons and
  substantially overstates per-hexagon totals. Area-weighted apportionment
  keeps the sum over all hexagons equal to the sum over the source pixels.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from tqdm import tqdm

from . import config as cfg

logger = logging.getLogger(__name__)

# Hexagon diameter (point-to-point) relative to side length, used to decide
# whether raster pixels are "large" relative to the grid cell.
_HEX_DIAMETER_FACTOR = 1.732  # sqrt(3), point-to-point width of a flat-top hex


def _resolve_all_touched(pixel_size: float, grid_resolution_m: float, configured: bool) -> bool:
    """Decide whether to include partially-overlapping pixels.

    `all_touched=True` is normally desired so no hexagon silently gets no
    data. But it is also the configured default per source. The one case
    that needs to override a source's own setting is when raster pixels are
    coarse relative to the hexagon (e.g. a low-resolution raster against a
    small grid cell): in that regime, using only "pixel center inside
    hexagon" (all_touched=False) risks a hexagon capturing zero pixels
    entirely, so we force all_touched=True whenever the pixel size exceeds
    80% of the hexagon's diameter.
    """
    grid_size = grid_resolution_m * _HEX_DIAMETER_FACTOR
    if pixel_size > grid_size * 0.8:
        return True
    return configured


def extract_land_cover_percentages(
    grid_geometry,
    raster_src,
    nodata_values: set,
    raster_bounds,
    all_touched: bool,
    land_cover_classes: Dict[int, str],
    valid_range: Optional[Tuple[float, float]] = None,
    scale_factor: Optional[float] = None,
) -> Dict[int, float]:
    """Compute the percentage of each land cover class within a hexagon.

    Returns a dict of {class_value: percentage} for classes present in the
    cell (missing classes are implicitly 0%). Returns {} if there is no
    spatial overlap or no valid pixel data.
    """
    try:
        grid_bounds = grid_geometry.bounds
        if (
            grid_bounds[2] < raster_bounds[0]
            or grid_bounds[0] > raster_bounds[2]
            or grid_bounds[3] < raster_bounds[1]
            or grid_bounds[1] > raster_bounds[3]
        ):
            return {}

        masked_data, _ = rasterio_mask(
            raster_src, [grid_geometry], crop=True, nodata=raster_src.nodata, all_touched=all_touched
        )
        if masked_data.ndim > 2:
            masked_data = masked_data[0]

        valid_mask = np.ones(masked_data.shape, dtype=bool)
        valid_mask &= ~np.isnan(masked_data)
        valid_mask &= ~np.isinf(masked_data)

        for nodata_val in nodata_values:
            if isinstance(nodata_val, float):
                valid_mask &= ~np.isclose(masked_data, nodata_val, rtol=1e-5, atol=1e-8)
            else:
                valid_mask &= masked_data != nodata_val

        if valid_range is not None:
            min_val, max_val = valid_range
            valid_mask &= (masked_data >= min_val) & (masked_data <= max_val)

        valid_data = masked_data[valid_mask]
        if len(valid_data) == 0:
            return {}

        if scale_factor is not None and scale_factor != 1.0:
            valid_data = valid_data * scale_factor

        unique_classes, counts = np.unique(valid_data, return_counts=True)
        total_pixels = len(valid_data)

        percentages = {}
        for class_val, count in zip(unique_classes, counts):
            class_int = int(class_val)
            if class_int in land_cover_classes:
                percentages[class_int] = (count / total_pixels) * 100.0
        return percentages

    except Exception as exc:
        if "overlap" not in str(exc).lower():
            logger.debug("Land cover extraction failed: %s", exc)

        # Fall back to a single centroid sample for edge cases the mask
        # operation cannot handle (e.g. a hexagon that only grazes the
        # raster edge).
        try:
            centroid = grid_geometry.centroid
            row, col = rasterio.transform.rowcol(raster_src.transform, centroid.x, centroid.y)
            if 0 <= row < raster_src.height and 0 <= col < raster_src.width:
                value = raster_src.read(1)[row, col]
                if np.isnan(value) or np.isinf(value):
                    return {}
                if nodata_values and value in nodata_values:
                    return {}
                if valid_range is not None:
                    min_val, max_val = valid_range
                    if value < min_val or value > max_val:
                        return {}
                class_int = int(value)
                if class_int in land_cover_classes:
                    return {class_int: 100.0}
        except Exception:
            pass
        return {}


def _process_grid_cell_stable(args: tuple) -> Dict[str, Optional[float]]:
    """Worker function: compute zonal statistics for one hexagon.

    Runs in a separate process (via multiprocessing.Pool), so it reopens the
    raster file independently rather than sharing a handle — slower I/O per
    call, but avoids the instability of sharing GDAL dataset handles across
    process boundaries.

    Args:
        args: (geometry, raster_path, stat_methods, nodata_values, all_touched,
               filter_negative, valid_range)

    Returns:
        {stat_name: value_or_None} for each requested stat method.
    """
    geometry, raster_path, stat_methods, nodata_values, all_touched, filter_negative, valid_range = args

    try:
        with rasterio.open(raster_path) as src:
            masked_data, _ = rasterio_mask(
                src, [geometry], crop=True, nodata=src.nodata, all_touched=all_touched
            )
            if masked_data.ndim > 2:
                masked_data = masked_data[0]

            valid_mask = np.ones(masked_data.shape, dtype=bool)
            valid_mask &= ~np.isnan(masked_data)
            valid_mask &= ~np.isinf(masked_data)

            if filter_negative:
                valid_mask &= masked_data >= 0

            for nodata_val in nodata_values:
                if isinstance(nodata_val, float):
                    valid_mask &= ~np.isclose(masked_data, nodata_val, rtol=1e-5, atol=1e-8)
                else:
                    valid_mask &= masked_data != nodata_val

            if valid_range is not None:
                valid_mask &= (masked_data >= valid_range[0]) & (masked_data <= valid_range[1])

            valid_data = masked_data[valid_mask]

            result: Dict[str, Optional[float]] = {}
            if len(valid_data) > 0:
                for stat in stat_methods:
                    if stat == "max":
                        result[stat] = float(np.max(valid_data))
                    elif stat == "min":
                        result[stat] = float(np.min(valid_data))
                    elif stat == "mean":
                        result[stat] = float(np.mean(valid_data))
                    elif stat == "median":
                        result[stat] = float(np.median(valid_data))
                    elif stat == "sum":
                        result[stat] = float(np.sum(valid_data))
                    elif stat == "std":
                        result[stat] = float(np.std(valid_data))
                    else:
                        result[stat] = None
            else:
                result = {stat: None for stat in stat_methods}
            return result

    except Exception:
        return {stat: None for stat in stat_methods}


_STAT_NAME_BY_METHOD = {
    "max": "max", "min": "min", "mean": "mean",
    "median": "median", "sum": "sum", "std": "std",
}


def _area_weighted_sum_batch(
    grid: gpd.GeoDataFrame,
    raster_path: Path,
    nodata_values: Tuple[float, ...],
    valid_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Area-weighted sum of raster values within each hexagon, via exactextract.

    Unlike `_process_grid_cell_stable`'s `all_touched=True` sum, this
    apportions each pixel's value across intersecting hexagons in
    proportion to the pixel's covered area, so the sum over all hexagons
    equals the sum over the source pixels (no double-counting of pixels
    shared between adjacent hexagons).

    Pixel-level filtering (explicit `nodata_values` beyond the raster's own
    `nodata` tag, and `valid_range`) is applied before handing the raster to
    exactextract: invalid pixels are rewritten to the raster's nodata value
    in an in-memory copy, so exactextract's own nodata handling excludes
    them from the weighted sum exactly as `_process_grid_cell_stable` does
    for the other resampling methods.

    Args:
        grid: Grid GeoDataFrame, already reprojected to the raster's CRS.
        raster_path: Path to the source raster.
        nodata_values: Extra sentinel values (beyond the raster's own
            `nodata` tag) to treat as NoData.
        valid_range: (min, max) inclusive bounds; pixels outside this range
            are treated as NoData.

    Returns:
        Array of per-hexagon sums, shape (len(grid),), NaN where no valid
        pixels intersect the hexagon.
    """
    from exactextract import exact_extract
    from exactextract.raster import RasterioRasterSource
    from rasterio.io import MemoryFile

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        band_nodata = src.nodata

    fill_value = band_nodata if band_nodata is not None else -9999.0
    invalid = ~np.isfinite(data)
    if band_nodata is not None:
        invalid |= data == band_nodata
    for extra_nodata in nodata_values:
        invalid |= np.isclose(data, float(extra_nodata), rtol=1e-5, atol=1e-8)
    if valid_range is not None:
        lo, hi = valid_range
        invalid |= (data < lo) | (data > hi)

    data = np.where(invalid, fill_value, data)
    profile.update(nodata=fill_value)

    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        with mem.open() as masked_src:
            result = exact_extract(
                RasterioRasterSource(masked_src), grid, ["sum"], output="pandas"
            )

    return result["sum"].to_numpy(dtype=np.float64)


def attach_land_cover(
    grid: gpd.GeoDataFrame,
    raster_file: Path,
    source: cfg.DataSource,
    grid_resolution_m: float = cfg.GRID_RESOLUTION_M,
) -> gpd.GeoDataFrame:
    """Attach land cover class percentage columns (one per class) to `grid`.

    Returns a new GeoDataFrame with LC_* columns added (does not mutate the
    input in place, though callers typically discard the old reference).
    """
    land_cover_classes = cfg.LAND_COVER_CLASSES
    land_cover_column_names = cfg.LAND_COVER_COLUMN_NAMES

    grid = grid.copy()

    with rasterio.open(raster_file) as src:
        raster_crs = src.crs
        pixel_size = abs(src.transform.a)
        raster_bounds = src.bounds

        nodata_values = set(source.nodata_values)
        valid_range = source.valid_range
        scale_factor = source.scale_factor

        grid_for_sampling = _reproject_if_needed(grid, raster_crs, raster_bounds)

        configured = cfg.HEXAGON_RESAMPLING_CONFIG.get(
            "land_cover", cfg.HEXAGON_RESAMPLING_CONFIG["default"]
        )
        use_all_touched = _resolve_all_touched(
            pixel_size, grid_resolution_m, configured.get("all_touched", True)
        )

        lc_results = {
            land_cover_column_names.get(class_val, f"LC_{class_val}"): np.zeros(len(grid), dtype=np.float32)
            for class_val in land_cover_classes
        }

        geometries = grid_for_sampling.geometry.values
        with tqdm(total=len(geometries), desc=f"Extracting {source.output_column}", unit="cells") as pbar:
            for idx, geom in enumerate(geometries):
                percentages = extract_land_cover_percentages(
                    grid_geometry=geom,
                    raster_src=src,
                    nodata_values=nodata_values,
                    raster_bounds=raster_bounds,
                    all_touched=use_all_touched,
                    land_cover_classes=land_cover_classes,
                    valid_range=valid_range,
                    scale_factor=scale_factor,
                )
                for class_val, percentage in percentages.items():
                    col_name = land_cover_column_names.get(class_val, f"LC_{class_val}")
                    if col_name in lc_results:
                        lc_results[col_name][idx] = round(percentage, 2)
                pbar.update(1)

        for col_name, values in lc_results.items():
            grid[col_name] = values

    any_lc_data = any((grid[col] > 0).any() for col in lc_results)
    if not any_lc_data:
        logger.warning("No land cover data found for any grid cell")

    return grid


def _reproject_if_needed(grid: gpd.GeoDataFrame, raster_crs, raster_bounds) -> gpd.GeoDataFrame:
    """Reproject `grid` to `raster_crs` if it differs, checking overlap first."""
    grid_crs_str = str(grid.crs).replace(" ", "").upper()
    raster_crs_str = str(raster_crs).replace(" ", "").upper()

    # ESRI:102022 can appear under slightly different string spellings
    # depending on how it was loaded (WKT vs authority code); treat those as
    # equal rather than triggering a spurious reprojection.
    is_same_crs = grid_crs_str == raster_crs_str or (
        "102022" in grid_crs_str and "102022" in raster_crs_str
    )

    if is_same_crs:
        return grid

    logger.info("Reprojecting grid from %s to %s for sampling", grid.crs, raster_crs)
    grid_for_sampling = grid.to_crs(raster_crs)

    grid_bounds = grid_for_sampling.total_bounds
    if (
        grid_bounds[2] < raster_bounds.left
        or grid_bounds[0] > raster_bounds.right
        or grid_bounds[3] < raster_bounds.bottom
        or grid_bounds[1] > raster_bounds.top
    ):
        raise ValueError("No spatial overlap between grid and raster")

    return grid_for_sampling


def attach_rasters_batch(
    grid: gpd.GeoDataFrame,
    raster_files: Dict[str, Path],
    data_sources: Dict[str, cfg.DataSource],
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Attach all single-column raster sources to `grid` in one batched pass.

    Groups output columns by their backing raster file (e.g. building
    height's max/mean/std/sum all read `building_height.tif`), so each
    raster is masked against each hexagon only once regardless of how many
    output columns it produces, then fans the per-cell statistics dict out
    into the requested columns. Each unique raster's per-cell extraction is
    parallelized across a `multiprocessing.Pool`.

    Args:
        grid: Input grid GeoDataFrame (not mutated).
        raster_files: {data_source_key: raster_file_path} for every
            single-column source to attach.
        data_sources: The DATA_SOURCES mapping (or a subset covering the
            keys in raster_files).
        n_workers: Worker process count; defaults to min(cpu_count - 1, 12).

    Returns:
        A new GeoDataFrame with one column added per data source key.
    """
    if not raster_files:
        logger.warning("No raster files to attach")
        return grid

    grid = grid.copy()
    n_workers = n_workers or min(mp.cpu_count() - 1, 12)

    # Group requested columns by the raster file that backs them.
    raster_to_sources: Dict[str, Dict] = {}
    for data_source_key, raster_path in raster_files.items():
        source = data_sources[data_source_key]
        path_key = str(raster_path)
        raster_to_sources.setdefault(
            path_key, {"path": raster_path, "sources": [], "nodata_values": list(source.nodata_values)}
        )
        resampling_cfg = cfg.HEXAGON_RESAMPLING_CONFIG.get(
            data_source_key, cfg.HEXAGON_RESAMPLING_CONFIG["default"]
        )
        raster_to_sources[path_key]["sources"].append(
            {
                "key": data_source_key,
                "method": resampling_cfg.get("method", "max"),
                "col_name": source.output_column,
                "scale_factor": source.scale_factor,
                "source": source,
            }
        )

    for path_key, raster_info in raster_to_sources.items():
        raster_path = raster_info["path"]
        sources = raster_info["sources"]
        nodata_values = raster_info["nodata_values"]
        col_names = [s["col_name"] for s in sources]

        logger.info("Computing %s from %s", ", ".join(col_names), Path(raster_path).name)

        if not Path(raster_path).exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

        valid_range = sources[0]["source"].valid_range
        area_weighted_methods = {s["method"] == "area_weighted_sum" for s in sources}
        if len(area_weighted_methods) > 1:
            raise ValueError(
                f"{raster_path}: sources backed by the same raster cannot mix "
                "area_weighted_sum with other resampling methods "
                f"({[s['key'] for s in sources]})."
            )
        is_area_weighted = area_weighted_methods.pop()

        with rasterio.open(raster_path) as raster_src:
            raster_crs = raster_src.crs
            raster_bounds = raster_src.bounds

        grid_for_sampling = _reproject_if_needed(grid, raster_crs, raster_bounds)

        if is_area_weighted:
            logger.info("Processing %d cells (area-weighted sum, exactextract)...", len(grid))
            aw_values = _area_weighted_sum_batch(
                grid_for_sampling, raster_path, tuple(nodata_values), valid_range
            )
            stats = [{"sum": v if np.isfinite(v) else None} for v in aw_values]
        else:
            filter_negative = any(s["source"].filter_negative for s in sources)
            stats_needed = list({_STAT_NAME_BY_METHOD.get(s["method"], "max") for s in sources})
            process_args = [
                (geom, str(raster_path), stats_needed, nodata_values, True, filter_negative, valid_range)
                for geom in grid_for_sampling.geometry
            ]
            logger.info("Processing %d cells using %d worker processes...", len(grid), n_workers)
            with Pool(n_workers) as pool:
                stats = pool.map(_process_grid_cell_stable, process_args)

        for source in sources:
            stat_name = "sum" if is_area_weighted else _STAT_NAME_BY_METHOD.get(source["method"], "max")
            col_name = source["col_name"]
            scale_factor = source["scale_factor"]

            values = np.array(
                [s[stat_name] if s[stat_name] is not None else np.nan for s in stats], dtype=np.float32
            )
            if scale_factor is not None and scale_factor != 1.0:
                valid_mask = ~np.isnan(values)
                values[valid_mask] = values[valid_mask] * scale_factor
            grid[col_name] = values

            data_type = source["source"].data_type
            precision = cfg.PRECISION_CONFIG.get(data_type, cfg.PRECISION_CONFIG["default"])
            target_dtype = cfg.DTYPE_CONFIG.get(data_type, cfg.DTYPE_CONFIG["default"])
            if precision < 6:
                grid[col_name] = grid[col_name].round(precision)
            grid[col_name] = grid[col_name].astype(target_dtype)

            valid_count = grid[col_name].notna().sum()
            logger.info("  %s: %s valid values", col_name, f"{valid_count:,}")

    return grid
