"""Build an Area of Interest (AOI) polygon from an administrative/agglomeration
boundary file: filter by name (and optional extra column filters) -> optionally
dissolve into one polygon -> optionally buffer -> simplify pathologically
complex geometries -> save.

This is a plain-function equivalent of what a generic `AOIProcessor` class
used to do through a multi-processor framework base class. There is exactly
one AOI operation in this pipeline, so the indirection added nothing;
everything it did is inlined here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

#: Africa Albers Equal Area Conic. Used only as an intermediate projected CRS
#: for accurate metric buffering and area calculation of geographic (lon/lat)
#: input data — not the CRS grids are ultimately generated in (that is a
#: per-city UTM zone; see pipeline.detect_utm_crs).
_AFRICA_ALBERS_CRS = "ESRI:102022"

#: Above this coordinate count, write GeoPackage instead of GeoJSON. Some
#: agglomeration boundaries (e.g. large deltas with many small enclosed water
#: bodies) produce polygons with hundreds of thousands of vertices; GeoJSON
#: text encoding of geometries that large is slow to write/read and can
#: produce multi-hundred-MB files, while GeoPackage handles them natively.
_GPKG_COORD_THRESHOLD = 1_000_000

#: Above these thresholds, simplify the geometry before saving (see
#: `_simplify_if_pathological`). Set high enough to only affect the most
#: extreme outlier geometries, not ordinary complex coastlines.
_SIMPLIFY_COORD_THRESHOLD = 400_000
_SIMPLIFY_HOLE_THRESHOLD = 10_000


class AOIError(RuntimeError):
    """Raised when AOI construction fails."""


def _simplify_if_pathological(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Simplify only geometries complex enough to make GeoJSON I/O impractical.

    A handful of agglomeration boundaries include huge numbers of tiny
    interior holes (water bodies, enclaves) that inflate vertex count far
    beyond what is meaningful at the 250 m grid resolution used downstream.
    Left alone, these can produce >100 MB single-feature files and, worse,
    slow enough writes to create read-before-write-completes race conditions
    for concurrent city processing. Ordinary complex coastlines are well
    under the thresholds here and are left untouched.
    """
    for idx, geom in enumerate(gdf.geometry):
        if not hasattr(geom, "exterior"):
            continue

        n_coords = len(geom.exterior.coords)
        n_holes = len(geom.interiors)

        if n_coords <= _SIMPLIFY_COORD_THRESHOLD and n_holes <= _SIMPLIFY_HOLE_THRESHOLD:
            continue

        logger.warning(
            "Complex geometry detected (%s coords, %s holes); simplifying for I/O",
            f"{n_coords:,}", f"{n_holes:,}",
        )

        if n_holes > _SIMPLIFY_HOLE_THRESHOLD:
            # Drop only very small holes (< 0.00035% of total area) — this
            # removes the long tail of negligible slivers while preserving
            # every hole large enough to matter at grid resolution.
            total_area = geom.area
            min_hole_area = total_area * 0.0000035
            large_holes = [h for h in geom.interiors if Polygon(h).area > min_hole_area]
            geom = Polygon(geom.exterior.coords, holes=large_holes)
            logger.info(
                "  Reduced holes %s -> %s", f"{n_holes:,}", f"{len(large_holes):,}",
            )

        tolerance = 5 if "+proj" in str(gdf.crs) else 0.00005
        geom = geom.simplify(tolerance, preserve_topology=True)
        gdf.geometry.iloc[idx] = geom

    return gdf


def build_aoi(
    input_file: Union[str, Path],
    name_column: str,
    names: List[str],
    output_file: Union[str, Path],
    dissolve: bool = False,
    buffer_km: float = 0.0,
    additional_filters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Filter, optionally dissolve/buffer, and save an Area of Interest.

    Args:
        input_file: Path to a boundary file (any format geopandas can read,
            e.g. GeoJSON, GPKG, Shapefile) containing polygon geometries.
        name_column: Column to filter `names` against (e.g. "NAME_0" for
            country boundaries, "Agglomeration_Name" for city boundaries).
        names: Values to keep from `name_column` (typically a single name;
            a list of country names is also supported).
        output_file: Where to write the result. If the simplified geometry
            still has more than `_GPKG_COORD_THRESHOLD` coordinates, a
            sibling `.gpkg` file is written instead and its path returned.
        dissolve: If True, union all matched features into a single polygon
            (used for country mode, where a country may be a MultiPolygon
            of several administrative rows). City/agglomeration boundaries
            are typically already single features and left un-dissolved.
        buffer_km: Optional buffer distance in kilometers, applied after
            reprojecting to an equal-area projected CRS for metric accuracy
            (buffering directly in lon/lat degrees would distort by latitude).
        additional_filters: Extra exact-match (or list-membership) column
            filters, e.g. `{"ISO3": "NGA"}`. Needed because agglomeration
            names are not unique across (or even within) countries — see
            `pipeline.find_city_with_iso3` for the full homonym-handling story.

    Returns:
        Path to the saved AOI file (`output_file`, or its `.gpkg` sibling if
        the geometry was too complex for GeoJSON).

    Raises:
        FileNotFoundError: If `input_file` does not exist.
        AOIError: If no features match, or the result fails to save.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Boundary file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    boundaries = gpd.read_file(input_file)
    if boundaries.crs is None:
        logger.warning("Input file has no CRS; assuming WGS84 (EPSG:4326)")
        boundaries = boundaries.set_crs("EPSG:4326")

    if name_column not in boundaries.columns:
        raise AOIError(
            f"Column '{name_column}' not found in {input_file}. "
            f"Available columns: {', '.join(boundaries.columns)}"
        )

    filtered = boundaries[boundaries[name_column].isin(names)].copy()

    if additional_filters:
        for column, value in additional_filters.items():
            if column not in filtered.columns:
                raise AOIError(
                    f"Filter column '{column}' not found. "
                    f"Available columns: {', '.join(filtered.columns)}"
                )
            if isinstance(value, list):
                filtered = filtered[filtered[column].isin(value)].copy()
            else:
                filtered = filtered[filtered[column] == value].copy()
            logger.info("Filtered by %s = %s: %d features remaining", column, value, len(filtered))

    if filtered.empty:
        raise AOIError(f"No features matched {name_column} in {names} (filters: {additional_filters})")

    if dissolve:
        dissolved_geom = filtered.geometry.union_all() if hasattr(filtered.geometry, "union_all") \
            else filtered.geometry.unary_union
        result = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=filtered.crs)
    else:
        result = filtered

    if buffer_km > 0:
        original_crs = result.crs
        try:
            projected = result.to_crs(_AFRICA_ALBERS_CRS)
            projected.geometry = projected.geometry.buffer(buffer_km * 1000)
            result = projected.to_crs(original_crs)
        except Exception:
            logger.warning(
                "Projected buffering failed; falling back to an approximate "
                "degree-based buffer (less accurate near the poles/equator)"
            )
            buffer_deg = buffer_km / 111.32
            result = result.copy()
            result.geometry = result.geometry.buffer(buffer_deg)

    result = _simplify_if_pathological(result)

    total_coords = sum(
        len(geom.exterior.coords) + sum(len(h.coords) for h in geom.interiors)
        for geom in result.geometry
        if hasattr(geom, "exterior")
    )

    if total_coords > _GPKG_COORD_THRESHOLD:
        output_file = output_file.with_suffix(".gpkg")
        driver = "GPKG"
        logger.info("Complex geometry (%s coordinates); writing GeoPackage instead of GeoJSON", f"{total_coords:,}")
    else:
        driver = "GeoJSON"

    result.to_file(output_file, driver=driver)

    if not output_file.exists():
        raise AOIError(f"Output file was not created: {output_file}")

    logger.info("Saved AOI with %d feature(s) to %s", len(result), output_file)
    return output_file
