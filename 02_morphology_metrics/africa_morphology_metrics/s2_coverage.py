"""
S2 cell coverage utilities for the Open Buildings GCS extraction.

The Open Buildings v3 bucket partitions data by S2 cell at a fixed level
(6), so extracting buildings for an AOI means first computing which S2
cells at that level cover the AOI's bounding box. Implemented with
``s2sphere`` (a pure-Python S2 implementation) rather than Google's
``s2geometry`` bindings, since s2sphere works on Windows and needs no
platform-specific build step.
"""

from typing import List
from s2sphere import CellId, Cell, RegionCoverer, LatLng, LatLngRect
import shapely.geometry
from shapely.geometry.base import BaseGeometry


def get_bounding_box_s2_covering_tokens(
    geometry: BaseGeometry,
    level: int = 6,
    max_cells: int = 1_000_000
) -> List[str]:
    """
    Get the S2 cell tokens covering a geometry's bounding box.

    Args:
        geometry: Shapely geometry to cover.
        level: S2 cell level (default 6, matching the GCS bucket's
            partitioning).
        max_cells: Maximum number of cells to return.

    Returns:
        List of S2 token strings.

    Example:
        >>> from shapely.geometry import box
        >>> bbox = box(-1, 5, 0, 6)
        >>> tokens = get_bounding_box_s2_covering_tokens(bbox, level=6)
    """
    minx, miny, maxx, maxy = geometry.bounds

    lo = LatLng.from_degrees(miny, minx)
    hi = LatLng.from_degrees(maxy, maxx)
    rect = LatLngRect.from_point_pair(lo, hi)

    coverer = RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = max_cells

    covering = coverer.get_covering(rect)
    tokens = [cell.to_token() for cell in covering]

    return tokens


def s2_token_to_shapely_polygon(token: str) -> shapely.geometry.Polygon:
    """
    Convert an S2 token to its Shapely polygon footprint.

    Args:
        token: S2 cell token string.

    Returns:
        Shapely Polygon representing the S2 cell.

    Example:
        >>> polygon = s2_token_to_shapely_polygon("89c25")
        >>> print(polygon.bounds)
    """
    cell_id = CellId.from_token(token)
    cell = Cell(cell_id)

    vertices = []
    for i in range(4):
        vertex = cell.get_vertex(i)
        lat_lng = LatLng.from_point(vertex)
        vertices.append((lat_lng.lng().degrees, lat_lng.lat().degrees))

    vertices.append(vertices[0])

    return shapely.geometry.Polygon(vertices)


def estimate_cells_for_geometry(geometry: BaseGeometry, level: int = 6) -> int:
    """
    Estimate the number of S2 cells needed to cover a geometry.

    Args:
        geometry: Shapely geometry.
        level: S2 cell level.

    Returns:
        Estimated number of cells, based on bounding box area (the actual
        count may differ, since the covering follows the cell grid, not
        the bounding box exactly).
    """
    minx, miny, maxx, maxy = geometry.bounds
    bbox_area = (maxx - minx) * (maxy - miny)

    # Approximate cell edge length at each level, in degrees.
    cell_sizes = {
        4: 5.09,
        5: 2.54,
        6: 1.27,
        7: 0.635,
        8: 0.318,
    }

    cell_size = cell_sizes.get(level, 1.27)
    estimated_cells = int(bbox_area / (cell_size * cell_size)) + 1

    return estimated_cells


def validate_s2_level(level: int) -> bool:
    """
    Check whether an S2 level is appropriate for Open Buildings data.

    Args:
        level: S2 cell level.

    Returns:
        True if valid. Open Buildings v3 uses level 6 in its GCS bucket
        partitioning; other levels may not have corresponding data.
    """
    return 4 <= level <= 8


def get_s2_cell_area(level: int) -> float:
    """
    Get the approximate area of an S2 cell at a given level, in km².

    Args:
        level: S2 cell level.

    Returns:
        Approximate area in km².
    """
    earth_area_km2 = 510_000_000
    num_cells = 6 * (4 ** level)
    return earth_area_km2 / num_cells
