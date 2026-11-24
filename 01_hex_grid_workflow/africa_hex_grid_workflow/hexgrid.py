"""Flat-top hexagonal grid generation.

Uses axial coordinates (q, r) as described in the standard hex-grid
literature (e.g. Red Blob Games' hexagon guide): each hexagon has an integer
(q, r) address, and cartesian <-> axial conversion is exact except for the
necessary rounding step when going from an arbitrary cartesian point back to
the nearest hexagon.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HexagonalGridGenerator:
    """Generator for flat-top hexagonal grids with a configurable side length."""

    def __init__(self, side_length: float = 250.0):
        self.side_length = side_length
        # Flat-top hexagon dimensions.
        self.width = side_length * np.sqrt(3)   # point-to-point horizontally
        self.height = side_length * 2           # point-to-point vertically
        self.vert_spacing = side_length * 1.5   # center-to-center vertically
        self.horiz_spacing = self.width         # center-to-center horizontally

    def axial_to_cartesian(
        self, q: int, r: int, origin_x: float = 0, origin_y: float = 0
    ) -> Tuple[float, float]:
        """Convert axial coordinates (q, r) to cartesian coordinates (x, y)."""
        x = origin_x + self.side_length * np.sqrt(3) * (q + r / 2)
        y = origin_y + self.side_length * 3 / 2 * r
        return x, y

    def cartesian_to_axial(
        self, x: float, y: float, origin_x: float = 0, origin_y: float = 0
    ) -> Tuple[int, int]:
        """Convert cartesian coordinates to the nearest axial coordinates."""
        rel_x = x - origin_x
        rel_y = y - origin_y

        q_float = (np.sqrt(3) / 3 * rel_x - 1 / 3 * rel_y) / self.side_length
        r_float = (2 / 3 * rel_y) / self.side_length

        return self._axial_round(q_float, r_float)

    def _axial_round(self, q_float: float, r_float: float) -> Tuple[int, int]:
        """Round fractional axial coordinates to the nearest integer hex.

        Naive independent rounding of q and r can violate the cube-coordinate
        constraint q + r + s = 0; this reconciles the largest rounding error
        against the other two components, which is the standard correct
        algorithm for hex-grid rounding.
        """
        s_float = -q_float - r_float

        q_round = round(q_float)
        r_round = round(r_float)
        s_round = round(s_float)

        q_diff = abs(q_round - q_float)
        r_diff = abs(r_round - r_float)
        s_diff = abs(s_round - s_float)

        if q_diff > r_diff and q_diff > s_diff:
            q_round = -r_round - s_round
        elif r_diff > s_diff:
            r_round = -q_round - s_round

        return int(q_round), int(r_round)

    def create_hexagon_polygon(self, center_x: float, center_y: float) -> Polygon:
        """Create a flat-top hexagon polygon centered at (center_x, center_y)."""
        vertices = []
        for i in range(6):
            # Flat-top hexagons start at 30 degrees and step by 60 degrees.
            angle = np.pi / 3 * i + np.pi / 6
            vertex_x = center_x + self.side_length * np.cos(angle)
            vertex_y = center_y + self.side_length * np.sin(angle)
            vertices.append((vertex_x, vertex_y))
        vertices.append(vertices[0])
        return Polygon(vertices)

    def generate_hex_id(
        self,
        q: int,
        r: int,
        country_iso3: Optional[str] = None,
        city_index: Optional[int] = None,
        cell_number: int = 0,
    ) -> str:
        """Generate a standardized hexagon GridID.

        Args:
            q, r: Hexagon axial coordinates.
            country_iso3: Country ISO3 code (e.g. "GHA"), used when
                available to make the ID human-traceable to its source area.
            city_index: 1-based index of the city within its country.
            cell_number: 0-based index of the cell within its city/country.

        Returns:
            "{ISO3}_{city_index:04d}_{cell_number:06d}" when `country_iso3`
            and `city_index` are both given (e.g. "GHA_0005_000123");
            otherwise an axial-coordinate-based ID ("HQ{q:+07d}_R{r:+07d}")
            usable for standalone/exploratory grid generation.
        """
        if country_iso3 and city_index is not None:
            return f"{country_iso3}_{city_index:04d}_{cell_number:06d}"
        return f"HQ{q:+07d}_R{r:+07d}"

    def get_hexagons_in_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        origin_x: float,
        origin_y: float,
    ) -> List[Dict]:
        """Generate hexagons that intersect the given bounds (no progress bar)."""
        hexagons = []

        min_q, max_r = self.cartesian_to_axial(min_x, max_y, origin_x, origin_y)
        max_q, min_r = self.cartesian_to_axial(max_x, min_y, origin_x, origin_y)

        # Pad the coordinate range so edge hexagons whose center falls just
        # outside the naive axial range, but whose body still overlaps the
        # bounds, are not missed.
        padding = 3
        min_q -= padding
        max_q += padding
        min_r -= padding
        max_r += padding

        for r in range(min_r, max_r + 1):
            for q in range(min_q, max_q + 1):
                center_x, center_y = self.axial_to_cartesian(q, r, origin_x, origin_y)

                if (
                    center_x - self.width / 2 <= max_x
                    and center_x + self.width / 2 >= min_x
                    and center_y - self.height / 2 <= max_y
                    and center_y + self.height / 2 >= min_y
                ):
                    polygon = self.create_hexagon_polygon(center_x, center_y)
                    hex_id = self.generate_hex_id(q, r)
                    hexagons.append(
                        {
                            "GridID": hex_id,
                            "geometry": polygon,
                            "q": q,
                            "r": r,
                            "center_x": center_x,
                            "center_y": center_y,
                        }
                    )

        return hexagons

    def get_hexagons_in_bounds_with_progress(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        origin_x: float,
        origin_y: float,
        area_name: str = "",
        country_iso3: Optional[str] = None,
        city_index: Optional[int] = None,
    ) -> List[Dict]:
        """Generate hexagons within bounds, reporting progress and encoding GridIDs.

        Args:
            min_x, min_y, max_x, max_y: Bounds in the target CRS.
            origin_x, origin_y: Grid origin coordinates (see
                config.GLOBAL_GRID_ORIGIN_X/Y for the shared reference point).
            area_name: Display name for the progress bar.
            country_iso3: ISO3 code for GridID encoding.
            city_index: City index for GridID encoding.
        """
        hexagons = []

        min_q, max_r = self.cartesian_to_axial(min_x, max_y, origin_x, origin_y)
        max_q, min_r = self.cartesian_to_axial(max_x, min_y, origin_x, origin_y)

        padding = 3
        min_q -= padding
        max_q += padding
        min_r -= padding
        max_r += padding

        total_r = max_r - min_r + 1
        total_q = max_q - min_q + 1
        total_iterations = total_r * total_q

        logger.info(
            "Scanning %d rows x %d cols = %s coordinate positions",
            total_r, total_q, f"{total_iterations:,}",
        )

        cell_counter = 0
        with tqdm(
            total=total_iterations, desc=f"Creating {area_name} hexagonal grid", unit="coords"
        ) as pbar:
            for r in range(min_r, max_r + 1):
                for q in range(min_q, max_q + 1):
                    center_x, center_y = self.axial_to_cartesian(q, r, origin_x, origin_y)

                    if (
                        center_x - self.width / 2 <= max_x
                        and center_x + self.width / 2 >= min_x
                        and center_y - self.height / 2 <= max_y
                        and center_y + self.height / 2 >= min_y
                    ):
                        polygon = self.create_hexagon_polygon(center_x, center_y)
                        hex_id = self.generate_hex_id(q, r, country_iso3, city_index, cell_counter)
                        cell_counter += 1

                        hexagons.append(
                            {
                                "GridID": hex_id,
                                "geometry": polygon,
                                "q": q,
                                "r": r,
                                "center_x": center_x,
                                "center_y": center_y,
                            }
                        )

                    pbar.update(1)

        return hexagons
