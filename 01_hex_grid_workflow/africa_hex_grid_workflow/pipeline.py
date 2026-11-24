"""Orchestrates the full grid workflow for one area of interest (AOI):
generate AOI -> generate hexagonal grid -> attach raster data sources ->
compute Shannon diversity -> export a grid boundary envelope.

Everything that used to be a hardcoded absolute path is a constructor
argument here: the boundary files, the pre-clipped country-TIF cache
directory, and the output data root are all supplied by the caller (or via
CLI flags in `main()`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import fiona
import geopandas as gpd
from shapely.geometry import Point

from . import config as cfg
from .aoi import build_aoi
from .diversity import calculate_shannon_diversity
from .hexgrid import HexagonalGridGenerator
from .raster_extract import attach_land_cover, attach_rasters_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary file configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryConfig:
    """Where to find one boundary layer and which column holds its name."""

    file_path: Path
    name_column: str


@dataclass(frozen=True)
class BoundariesConfig:
    """Country and city/agglomeration boundary sources.

    `country.file_path` must resolve to a polygon layer with at least a
    country-name column (default "NAME_0") and, for city mode, an "ISO"
    column with ISO3 country codes. `city.file_path` must resolve to a
    polygon layer with at least a city/agglomeration name column (default
    "Agglomeration_Name") and an "ISO3" column, plus an ID column
    ("Agglomeration_ID" or "agglosID") used to disambiguate cities that
    share a name. See the README for the AFRICAPOLIS 2020 and Natural Earth
    sources this is designed around.
    """

    country: BoundaryConfig
    city: BoundaryConfig


# ---------------------------------------------------------------------------
# City homonym resolution
# ---------------------------------------------------------------------------


def find_city_with_iso3(
    boundaries: BoundaryConfig, city_name: str, iso3_filter: Optional[str] = None
) -> Optional[Dict]:
    """Find a city's feature properties, correctly handling duplicate names.

    AFRICAPOLIS-style agglomeration data contains multiple cities that share
    a name — either across countries (e.g. a city named "Offa" exists in
    both Nigeria and Cote d'Ivoire) or, more subtly, within the same country
    (e.g. two places named "Baga" roughly 5 km apart in Nigeria). Returning
    the first name match without a country filter silently picks whichever
    city happens to come first in the file, which has previously produced
    grids for the wrong physical location. Providing `iso3_filter` is the
    only way to reliably disambiguate; without it, this returns the first
    match and logs a warning if more than one exists.

    Matching is intentionally exact-string first (not stripped): some source
    records carry a trailing space in the name field (e.g. "Baga "), and
    stripping unconditionally could accidentally conflate two distinct
    close-but-not-identical name variants. Exact match is tried first, and
    a stripped-name fallback is only used if no exact match exists at all.

    Args:
        boundaries: The city BoundaryConfig (file path + name column).
        city_name: Name to search for (as provided by the caller; not
            pre-stripped).
        iso3_filter: If given, restrict matches to this ISO3 code.

    Returns:
        The matched feature's properties dict, or None if no match exists.
    """
    logger.info(
        "Searching for city '%s' (len=%d, ISO3 filter=%s)", city_name, len(city_name), iso3_filter
    )

    matches = []
    with fiona.open(boundaries.file_path) as src:
        for feature in src:
            if feature["properties"][boundaries.name_column] == city_name:
                matches.append(feature)

    if not matches:
        logger.warning("City '%s' not found with exact match; trying stripped name", city_name)
        city_name_stripped = city_name.strip()
        with fiona.open(boundaries.file_path) as src:
            for feature in src:
                feature_name = feature["properties"][boundaries.name_column]
                if feature_name == city_name_stripped or feature_name.strip() == city_name_stripped:
                    matches.append(feature)

        if not matches:
            logger.error("City '%s' not found even with stripped-name matching", city_name)
            return None

    if len(matches) > 1:
        countries = list({f["properties"]["ISO3"] for f in matches})
        logger.warning(
            "Multiple cities named '%s' found (%d entries) across countries: %s",
            city_name, len(matches), countries,
        )

    if iso3_filter:
        filtered = [f for f in matches if f["properties"]["ISO3"] == iso3_filter]
        if not filtered:
            logger.error(
                "City '%s' not found in country %s (available: %s)",
                city_name, iso3_filter, [f["properties"]["ISO3"] for f in matches],
            )
            return None
        if len(filtered) > 1:
            logger.warning(
                "Multiple cities named '%s' remain within %s; using the first entry",
                city_name, iso3_filter,
            )
        return filtered[0]["properties"]

    if len(matches) > 1:
        logger.warning("No ISO3 filter given; using the first of %d matches", len(matches))
    return matches[0]["properties"]


# ---------------------------------------------------------------------------
# UTM detection
# ---------------------------------------------------------------------------


def detect_utm_crs(aoi: gpd.GeoDataFrame) -> str:
    """Pick the UTM zone whose central meridian is closest to the AOI's centroid.

    A single continent-wide projection (e.g. an Albers equal-area conic)
    distorts hexagon shape and area non-uniformly across its extent, most
    severely near the projection's edges. Assigning each city its own UTM
    zone based on its own centroid keeps distortion negligible at the scale
    of a single city, at the cost of grids from different cities not sharing
    one common CRS (each is independently valid; see config.py for how grid
    alignment across cities is preserved anyway).

    Args:
        aoi: The AOI GeoDataFrame (any CRS; reprojected to WGS84 internally).

    Returns:
        An EPSG code string, e.g. "EPSG:32630" (UTM zone 30N).
    """
    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    centroid = aoi_wgs84.geometry.centroid.iloc[0]
    lon, lat = centroid.x, centroid.y

    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone

    crs = f"EPSG:{epsg_code}"
    logger.info("Detected UTM CRS from centroid (%.2f, %.2f): %s", lat, lon, crs)
    return crs


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Sanitize a name for use as a path component."""
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    for char in "?*|\"<>":
        name = name.replace(char, "")
    return name.strip().strip("_")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GridPipeline:
    """Runs the full AOI -> grid -> raster-attachment -> diversity workflow
    for one country or city.
    """

    def __init__(
        self,
        aoi_name: str,
        aoi_mode: str,
        data_dir: Path,
        country_tif_dir: Path,
        boundaries: BoundariesConfig,
        grid_resolution_m: float = cfg.GRID_RESOLUTION_M,
        buffer_km: float = 0.0,
        target_crs: Optional[str] = None,
        country_iso3: Optional[str] = None,
        city_index: Optional[int] = None,
        force_regenerate: Optional[Dict[str, bool]] = None,
    ):
        """
        Args:
            aoi_name: Country or city/agglomeration name.
            aoi_mode: "country" or "city".
            data_dir: Root output directory; AOI, grid, and boundary
                subdirectories (config.DIR_STRUCTURE) are created under it.
            country_tif_dir: Root of the pre-clipped per-country raster
                cache (see clip_country_rasters.py); expected layout is
                `{country_tif_dir}/{Country_With_Underscores}/{file}.tif`.
            boundaries: Country and city boundary file configuration.
            grid_resolution_m: Hexagon side length in meters.
            buffer_km: Buffer applied to the AOI before gridding.
            target_crs: Explicit CRS (e.g. "EPSG:32630"); if None, detected
                from the AOI centroid (see detect_utm_crs).
            country_iso3: ISO3 code, used both to disambiguate homonym
                cities and to encode into the GridID.
            city_index: 1-based index of this city within its country, used
                to encode into the GridID; if None, a fallback ID scheme is
                used (see hexgrid.HexagonalGridGenerator.generate_hex_id).
            force_regenerate: Optional overrides for the default "skip if
                output already exists" behavior, e.g.
                {"aoi": True, "grid": True, "land_cover": True,
                 "land_cover_fraction": True, "population": True,
                 "attachment": True, "shannon": True}.
        """
        if aoi_mode not in ("country", "city"):
            raise ValueError(f"aoi_mode must be 'country' or 'city', got {aoi_mode!r}")

        self.aoi_name = aoi_name
        self.aoi_mode = aoi_mode
        self.data_dir = Path(data_dir)
        self.country_tif_dir = Path(country_tif_dir)
        self.boundaries = boundaries
        self.grid_resolution = grid_resolution_m
        self.buffer_km = buffer_km
        self.country_iso3 = country_iso3
        self.city_index = city_index
        self.force = {
            "aoi": False, "grid": False, "attachment": False,
            "land_cover": False, "land_cover_fraction": False,
            "population": False, "shannon": False,
            **(force_regenerate or {}),
        }

        self._country_name: Optional[str] = None
        if self.aoi_mode == "city":
            self._resolve_country_for_city()

        self._setup_directories()

        if target_crs:
            self.target_crs = target_crs
        else:
            self.target_crs = self._detect_crs_for_aoi()

        logger.info("Using CRS: %s", self.target_crs)
        logger.info(
            "Initialized GridPipeline for %s (%s mode), resolution=%sm, buffer=%skm",
            self.aoi_name, self.aoi_mode, self.grid_resolution, self.buffer_km,
        )

    # -- setup -------------------------------------------------------------

    def _resolve_country_for_city(self) -> None:
        """Look up the country name and ISO3 for a city, populating `_country_name`."""
        city_data = find_city_with_iso3(self.boundaries.city, self.aoi_name, self.country_iso3)
        if not city_data:
            raise ValueError(f"City '{self.aoi_name}' not found in agglomeration data")

        iso3_code = city_data["ISO3"]
        countries_gdf = gpd.read_file(self.boundaries.country.file_path)
        country_row = countries_gdf[countries_gdf["ISO"] == iso3_code]

        if len(country_row) == 0:
            raise RuntimeError(
                f"Country with ISO3 '{iso3_code}' not found in {self.boundaries.country.file_path}"
            )

        country_name = country_row.iloc[0][self.boundaries.country.name_column]
        self._country_name = country_name.replace("'", "").replace("(", "").replace(")", "")
        if not self.country_iso3:
            self.country_iso3 = iso3_code

    def _setup_directories(self) -> None:
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        for path in [self._aoi_path(), self._grid_path(), self.data_dir / cfg.DIR_STRUCTURE["boundaries"]]:
            path.mkdir(parents=True, exist_ok=True)

    def _aoi_path(self) -> Path:
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        if self.aoi_mode == "city" and self._country_name:
            return self.data_dir / cfg.DIR_STRUCTURE["aoi"] / self._country_name / safe_aoi_name
        return self.data_dir / cfg.DIR_STRUCTURE["aoi"] / safe_aoi_name

    def _grid_path(self) -> Path:
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        if self.aoi_mode == "city" and self._country_name:
            return self.data_dir / cfg.DIR_STRUCTURE["grids"] / self._country_name / safe_aoi_name
        return self.data_dir / cfg.DIR_STRUCTURE["grids"] / safe_aoi_name

    def _country_tif_subdir(self) -> Path:
        safe_name = (self._country_name or "").replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
        return self.country_tif_dir / safe_name

    def _detect_crs_for_aoi(self) -> str:
        aoi = self._load_aoi_geometry_for_crs_detection()
        if aoi is None:
            logger.warning("Could not load AOI geometry; falling back to ESRI:102022 (may distort hexagons)")
            return "ESRI:102022"
        return detect_utm_crs(aoi)

    def _load_aoi_geometry_for_crs_detection(self) -> Optional[gpd.GeoDataFrame]:
        if self.aoi_mode == "city":
            city_data = find_city_with_iso3(self.boundaries.city, self.aoi_name, self.country_iso3)
            if not city_data:
                return None
            agglom = gpd.read_file(self.boundaries.city.file_path)

            id_field = next(
                (f for f in ("Agglomeration_ID", "agglosID") if f in agglom.columns), None
            )
            if id_field is None:
                matched = agglom[agglom[self.boundaries.city.name_column] == self.aoi_name]
                return matched if len(matched) > 0 else None

            city_id = city_data.get("Agglomeration_ID") or city_data.get("agglosID")
            if not city_id:
                return None
            matched = agglom[agglom[id_field] == city_id]
            return matched if len(matched) > 0 else None

        # country mode
        boundaries = gpd.read_file(self.boundaries.country.file_path)
        matched = boundaries[boundaries[self.boundaries.country.name_column] == self.aoi_name]
        return matched if len(matched) > 0 else None

    # -- AOI -----------------------------------------------------------

    def _aoi_output_paths(self) -> tuple[Path, Path]:
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        base = self._aoi_path() / f"{safe_aoi_name}_aoi"
        return base.with_suffix(".geojson"), base.with_suffix(".gpkg")

    def generate_aoi(self) -> Path:
        """Build and save this pipeline's AOI polygon."""
        logger.info("Generating AOI for %s (%s mode)", self.aoi_name, self.aoi_mode)
        aoi_json, aoi_gpkg = self._aoi_output_paths()

        if self.aoi_mode == "country":
            boundaries_cfg = self.boundaries.country
            names = [self.aoi_name]
            additional_filters = None
            dissolve = True
        else:
            boundaries_cfg = self.boundaries.city
            city_data = find_city_with_iso3(boundaries_cfg, self.aoi_name, self.country_iso3)
            if not city_data:
                raise ValueError(f"City '{self.aoi_name}' not found in agglomeration data")
            iso3_code = city_data["ISO3"]
            # Use the exact name string from the source data (which may
            # carry formatting quirks like a trailing space) rather than the
            # caller-supplied name, so the AOIProcessor filter matches.
            actual_city_name = city_data[boundaries_cfg.name_column]
            names = [actual_city_name]
            additional_filters = {"ISO3": iso3_code}
            dissolve = False

        return build_aoi(
            input_file=boundaries_cfg.file_path,
            name_column=boundaries_cfg.name_column,
            names=names,
            output_file=aoi_json,
            dissolve=dissolve,
            buffer_km=0,
            additional_filters=additional_filters,
        )

    # -- grid ------------------------------------------------------------

    def _grid_file_path(self) -> Path:
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        return self._grid_path() / f"{safe_aoi_name}_grid_{self.grid_resolution}m_hex.shp"

    def generate_grid(self, aoi_file: Path) -> Path:
        """Generate the hexagonal grid covering `aoi_file`, aligned to the
        global reference origin, filtered to hexagons whose center falls
        inside the (optionally buffered) AOI.

        Filtering by centroid containment rather than full polygon
        intersection keeps every hexagon's full geometry intact (no clipped
        partial hexagons at the boundary) while still tracking the AOI
        extent closely — a hexagon whose center lies just outside the AOI
        is excluded even though a sliver of it might overlap.
        """
        grid_file = self._grid_file_path()

        aoi = gpd.read_file(aoi_file)
        if str(aoi.crs) != self.target_crs:
            aoi = aoi.to_crs(self.target_crs)

        buffered_aoi = aoi.buffer(self.buffer_km * 1000)
        min_x, min_y, max_x, max_y = buffered_aoi.total_bounds

        hex_generator = HexagonalGridGenerator(side_length=self.grid_resolution)
        hex_candidates = hex_generator.get_hexagons_in_bounds_with_progress(
            min_x, min_y, max_x, max_y,
            cfg.GLOBAL_GRID_ORIGIN_X, cfg.GLOBAL_GRID_ORIGIN_Y,
            area_name=self.aoi_name,
            country_iso3=self.country_iso3,
            city_index=self.city_index,
        )
        logger.info("Generated %s hexagon candidates", f"{len(hex_candidates):,}")

        grid_gdf = gpd.GeoDataFrame(
            {
                "GridID": [h["GridID"] for h in hex_candidates],
                "geometry": [h["geometry"] for h in hex_candidates],
                "center_x": [h["center_x"] for h in hex_candidates],
                "center_y": [h["center_y"] for h in hex_candidates],
            },
            crs=self.target_crs,
        )

        buffered_union = (
            buffered_aoi.union_all() if hasattr(buffered_aoi, "union_all") else buffered_aoi.unary_union
        )
        aoi_bounds = buffered_union.bounds

        bounds_mask = (
            (grid_gdf.geometry.bounds.minx < aoi_bounds[2])
            & (grid_gdf.geometry.bounds.maxx > aoi_bounds[0])
            & (grid_gdf.geometry.bounds.miny < aoi_bounds[3])
            & (grid_gdf.geometry.bounds.maxy > aoi_bounds[1])
        )
        candidates = grid_gdf[bounds_mask].copy()
        logger.info("Bounds pre-filter: %s / %s candidates", f"{len(candidates):,}", f"{len(grid_gdf):,}")

        if len(candidates) > 0:
            center_points = gpd.GeoDataFrame(
                {"temp_idx": candidates.index},
                geometry=[Point(x, y) for x, y in zip(candidates["center_x"], candidates["center_y"])],
                crs=self.target_crs,
            )
            aoi_gdf = gpd.GeoDataFrame({"aoi_id": [1]}, geometry=[buffered_union], crs=self.target_crs)
            matched = gpd.sjoin(center_points, aoi_gdf, how="inner", predicate="within")
            grid_gdf = candidates.loc[matched["temp_idx"]].copy()
        else:
            grid_gdf = gpd.GeoDataFrame(columns=grid_gdf.columns, crs=self.target_crs)

        logger.info("Filtered to %s cells with centroid inside the AOI", f"{len(grid_gdf):,}")

        if self.aoi_mode == "country":
            grid_gdf["Country"] = self.aoi_name
        else:
            grid_gdf["City"] = self.aoi_name

        grid_gdf["Area_m2"] = grid_gdf.geometry.area
        grid_gdf["Center_X"] = grid_gdf.geometry.centroid.x.round(cfg.PRECISION_CONFIG["coordinates"])
        grid_gdf["Center_Y"] = grid_gdf.geometry.centroid.y.round(cfg.PRECISION_CONFIG["coordinates"])

        self._save_grid(grid_gdf, grid_file)
        logger.info("Grid created: %s (%s cells)", grid_file, f"{len(grid_gdf):,}")
        return grid_file

    # -- raster attachment -------------------------------------------------

    def _find_country_tif(self, data_source_key: str) -> Path:
        for tif_name, source_keys in cfg.COUNTRY_TIF_MAPPING.items():
            if data_source_key in source_keys:
                return self._country_tif_subdir() / tif_name
        raise ValueError(f"No country-TIF mapping found for {data_source_key}")

    def attach_data_sources(self, grid_file: Path) -> Path:
        """Attach every configured raster data source to the grid, skipping
        sources whose output columns already exist unless force-regenerated.
        """
        grid = gpd.read_file(grid_file)

        multi_column_sources: Dict[str, Path] = {}
        batch_sources: Dict[str, Path] = {}

        for key, source in cfg.DATA_SOURCES.items():
            tif_file = self._find_country_tif(key)

            if key == "land_cover":
                force = self.force["land_cover"]
            elif key.startswith("lc_fraction_"):
                force = self.force["land_cover_fraction"]
            elif key == "population_2022":
                force = self.force["population"]
            else:
                force = self.force["attachment"]

            if not force and self._column_exists(grid, key, source):
                continue

            if source.output_type == "multi_column":
                multi_column_sources[key] = tif_file
            else:
                batch_sources[key] = tif_file

        for key, tif_file in multi_column_sources.items():
            source = cfg.DATA_SOURCES[key]
            grid = attach_land_cover(grid, tif_file, source, self.grid_resolution)
            logger.info("[%s] attached (multi-column)", source.output_column)

        if batch_sources:
            grid = attach_rasters_batch(grid, batch_sources, cfg.DATA_SOURCES)
            for key in batch_sources:
                logger.info("[%s] batch attached", cfg.DATA_SOURCES[key].output_column)

        enriched_file = grid_file.with_name(grid_file.stem + "_enriched.shp")
        self._save_grid(grid, enriched_file)
        return enriched_file

    def _column_exists(self, grid: gpd.GeoDataFrame, data_source_key: str, source: cfg.DataSource) -> bool:
        if source.output_type == "multi_column":
            existing = [
                cfg.LAND_COVER_COLUMN_NAMES.get(v, f"LC_{v}")
                for v in cfg.LAND_COVER_CLASSES
                if cfg.LAND_COVER_COLUMN_NAMES.get(v, f"LC_{v}") in grid.columns
            ]
            if not existing:
                return False
            has_data = any((grid[col] > 0).any() for col in existing)
            return has_data and len(existing) == len(cfg.LAND_COVER_CLASSES)

        if source.output_column not in grid.columns:
            return False
        return grid[source.output_column].notna().sum() > 0

    # -- I/O -----------------------------------------------------------

    @staticmethod
    def _save_grid(grid: gpd.GeoDataFrame, grid_file: Path) -> None:
        """Save the grid as Shapefile, CSV, and GeoJSON.

        All three are written on every save: Shapefile for legacy GIS
        compatibility, CSV for quick tabular analysis without a GIS
        library, and GeoJSON for web/lightweight-GIS use. This is
        deliberate output redundancy for downstream consumers, not an
        accident of the pipeline's history.
        """
        grid.to_file(grid_file)

        csv_file = grid_file.with_suffix(".csv")
        grid.drop(columns=["geometry"]).to_csv(csv_file, index=False)

        geojson_file = grid_file.with_suffix(".geojson")
        grid.to_file(geojson_file, driver="GeoJSON")

    def save_grid_boundary(self, grid_file: Path) -> Path:
        """Save the dissolved union of all grid hexagons as a boundary GeoJSON.

        This is the primary hand-off artifact for downstream modules that
        need the exact analysis extent (e.g. to clip other datasets), as
        distinct from the per-cell enriched grid itself. May be a
        MultiPolygon if the underlying grid is not spatially contiguous.
        """
        safe_aoi_name = _sanitize_filename(self.aoi_name)
        boundary_file = grid_file.parent / f"{safe_aoi_name}_grid_boundary.geojson"

        grid = gpd.read_file(grid_file)
        if len(grid) == 0:
            logger.warning("No grid cells to build a boundary from")
            return boundary_file

        grid_union = grid.geometry.union_all() if hasattr(grid.geometry, "union_all") else grid.geometry.unary_union

        boundary_gdf = gpd.GeoDataFrame(
            {
                "name": [f"{safe_aoi_name}_grid_boundary"],
                "grid_cells": [len(grid)],
                "total_area_m2": [grid_union.area],
                "geometry": [grid_union],
            },
            crs=grid.crs,
        )
        boundary_gdf.to_file(boundary_file, driver="GeoJSON")
        logger.info(
            "Grid boundary saved: %s (%s, %.2f km2)",
            boundary_file.name, grid_union.geom_type, grid_union.area / 1e6,
        )
        return boundary_file

    # -- top-level entry point ------------------------------------------

    def run(self) -> Path:
        """Run the complete workflow, skipping steps whose outputs already exist."""
        logger.info("=== Starting workflow: %s ===", self.aoi_name)

        aoi_json, aoi_gpkg = self._aoi_output_paths()
        aoi_file = aoi_gpkg if aoi_gpkg.exists() else aoi_json
        if not self.force["aoi"] and aoi_file.exists():
            logger.info("[1/4] AOI OK (existing)")
        else:
            aoi_file = self.generate_aoi()

        grid_file = self._grid_file_path()
        if not self.force["grid"] and grid_file.exists():
            logger.info("[2/4] Grid OK (existing)")
        else:
            grid_file = self.generate_grid(aoi_file)

        logger.info("[3/4] Attaching raster data sources...")
        final_grid_file = self.attach_data_sources(grid_file)

        logger.info("[4/4] Computing Shannon diversity index and grid boundary...")
        grid = gpd.read_file(final_grid_file)
        if "LC_Shannon" not in grid.columns or self.force["shannon"]:
            grid = calculate_shannon_diversity(grid, cfg.PRECISION_CONFIG["default"])
            self._save_grid(grid, final_grid_file)

        self.save_grid_boundary(final_grid_file)

        logger.info("=== Workflow complete: %s -> %s ===", self.aoi_name, final_grid_file.name)
        return final_grid_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a hexagonal analysis grid for a country or city AOI."
    )
    parser.add_argument("--aoi-mode", choices=["country", "city"], required=True)
    parser.add_argument("--aoi-name", required=True, help="Country or city name")
    parser.add_argument("--data-dir", required=True, type=Path, help="Output data root directory")
    parser.add_argument(
        "--country-tif-dir", required=True, type=Path,
        help="Root of the pre-clipped per-country raster cache (see clip_country_rasters.py)",
    )
    parser.add_argument("--country-boundaries", required=True, type=Path, help="Country boundary file")
    parser.add_argument("--country-name-column", default="NAME_0")
    parser.add_argument("--city-boundaries", required=True, type=Path, help="City/agglomeration boundary file")
    parser.add_argument("--city-name-column", default="Agglomeration_Name")
    parser.add_argument("--grid-resolution", type=float, default=cfg.GRID_RESOLUTION_M)
    parser.add_argument("--buffer-km", type=float, default=0.0)
    parser.add_argument("--target-crs", default=None, help="e.g. EPSG:32630; auto-detected if omitted")
    parser.add_argument("--country-iso3", default=None)
    parser.add_argument("--city-index", type=int, default=None)
    parser.add_argument("--force-regenerate-aoi", action="store_true")
    parser.add_argument("--force-regenerate-grid", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    boundaries = BoundariesConfig(
        country=BoundaryConfig(args.country_boundaries, args.country_name_column),
        city=BoundaryConfig(args.city_boundaries, args.city_name_column),
    )

    pipeline = GridPipeline(
        aoi_name=args.aoi_name,
        aoi_mode=args.aoi_mode,
        data_dir=args.data_dir,
        country_tif_dir=args.country_tif_dir,
        boundaries=boundaries,
        grid_resolution_m=args.grid_resolution,
        buffer_km=args.buffer_km,
        target_crs=args.target_crs,
        country_iso3=args.country_iso3,
        city_index=args.city_index,
        force_regenerate={"aoi": args.force_regenerate_aoi, "grid": args.force_regenerate_grid},
    )

    try:
        result_file = pipeline.run()
    except Exception as exc:
        logger.error("Workflow failed: %s", exc)
        return 1

    print(f"Final grid file: {result_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
