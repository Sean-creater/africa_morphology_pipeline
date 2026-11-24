#!/usr/bin/env python3
"""Download Sentinel-2 RGB satellite imagery for sampled hexagonal grid cells.

For each grid cell, builds a cloud-filtered median composite from the
`COPERNICUS/S2_SR_HARMONIZED` collection via Google Earth Engine, scales it
to 8-bit RGB, and saves it as a PNG named after the cell's GridID.

Earth Engine authentication: this module calls `ee.Initialize()` (optionally
with a `--project-id`), which uses whatever credentials are already present
in the environment via the standard interactive `earthengine authenticate`
/ `ee.Authenticate()` flow (typically stored under `~/.config/earthengine/`).
It does not read or embed a service-account key; if `ee.Initialize()` raises
because the user has never authenticated, it prints the standard
interactive-authentication command rather than falling back to any
credential file.

This module is normally used as a library by `render.py`, which downloads
one buffered image per cell and reuses it for all three rendered output
styles, but can also be run standalone to pre-populate a satellite-imagery
cache (`--output-dir`, default: `png_satellite_imagery/{City}/{GridID}.png`).
"""

import sys
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import ee
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    import requests
    from rasterio.io import MemoryFile
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    print("Install: pip install earthengine-api geopandas rasterio requests shapely pillow pandas")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Sentinel-2 Surface Reflectance (Harmonized), the Earth Engine collection
# this pipeline reads imagery from.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = {"red": "B4", "green": "B3", "blue": "B2", "scl": "SCL"}
# Scene Classification Layer values masked out: 3 = cloud shadow,
# 8/9 = cloud (medium/high probability), 10 = thin cirrus.
SCL_CLOUD_MASK_VALUES = [3, 8, 9, 10]

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-06-30"
DEFAULT_MAX_CLOUD_PERCENT = 30.0
DEFAULT_RESOLUTION_M = 10
DEFAULT_BUFFER_M = 20.0  # see render.py docstring for why a buffer is needed

# ============================================================================
# Data classes
# ============================================================================


@dataclass
class GridCell:
    """A single hexagonal grid cell to download imagery for."""
    grid_id: str
    city: str
    country: str
    crs: str
    center_x: float
    center_y: float
    geometry: object  # shapely geometry, in `crs`


@dataclass
class DownloadResult:
    grid_id: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


# ============================================================================
# Earth Engine authentication
# ============================================================================


class EarthEngineAuth:
    """Thin wrapper around interactive/environment-based Earth Engine auth."""

    @staticmethod
    def authenticate(project_id: Optional[str] = None) -> str:
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
        except Exception as e:
            error_msg = str(e)
            if "Please authorize" in error_msg:
                print("\nEarth Engine is not authenticated. Run:")
                print("  python -c \"import ee; ee.Authenticate(auth_mode='notebook')\"")
                raise
            elif "not registered" in error_msg and project_id:
                print(f"Note: project {project_id} not accessible, falling back to default project")
                ee.Initialize()
            else:
                raise
        return project_id or "default"


# ============================================================================
# Downloader
# ============================================================================


class GridCellImageDownloader:
    """Downloads a cloud-filtered Sentinel-2 RGB composite per grid cell."""

    def __init__(self, output_dir: Path,
                 start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE,
                 max_cloud_percent: float = DEFAULT_MAX_CLOUD_PERCENT,
                 resolution_m: int = DEFAULT_RESOLUTION_M,
                 project_id: Optional[str] = None, num_workers: int = 4,
                 logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud_percent = max_cloud_percent
        self.resolution_m = resolution_m
        self.project_id = project_id
        self.num_workers = num_workers
        self.logger = logger or logging.getLogger(__name__)
        self.collection = None

    def setup_earth_engine(self):
        """Authenticate and bind the Sentinel-2 collection. Call once before downloading."""
        try:
            project_id = EarthEngineAuth.authenticate(self.project_id)
            self.collection = ee.ImageCollection(S2_COLLECTION)
            self.logger.info(f"Earth Engine initialized (project: {project_id})")
        except Exception as e:
            raise RuntimeError(f"Failed to set up Earth Engine: {e}")

    def _mask_clouds(self, image: 'ee.Image') -> 'ee.Image':
        """Mask cloud/cloud-shadow/cirrus pixels using the Scene Classification Layer."""
        scl = image.select('SCL')
        mask = scl.neq(SCL_CLOUD_MASK_VALUES[0])
        for val in SCL_CLOUD_MASK_VALUES[1:]:
            mask = mask.And(scl.neq(val))
        return image.updateMask(mask)

    def _create_composite(self, geometry: 'ee.Geometry') -> 'ee.Image':
        """Build a per-pixel median RGB composite over the date range and cloud filter."""
        filtered = (
            self.collection
            .filterBounds(geometry)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.max_cloud_percent))
        )

        image_count = filtered.size().getInfo()
        if image_count == 0:
            raise RuntimeError(
                f"No imagery found for {self.start_date} to {self.end_date}. "
                f"Try extending the date range or increasing --max-cloud."
            )
        self.logger.debug(f"Found {image_count} candidate images")

        filtered = filtered.map(self._mask_clouds)
        rgb_bands = [S2_BANDS["red"], S2_BANDS["green"], S2_BANDS["blue"]]
        composite = filtered.select(rgb_bands).median()
        return composite.clip(geometry)

    @staticmethod
    def _scale_to_uint8(image: 'ee.Image') -> 'ee.Image':
        """Scale Sentinel-2 surface reflectance (0-10000 scaled by 1e4) to 0-255.

        Reflectance is clamped to [0, 0.3] before scaling: Sentinel-2's
        reflectance range is much wider than typical visible-light brightness,
        so clamping to a realistic reflectance ceiling before stretching to
        8-bit avoids most scenes coming out over-dark.
        """
        return image.multiply(0.0001).clamp(0, 0.3).multiply(255 / 0.3).uint8()

    def _download_to_geotiff(self, image: 'ee.Image', geometry: 'ee.Geometry') -> bytes:
        image = self._scale_to_uint8(image)
        url = image.getDownloadURL({
            'scale': self.resolution_m,
            'crs': "EPSG:4326",
            'region': geometry,
            'format': 'GEO_TIFF',
            'filePerBand': False,
        })
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _geotiff_to_png(geotiff_bytes: bytes, output_path: Path, logger: logging.Logger) -> Path:
        with MemoryFile(geotiff_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()
                if data.shape[0] != 3:
                    raise ValueError(f"Expected 3 bands, got {data.shape[0]}")
                rgb = np.transpose(data, (1, 2, 0))

                img = Image.fromarray(rgb.astype('uint8'), 'RGB')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, 'PNG', optimize=False, compress_level=0)
                logger.debug(f"Saved {output_path.name} ({img.size[0]}x{img.size[1]})")
        return output_path

    def download_grid_cell(self, grid_cell: GridCell, buffer_m: float = DEFAULT_BUFFER_M) -> DownloadResult:
        """Download imagery for one grid cell, buffered outward by `buffer_m`.

        The buffer extends the download region beyond the hexagon boundary
        (see render.py for why this matters at render time); the buffered
        geometry, not the raw hexagon, is what is sent to Earth Engine.
        """
        try:
            gdf = gpd.GeoDataFrame({'geometry': [grid_cell.geometry]}, crs=grid_cell.crs)
            gdf_buffered = gdf.copy()
            gdf_buffered['geometry'] = gdf_buffered.geometry.buffer(buffer_m)

            gdf_wgs84 = gdf_buffered.to_crs("EPSG:4326")
            geom_wgs84 = gdf_wgs84.geometry.iloc[0]
            ee_geometry = ee.Geometry(geom_wgs84.__geo_interface__)

            composite = self._create_composite(ee_geometry)
            geotiff_bytes = self._download_to_geotiff(composite, ee_geometry)
            output_path = self.output_dir / grid_cell.city / f"{grid_cell.grid_id}.png"
            self._geotiff_to_png(geotiff_bytes, output_path, self.logger)

            return DownloadResult(grid_id=grid_cell.grid_id, success=True, output_path=output_path)
        except Exception as e:
            self.logger.error(f"Failed {grid_cell.grid_id}: {e}")
            return DownloadResult(grid_id=grid_cell.grid_id, success=False, error=str(e))

    def batch_download(self, grid_cells: List[GridCell]) -> Dict[str, List[DownloadResult]]:
        """Download imagery for multiple grid cells in parallel."""
        results = {'succeeded': [], 'failed': []}
        self.logger.info(f"Processing {len(grid_cells)} grid cells with {self.num_workers} workers")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_cell = {
                executor.submit(self.download_grid_cell, cell): cell
                for cell in grid_cells
            }
            for future in as_completed(future_to_cell):
                result = future.result()
                if result.success:
                    results['succeeded'].append(result)
                    self.logger.info(f"  succeeded: {result.grid_id}")
                else:
                    results['failed'].append(result)
                    self.logger.error(f"  failed: {result.grid_id}: {result.error}")
        return results


# ============================================================================
# Data loading
# ============================================================================


class GridCellLoader:
    """Loads GridCell objects from the sampled-cell shapefiles and metadata CSV."""

    def __init__(self, shp_dir: Path, csv_file: Path, logger: Optional[logging.Logger] = None):
        self.shp_dir = shp_dir
        self.csv_file = csv_file
        self.logger = logger or logging.getLogger(__name__)
        self.metadata_df = None

    def load_metadata(self) -> pd.DataFrame:
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_file}")
        self.metadata_df = pd.read_csv(self.csv_file)
        self.logger.info(f"Loaded metadata for {len(self.metadata_df)} grid cells")
        return self.metadata_df

    def load_city_grid_cells(self, city: str) -> List[GridCell]:
        shp_file = self.shp_dir / f"{city}_sampled_cells.shp"
        if not shp_file.exists():
            raise FileNotFoundError(f"Shapefile not found: {shp_file}")

        gdf = gpd.read_file(shp_file)
        self.logger.info(f"Loaded {len(gdf)} grid cells from {shp_file.name}")

        if self.metadata_df is None:
            self.load_metadata()
        city_meta = self.metadata_df[self.metadata_df['City'] == city]

        grid_cells = []
        for idx, row in gdf.iterrows():
            grid_id = None
            for col in ['GridID', 'grid_id', 'GRID_ID', 'id', 'ID']:
                if col in row and pd.notna(row[col]):
                    grid_id = str(row[col])
                    break
            if not grid_id:
                grid_id = city_meta.iloc[idx]['GridID'] if idx < len(city_meta) else f"{city}_{idx:06d}"

            meta_row = city_meta[city_meta['GridID'] == grid_id]
            if len(meta_row) == 0:
                crs, country = str(gdf.crs), "Unknown"
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
            else:
                meta = meta_row.iloc[0]
                crs, country = meta['CRS'], meta['Country']
                center_x, center_y = meta['Center_X'], meta['Center_Y']

            grid_cells.append(GridCell(
                grid_id=grid_id, city=city, country=country, crs=crs,
                center_x=center_x, center_y=center_y, geometry=row.geometry,
            ))
        return grid_cells

    def get_available_cities(self) -> List[str]:
        return sorted(
            shp_file.stem.replace("_sampled_cells", "")
            for shp_file in self.shp_dir.glob("*_sampled_cells.shp")
        )


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description='Download Sentinel-2 imagery for sampled grid cells')
    parser.add_argument('--shp-dir', type=Path, required=True, help='Directory of {City}_sampled_cells.shp')
    parser.add_argument('--csv-file', type=Path, required=True, help='SampleGridCell.csv metadata')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output root for {City}/{GridID}.png')
    parser.add_argument('--city', type=str, help='Restrict to a single city (default: all)')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE)
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE)
    parser.add_argument('--max-cloud', type=float, default=DEFAULT_MAX_CLOUD_PERCENT)
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION_M)
    parser.add_argument('--buffer-m', type=float, default=DEFAULT_BUFFER_M)
    parser.add_argument('--project-id', type=str, default=None, help='Earth Engine cloud project id')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    loader = GridCellLoader(args.shp_dir, args.csv_file, logger)
    cities = [args.city] if args.city else loader.get_available_cities()
    logger.info(f"Cities to process: {len(cities)}")

    downloader = GridCellImageDownloader(
        output_dir=args.output_dir, start_date=args.start_date, end_date=args.end_date,
        max_cloud_percent=args.max_cloud, resolution_m=args.resolution,
        project_id=args.project_id, num_workers=args.workers, logger=logger,
    )
    downloader.setup_earth_engine()

    total_succeeded = total_failed = 0
    for city in cities:
        logger.info(f"Processing: {city}")
        try:
            grid_cells = loader.load_city_grid_cells(city)
            results = downloader.batch_download(grid_cells)
            total_succeeded += len(results['succeeded'])
            total_failed += len(results['failed'])
        except Exception as e:
            logger.error(f"Failed to process {city}: {e}")
            continue

    logger.info(f"Done. Succeeded: {total_succeeded}, failed: {total_failed}")


if __name__ == "__main__":
    main()
