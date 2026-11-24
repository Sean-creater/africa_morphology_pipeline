#!/usr/bin/env python3
"""Clip buildings/roads to sampled grid cells and render the two output styles.

For each sampled hexagonal grid cell, clips that city's building footprints
and road network to the cell boundary and renders two PNG styles:

1. `png_street+building/{City}/{GridID}.png` - white hexagon background,
   buildings and roads only, no satellite. A lightweight reference render
   that does not require Earth Engine access.

2. `png_integrated/{City}/{GridID}.png` - Sentinel-2 satellite background
   (via `satellite_download.py`) with a black/dark-gray building+road
   overlay. Used both as DINOv2/DINOv3 training input and, unmodified, as
   the image shown to human experts for similarity annotation (see
   `annotation_pairs/`): the muted, low-contrast overlay leaves the
   satellite texture as the dominant visual signal.

Rendering decisions (all deliberate, not incidental):

- **20 m buffer on satellite downloads.** Earth Engine's per-cell GeoTIFF
  export can blur or clip pixels right at the requested region's edge.
  Downloading a region 20 m larger than the hexagon on every side and then
  cropping the *rendered* extent back down to the exact hexagon boundary
  (`ax.set_xlim`/`set_ylim` on the original geometry, not the buffered one)
  means the visible pixels always come from the interior of the downloaded
  tile, never from its own edge.
- **Antialiasing disabled** (`matplotlib.rcParams['path.simplify'] = False`,
  `antialiased=False` on the vector layers) so building and road edges stay
  crisp single-color pixels rather than being softened into intermediate
  colors — important both for a clean training signal and for legible
  annotation images at small display sizes.
- **Z-ordering: satellite (1) below roads (2) below buildings (3).**
  Buildings sit on top of roads because a building silhouette fully
  occluding a road segment beneath it is what a top-down satellite view
  actually looks like; rendering roads on top would draw thin lines through
  buildings that do not exist in reality.
- **Road width 3.5** (vs. the thinner 1.5 used in the satellite-free
  `png_street+building` style) because a 1.5-point line becomes nearly
  invisible once it competes with real satellite texture in the background.
- **Hexagon alpha mask applied after rendering**, not before: the vector
  overlay and satellite backdrop are rendered onto a bounding-box canvas,
  then a rasterio-derived hexagon mask zeroes the alpha channel outside the
  true hexagon boundary. Rendering directly onto a hexagon-clipped canvas
  would reintroduce the same edge artifacts the 20 m buffer is meant to
  avoid.
"""

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Disable matplotlib path simplification/antialiasing for crisp vector edges.
matplotlib.rcParams['path.simplify'] = False
matplotlib.rcParams['agg.path.chunksize'] = 0

from satellite_download import GridCellImageDownloader, GridCell

# ============================================================================
# Configuration
# ============================================================================

IMAGE_SIZE_INCHES = (10, 10)
DPI = 100

INTEGRATED_BUILDING_COLOR = '#000000'
INTEGRATED_ROAD_COLOR = '#222222'

# png_street+building style (no satellite; unchanged regardless of scheme).
BUILDING_COLOR_STREET_BUILDING = '#000000'
ROAD_COLOR_STREET_BUILDING = '#555555'
ROAD_WIDTH_STREET_BUILDING = 1.5
HEX_BACKGROUND_COLOR = 'white'

# Integrated (satellite-backed) styles.
ROAD_WIDTH_INTEGRATED = 3.5

DEFAULT_SATELLITE_BUFFER_M = 20.0
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-06-30"
DEFAULT_MAX_CLOUD_PERCENT = 30.0


# ============================================================================
# Data loading helpers
# ============================================================================


def load_city_config(city_config_json: Path):
    if not city_config_json or not city_config_json.exists():
        return None
    with open(city_config_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_city_range(config, range_name):
    if config is None or range_name == 'all':
        return (0, None)
    if range_name not in config['ranges']:
        return (0, None)
    range_config = config['ranges'][range_name]
    return (range_config['start_index'], range_config['end_index'])


def load_city_paths(city_path_csv: Path, start_idx=0, end_idx=None):
    """
    Load the city-path CSV. Path columns (Buildings_Path, Highways_Path,
    Grid_Path) may contain {country}/{city} template placeholders (as in
    data/reference/SampleCityPath.csv); these are expanded per-row using
    that row's own City/Country columns. Rows with already-literal paths
    are left unchanged.
    """
    df = pd.read_csv(city_path_csv)
    df = df.iloc[start_idx:end_idx] if end_idx is not None else df.iloc[start_idx:]
    path_cols = [c for c in ("Buildings_Path", "Highways_Path", "Grid_Path") if c in df.columns]
    for col in path_cols:
        df[col] = [
            p.format(country=row["Country"], city=row["City"]) if "{country}" in p or "{city}" in p else p
            for p, (_, row) in zip(df[col], df.iterrows())
        ]
    return df


def load_metadata(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_file}")
    return pd.read_csv(csv_file)


def clip_to_cell(gdf: gpd.GeoDataFrame, cell_geometry) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to a cell boundary, returning only the intersecting geometry."""
    try:
        clipped = gdf[gdf.intersects(cell_geometry)].copy()
        if len(clipped) > 0:
            clipped['geometry'] = clipped.geometry.intersection(cell_geometry)
            clipped = clipped[~clipped.geometry.is_empty]
        return clipped
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)


# ============================================================================
# Rendering
# ============================================================================


def render_street_building(cell_geom, buildings_clipped, highways_clipped, output_path, target_crs):
    """Render the satellite-free style: white hexagon, buildings + roads only."""
    fig, ax = plt.subplots(figsize=IMAGE_SIZE_INCHES, dpi=DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    bounds = cell_geom.bounds

    gpd.GeoSeries([cell_geom], crs=target_crs).plot(
        ax=ax, facecolor=HEX_BACKGROUND_COLOR, edgecolor='none', alpha=1.0,
    )
    if len(buildings_clipped) > 0:
        buildings_clipped.plot(
            ax=ax, color=BUILDING_COLOR_STREET_BUILDING, edgecolor='none', alpha=1.0, linewidth=0,
        )
    if len(highways_clipped) > 0:
        highways_clipped.plot(
            ax=ax, color=ROAD_COLOR_STREET_BUILDING, linewidth=ROAD_WIDTH_STREET_BUILDING, alpha=1.0,
        )

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def render_integrated(cell_geom, buildings_clipped, highways_clipped,
                       satellite_img, satellite_bounds, output_path, target_crs):
    """Render the satellite-backed style: black/dark-gray building+road overlay.

    `satellite_bounds` is the buffered download extent (see module
    docstring); the axis limits below use the original `cell_geom` bounds
    instead, which crops the buffered satellite tile back down to the exact
    hexagon bounding box before the hexagon alpha mask is applied.
    """
    fig, ax = plt.subplots(figsize=IMAGE_SIZE_INCHES, dpi=DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    cell_bounds = cell_geom.bounds

    if satellite_img is not None and satellite_bounds is not None:
        sat_array = np.array(satellite_img)
        sat_extent = [satellite_bounds[0], satellite_bounds[2], satellite_bounds[1], satellite_bounds[3]]
        ax.imshow(sat_array, extent=sat_extent, aspect='auto', interpolation='bilinear', zorder=1)

    # Roads first (zorder=2): must sit below buildings.
    if len(highways_clipped) > 0:
        highways_clipped.plot(
            ax=ax, color=INTEGRATED_ROAD_COLOR, linewidth=ROAD_WIDTH_INTEGRATED,
            alpha=1.0, zorder=2, antialiased=False,
        )
    # Buildings on top (zorder=3), sharp edges.
    if len(buildings_clipped) > 0:
        buildings_clipped.plot(
            ax=ax, color=INTEGRATED_BUILDING_COLOR, edgecolor='none',
            alpha=1.0, linewidth=0, zorder=3, antialiased=False,
        )

    ax.set_xlim(cell_bounds[0], cell_bounds[2])
    ax.set_ylim(cell_bounds[1], cell_bounds[3])
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='png', dpi=DPI, bbox_inches='tight',
                pad_inches=0, transparent=True, facecolor='none')
    plt.close(fig)

    apply_hexagon_mask(output_path, cell_geom, cell_bounds)


def apply_hexagon_mask(image_path, cell_geom, bounds):
    """Zero the alpha channel outside the true hexagon boundary."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    img = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    x_min, y_min, x_max, y_max = bounds

    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    mask = geometry_mask([cell_geom], transform=transform, invert=True, out_shape=(height, width))
    img_array[~mask, 3] = 0

    Image.fromarray(img_array, 'RGBA').save(image_path, 'PNG', optimize=False, compress_level=0)


def download_satellite_for_cell(grid_cell, downloader, logger, buffer_m=DEFAULT_SATELLITE_BUFFER_M):
    """Download the buffered satellite tile for one cell; returns (PIL.Image, bounds) or (None, None)."""
    try:
        gdf = gpd.GeoDataFrame({'geometry': [grid_cell.geometry]}, crs=grid_cell.crs)
        buffered_bounds = gdf.geometry.iloc[0].buffer(buffer_m).bounds

        result = downloader.download_grid_cell(grid_cell, buffer_m=buffer_m)
        if result.success and result.output_path.exists():
            return Image.open(result.output_path).convert('RGB'), buffered_bounds
        logger.warning(f"  Satellite download failed for {grid_cell.grid_id}: {result.error}")
        return None, None
    except Exception as e:
        logger.warning(f"  Satellite error for {grid_cell.grid_id}: {e}")
        return None, None


# ============================================================================
# Per-city processing
# ============================================================================


def process_city(city_name, country_name, buildings_path, highways_path,
                  output_dirs, metadata_df, downloader=None, logger=None):
    """Render both PNG styles for every sampled cell of one city."""
    logger = logger or logging.getLogger(__name__)
    output_dir_street_building, output_dir_integrated, grid_cells_dir = output_dirs

    logger.info(f"Processing: {city_name}, {country_name}")

    grid_shp_path = grid_cells_dir / f"{city_name}_sampled_cells.shp"
    if not grid_shp_path.exists():
        logger.error(f"Grid shapefile not found: {grid_shp_path}")
        return 0
    grid_cells = gpd.read_file(grid_shp_path)

    city_metadata = metadata_df[metadata_df['City'] == city_name]
    if len(city_metadata) > 0 and 'Stratum_ID' in city_metadata.columns:
        grid_cells = grid_cells.merge(city_metadata[['GridID', 'Stratum_ID']], on='GridID', how='left')

    target_crs = grid_cells.crs.to_string()

    if not Path(buildings_path).exists():
        logger.error(f"Buildings file not found: {buildings_path}")
        return 0
    buildings = gpd.read_file(buildings_path)
    if buildings.crs.to_string() != target_crs:
        buildings = buildings.to_crs(target_crs)

    if not Path(highways_path).exists():
        logger.error(f"Highways file not found: {highways_path}")
        return 0
    highways = gpd.read_file(highways_path)
    if highways.crs.to_string() != target_crs:
        highways = highways.to_crs(target_crs)

    city_output_street_building = output_dir_street_building / city_name
    city_output_integrated = output_dir_integrated / city_name
    for d in (city_output_street_building, city_output_integrated):
        d.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for idx, row in tqdm(grid_cells.iterrows(), total=len(grid_cells), desc=f"  {city_name}"):
        grid_id = row['GridID']
        cell_geom = row['geometry']

        stratum_id = row.get('Stratum_ID', -1)
        filename = f"{grid_id}_{int(stratum_id)}.png" if stratum_id >= 0 else f"{grid_id}.png"

        output_street_building = city_output_street_building / filename
        output_integrated = city_output_integrated / filename

        if output_street_building.exists() and output_integrated.exists():
            processed_count += 1
            continue

        try:
            buildings_clipped = clip_to_cell(buildings, cell_geom)
            highways_clipped = clip_to_cell(highways, cell_geom)

            if not output_street_building.exists():
                render_street_building(cell_geom, buildings_clipped, highways_clipped,
                                        output_street_building, target_crs)

            if not output_integrated.exists() and downloader is not None:
                meta_row = metadata_df[metadata_df['GridID'] == grid_id]
                if len(meta_row) > 0:
                    meta = meta_row.iloc[0]
                    grid_cell_obj = GridCell(
                        grid_id=grid_id, city=city_name, country=country_name, crs=meta['CRS'],
                        center_x=meta['Center_X'], center_y=meta['Center_Y'], geometry=cell_geom,
                    )
                    sat_img, sat_bounds = download_satellite_for_cell(
                        grid_cell_obj, downloader, logger, buffer_m=DEFAULT_SATELLITE_BUFFER_M,
                    )
                    render_integrated(cell_geom, buildings_clipped, highways_clipped,
                                       sat_img, sat_bounds, output_integrated, target_crs)
                else:
                    logger.warning(f"  Metadata not found for {grid_id}")

            processed_count += 1
        except Exception as e:
            logger.error(f"  Error processing {grid_id}: {e}")
            continue

    logger.info(f"Processed {processed_count} cells for {city_name}")
    return processed_count


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Clip buildings/roads to sampled grid cells and render the 2 output styles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--city-path-csv', type=Path, required=True,
                         help='CSV with City, Country, Buildings_Path, Highways_Path, All_Files_Present columns')
    parser.add_argument('--city-config-json', type=Path, default=None,
                         help='Optional JSON defining named city index ranges (for --range)')
    parser.add_argument('--grid-cells-dir', type=Path, required=True,
                         help='Directory of {City}_sampled_cells.shp (sampling.py Stage 2 output)')
    parser.add_argument('--metadata-csv', type=Path, required=True, help='SampleGridCell.csv')
    parser.add_argument('--output-dir', type=Path, required=True,
                         help='Root output directory; the 3 style subdirectories are created under it')
    parser.add_argument('--range', type=str, default='all', help='Named range from --city-config-json, or "all"')
    parser.add_argument('--start', type=int, help='Custom start index (overrides --range)')
    parser.add_argument('--end', type=int, help='Custom end index (overrides --range)')
    parser.add_argument('--no-satellite', action='store_true', help='Skip satellite download (renders style 1 only)')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE)
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE)
    parser.add_argument('--max-cloud', type=float, default=DEFAULT_MAX_CLOUD_PERCENT)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    config = load_city_config(args.city_config_json)
    if args.start is not None or args.end is not None:
        start_idx, end_idx = args.start or 0, args.end
    else:
        start_idx, end_idx = get_city_range(config, args.range)

    metadata_df = load_metadata(args.metadata_csv)
    logger.info(f"Loaded metadata for {len(metadata_df)} grid cells")

    downloader = None
    if not args.no_satellite:
        try:
            logger.info("Initializing Earth Engine...")
            downloader = GridCellImageDownloader(
                output_dir=args.output_dir / 'png_satellite_imagery',
                start_date=args.start_date, end_date=args.end_date,
                max_cloud_percent=args.max_cloud, num_workers=1, logger=logger,
            )
            downloader.setup_earth_engine()
            logger.info("Earth Engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Earth Engine: {e}; continuing without satellite imagery")

    output_dir_street_building = args.output_dir / 'png_street+building'
    output_dir_integrated = args.output_dir / 'png_integrated'
    for d in (output_dir_street_building, output_dir_integrated):
        d.mkdir(parents=True, exist_ok=True)

    city_paths = load_city_paths(args.city_path_csv, start_idx, end_idx)
    if 'All_Files_Present' in city_paths.columns:
        city_paths = city_paths[city_paths['All_Files_Present'] == 'Yes']
    logger.info(f"Processing {len(city_paths)} cities with complete data")

    total_cells, successful_cities, failed_cities = 0, 0, []
    for _, row in city_paths.iterrows():
        try:
            cells_processed = process_city(
                row['City'], row['Country'], row['Buildings_Path'], row['Highways_Path'],
                (output_dir_street_building, output_dir_integrated, args.grid_cells_dir),
                metadata_df, downloader, logger,
            )
            if cells_processed > 0:
                total_cells += cells_processed
                successful_cities += 1
            else:
                failed_cities.append(row['City'])
        except Exception as e:
            logger.error(f"Failed to process {row['City']}: {e}")
            failed_cities.append(row['City'])

    logger.info(f"Done. Successful cities: {successful_cities}/{len(city_paths)}, total cells: {total_cells}")
    if failed_cities:
        logger.info(f"Failed cities ({len(failed_cities)}): {', '.join(failed_cities)}")


if __name__ == '__main__':
    main()
