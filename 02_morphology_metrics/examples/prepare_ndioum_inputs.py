"""Recreate the small reviewer input bundle from the original project data.

This provenance utility is for maintainers who have the original project
data. Reviewers can use the checked-in bundle directly.
"""

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform


def copy_shapefile(source: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        shutil.copy2(source.with_suffix(suffix), target_dir / source.with_suffix(suffix).name)


def prepare(args):
    output = args.output_dir
    aoi = gpd.read_file(args.aoi_file)
    grid_source = output / 'grid_source'
    raster_output = grid_source / 'country_tif' / 'Senegal'
    raster_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.aoi_file, grid_source / 'Ndioum_aoi.geojson')

    country = gpd.read_file(args.country_boundaries, where="ISO = 'SEN'")
    if len(country) != 1:
        raise ValueError(f'Expected exactly one Senegal boundary; found {len(country)}')
    country.to_file(grid_source / 'Senegal_boundary.geojson', driver='GeoJSON')

    for source in sorted(args.country_tif_dir.glob('*.tif')):
        with rasterio.open(source) as src:
            bounds = aoi.to_crs(src.crs).total_bounds
            margin = 3 * max(src.res)
            window = from_bounds(
                bounds[0] - margin, bounds[1] - margin,
                bounds[2] + margin, bounds[3] + margin,
                transform=src.transform,
            ).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            profile = src.profile.copy()
            profile.update(width=int(window.width), height=int(window.height),
                           transform=window_transform(window, src.transform))
            with rasterio.open(raster_output / source.name, 'w', **profile) as target:
                target.write(src.read(window=window))

    if len(list(raster_output.glob('*.tif'))) != 14:
        raise ValueError('The original Senegal raster cache must contain 14 TIF files')

    copy_shapefile(args.grid_city_dir / 'Ndioum_grid_250m_hex_enriched.shp', output / 'grid')
    shutil.copy2(args.grid_city_dir / 'Ndioum_grid_boundary.geojson',
                 output / 'grid' / 'Ndioum_grid_boundary.geojson')
    copy_shapefile(args.metrix_city_dir / 'Output' / 'open_buildings.shp', output / 'buildings')
    highways = output / 'highways'
    highways.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metrix_city_dir / 'highways' / 'Ndioum_highways.geojson',
                 highways / 'Ndioum_highways.geojson')
    print(f'Prepared Ndioum inputs in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aoi-file', type=Path, required=True)
    parser.add_argument('--country-boundaries', type=Path, required=True)
    parser.add_argument('--country-tif-dir', type=Path, required=True)
    parser.add_argument('--grid-city-dir', type=Path, required=True)
    parser.add_argument('--metrix-city-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
