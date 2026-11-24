"""Run the published grid and morphology projects together for real Ndioum.

From the morphology repository root, after installing both projects:
    python -m examples.reproduce_ndioum_full \
        --grid-project /path/to/africa_hex_grid_workflow \
        --output-dir outputs/exports/YYYY-MM-DD/ndioum_full_chain

All data inputs are bundled under data/example/Ndioum_Senegal/inputs/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .reproduce_ndioum import INPUTS, reproduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-project', type=Path, required=True,
                        help='Root of the published africa_hex_grid_workflow repository.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    grid_project = args.grid_project.resolve()
    if not (grid_project / 'africa_hex_grid_workflow' / 'pipeline.py').is_file():
        parser.error('--grid-project must point to the published grid project root')

    source = INPUTS / 'grid_source'
    output = args.output_dir.resolve()
    grid_data = output / 'grid_data'
    command = [
        sys.executable, '-m', 'africa_hex_grid_workflow.pipeline',
        '--aoi-mode', 'city', '--aoi-name', 'Ndioum',
        '--data-dir', str(grid_data),
        '--country-tif-dir', str(source / 'country_tif'),
        '--country-boundaries', str(source / 'Senegal_boundary.geojson'),
        '--city-boundaries', str(source / 'Ndioum_aoi.geojson'),
        '--country-iso3', 'SEN', '--city-index', '54',
        '--target-crs', 'EPSG:32628', '--log-level', 'WARNING',
    ]
    subprocess.run(command, cwd=grid_project, check=True)
    grid_shp = grid_data / '03_country_grids' / 'Senegal' / 'Ndioum' / 'Ndioum_grid_250m_hex_enriched.shp'
    csv_path, shp_path = reproduce(output / 'morphology', grid_shp)
    print(f'Grid: {grid_shp}')
    print(f'CSV: {csv_path}')
    print(f'Shapefile: {shp_path}')
    print(f'Checks: {output / "morphology" / "verification.json"}')


if __name__ == '__main__':
    main()
