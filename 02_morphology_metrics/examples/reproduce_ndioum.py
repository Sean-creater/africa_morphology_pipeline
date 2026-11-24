"""Recompute Ndioum's morphology table from the bundled, real city inputs.

Run from the repository root after installing its environment:
    python -m examples.reproduce_ndioum --output-dir outputs/exports/YYYY-MM-DD/ndioum

The script runs entirely offline: building footprints and clipped OSM roads
are bundled so that the input-to-output metric calculation can be reviewed
without re-downloading mutable upstream datasets.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from africa_morphology_metrics.morphology_metrics import (
    calculate_building_metrics, aggregate_metrics_to_grid,
)
from africa_morphology_metrics.spatial_distribution_metrics import calculate_and_aggregate_spatial_metrics
from africa_morphology_metrics.streetnetwork_metrics import calculate_street_network_metrics_with_graph
from africa_morphology_metrics.connectivity import calculate_connectivity_metrics


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'data' / 'example' / 'Ndioum_Senegal'
INPUTS = EXAMPLE / 'inputs'
EXPECTED = EXAMPLE / 'Ndioum_grid_250m_hex_enriched_metrics_v15.csv'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reproduce(output_dir: Path, grid_shp: Path | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = gpd.read_file(grid_shp or INPUTS / 'grid' / 'Ndioum_grid_250m_hex_enriched.shp')
    buildings = gpd.read_file(INPUTS / 'buildings' / 'open_buildings.shp').to_crs(grid.crs)
    roads = gpd.read_file(INPUTS / 'highways' / 'Ndioum_highways.geojson')
    crs = grid.crs.to_string()

    buildings = calculate_building_metrics(
        buildings, n_jobs=1,
        height_raster_path=INPUTS / 'grid_source' / 'country_tif' / 'Senegal' / 'building_height.tif',
        volume_raster_path=INPUTS / 'grid_source' / 'country_tif' / 'Senegal' / 'building_volume.tif',
    )
    grid = aggregate_metrics_to_grid(buildings, grid, grid_id_col='GridID')
    buildings, grid = calculate_and_aggregate_spatial_metrics(
        buildings, grid, grid_id_col='GridID', target_crs=crs,
        k=5, orientation_col='Orientation', n_jobs=1,
    )
    if 'GridID' not in buildings.columns:
        assigned = gpd.sjoin(buildings, grid[['GridID', 'geometry']],
                             predicate='within', how='left')
        buildings['GridID'] = assigned['GridID'].values
    grid, nodes, edges, graph = calculate_street_network_metrics_with_graph(
        roads=roads, cells=grid, buildings=buildings,
        cell_id_col='GridID', area_col='Area_m2', target_crs=crs, n_jobs=1,
    )
    grid = calculate_connectivity_metrics(
        grid=grid, nodes_gdf=nodes, edges_gdf=edges, graph=graph,
        spatial_graphs_pkl=output_dir / 'subgraphs.pkl', grid_id_col='GridID',
        skip_coins=False, batch_size=100,
    )

    expected_columns = pd.read_csv(EXPECTED, nrows=0).columns.tolist()
    actual_columns = [c for c in grid.columns if c != 'geometry']
    missing = sorted(set(expected_columns) - set(actual_columns))
    extra = sorted(set(actual_columns) - set(expected_columns))
    if missing or extra or len(grid) != 33 or not grid['GridID'].is_unique:
        raise RuntimeError(
            f'Unexpected output schema: rows={len(grid)}, missing={missing}, extra={extra}'
        )
    grid = grid[expected_columns + ['geometry']]
    csv_path = output_dir / 'Ndioum_grid_250m_hex_enriched_metrics_v15.csv'
    shp_path = csv_path.with_suffix('.shp')
    grid.drop(columns='geometry').to_csv(csv_path, index=False)
    grid.to_file(shp_path)

    output_table = pd.read_csv(csv_path).sort_values('GridID').reset_index(drop=True)
    reference = pd.read_csv(EXPECTED).sort_values('GridID').reset_index(drop=True)
    report = {
        'city': 'Ndioum, Senegal',
        'grid_cells': len(grid),
        'attribute_columns': len(actual_columns),
        'grid_ids_match_reference': output_table.GridID.equals(reference.GridID),
        'reference_csv_sha256': sha256(EXPECTED),
        'output_csv_sha256': sha256(csv_path),
        'used_grid_shp_sha256': sha256(grid_shp or INPUTS / 'grid' / 'Ndioum_grid_250m_hex_enriched.shp'),
        'input_files_sha256': {
            str(p.relative_to(EXAMPLE)): sha256(p)
            for p in sorted(INPUTS.rglob('*')) if p.is_file()
        },
    }
    numeric_columns = [c for c in expected_columns
                       if pd.api.types.is_numeric_dtype(reference[c])]
    differing = [c for c in numeric_columns
                 if not np.allclose(reference[c], output_table[c],
                                    equal_nan=True, rtol=1e-8, atol=1e-8)]
    report['numeric_columns_compared'] = len(numeric_columns)
    report['numeric_columns_differing'] = differing
    if differing or not report['grid_ids_match_reference']:
        raise RuntimeError(f'Reference comparison failed: {differing}')
    (output_dir / 'verification.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    )
    return csv_path, shp_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--grid-shp', type=Path,
                        help='Optional 49-column grid from africa_hex_grid_workflow; '
                             'defaults to the bundled historical Ndioum grid.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    csv_path, shp_path = reproduce(args.output_dir, args.grid_shp)
    print(f'CSV: {csv_path}')
    print(f'Shapefile: {shp_path}')
    print('Verified: 33 grid cells, 172 attribute columns, unique GridID')
    print(f'Report: {args.output_dir / "verification.json"}')


if __name__ == '__main__':
    main()
