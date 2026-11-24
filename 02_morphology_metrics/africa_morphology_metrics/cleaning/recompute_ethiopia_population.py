"""
Recompute the Pop2022 column for Ethiopian cities directly from the WorldPop raster.

Ethiopia's per-city grid enrichment run used a population raster resampling
pass that produced unreliable Pop2022 values for this country's cities (the
rest of the continent was unaffected). This script re-derives Pop2022 from
the source population raster using the same resampling convention as the
main enrichment pipeline (``unified_grid_resampling``: sum of valid pixels
per cell, ``nodata=-99999``, no negative-value filtering, ``all_touched=True``),
so Ethiopia's values are produced by the identical method used for every
other country rather than by a one-off correction.

Only the Pop2022 column is overwritten; every other column in the city CSV
is left untouched.

Inputs
------
- Per-city grid geometry: ``<data_root>/Ethiopia/<city>/Output/<city><suffix>.shp``
- Per-city metrics CSV:   ``<data_root>/Ethiopia/<city>/Output/<city><suffix>.csv``
- Population raster: WorldPop 2020 UNadj constrained, reprojected to ESRI:102022
  (``population_2022.tif``)

Output
------
Overwrites Pop2022 in each city's metrics CSV in place.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask

SUFFIX = '_grid_250m_hex_enriched_metrics_v15'

# ── Resampling parameters (must match the main grid-enrichment pipeline's
#    HEXAGON_RESAMPLING_CONFIG / PRECISION_CONFIG / DTYPE_CONFIG for population) ──
NODATA_VALUES = {-99999.0}
ALL_TOUCHED = True
PRECISION = 2
OUT_DTYPE = 'float32'


def resample_pop(geom, src, raster_bounds) -> float | None:
    """Sum valid population pixels within a geometry. Returns None if no overlap."""
    gb = geom.bounds
    if (gb[2] < raster_bounds.left or gb[0] > raster_bounds.right or
            gb[3] < raster_bounds.bottom or gb[1] > raster_bounds.top):
        return None

    try:
        masked, _ = rasterio_mask(src, [geom], crop=True,
                                   nodata=src.nodata, all_touched=ALL_TOUCHED)
        data = masked[0] if masked.ndim > 2 else masked

        valid = np.ones(data.shape, dtype=bool)
        valid &= ~np.isnan(data)
        valid &= ~np.isinf(data)
        for nd in NODATA_VALUES:
            valid &= ~np.isclose(data, nd, rtol=1e-5, atol=1e-8)

        vals = data[valid]
        if len(vals) == 0:
            return None
        return float(np.sum(vals))
    except Exception:
        return None


def process_city(city_name: str, data_root: Path, src, raster_bounds, raster_crs) -> bool:
    city_dir = data_root / city_name / 'Output'
    shp_path = city_dir / f'{city_name}{SUFFIX}.shp'
    csv_path = city_dir / f'{city_name}{SUFFIX}.csv'

    if not shp_path.exists() or not csv_path.exists():
        print(f'  [SKIP] {city_name}: missing SHP or CSV')
        return False

    gdf = gpd.read_file(shp_path)
    gdf_proj = gdf.to_crs(raster_crs)

    pop_values = np.empty(len(gdf_proj), dtype=np.float32)
    pop_values[:] = np.nan

    for idx, geom in enumerate(gdf_proj.geometry.values):
        val = resample_pop(geom, src, raster_bounds)
        if val is not None:
            pop_values[idx] = round(val, PRECISION)

    df = pd.read_csv(csv_path, low_memory=False)

    # The grid SHP can carry one extra row relative to the CSV; align by
    # GridID in that case rather than assuming positional correspondence.
    if len(df) != len(gdf):
        print(f'  [WARN] {city_name}: CSV rows={len(df)}, SHP rows={len(gdf)} — aligning by GridID')
        shp_ids = gdf['GridID'].values if 'GridID' in gdf.columns else None
        if shp_ids is not None and 'GridID' in df.columns:
            id_to_pop = dict(zip(shp_ids, pop_values))
            df['Pop2022'] = df['GridID'].map(id_to_pop).astype('float32')
        else:
            print(f'  [ERROR] {city_name}: cannot align, skipping')
            return False
    else:
        df['Pop2022'] = pop_values

    df.to_csv(csv_path, index=False)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Recompute Pop2022 for Ethiopian cities from the WorldPop 2022 raster.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help="Country data root containing Ethiopia's per-city "
                              "Output/ directories (i.e. the parent of the "
                              "'Ethiopia' city folder), matching the working "
                              'pipeline layout: <data_root>/Ethiopia/<city>/Output/.')
    parser.add_argument('--population-tif', type=Path, required=True,
                         help='Path to the Ethiopia population_2022.tif raster '
                              '(WorldPop 2020 UNadj constrained, ESRI:102022).')
    args = parser.parse_args()

    ethiopia_dir = args.data_root / 'Ethiopia'
    cities = sorted(p.name for p in ethiopia_dir.iterdir())
    total = len(cities)
    print(f'Ethiopia cities to process: {total}')
    print(f'Population TIF: {args.population_tif}')
    print()

    success = 0
    failed = 0

    with rasterio.open(args.population_tif) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds

        print(f'TIF CRS: {raster_crs}')
        print()

        for i, city in enumerate(cities, 1):
            city_path = ethiopia_dir / city
            if not city_path.is_dir():
                continue

            print(f'[{i:>3}/{total}] {city}', end=' ... ', flush=True)
            ok = process_city(city, ethiopia_dir, src, raster_bounds, raster_crs)
            if ok:
                success += 1
                csv_path = city_path / 'Output' / f'{city}{SUFFIX}.csv'
                df = pd.read_csv(csv_path, usecols=['Pop2022'])
                valid = df['Pop2022'].dropna()
                nonzero = (valid > 0).sum()
                print(f'OK  rows={len(df)}, non-null={len(valid)}, non-zero={nonzero}, sum={valid.sum():,.0f}')
            else:
                failed += 1
                print('FAILED')

    print()
    print(f'Done. Success={success}, Failed={failed}')


if __name__ == '__main__':
    main()
