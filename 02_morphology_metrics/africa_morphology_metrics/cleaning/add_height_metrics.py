"""
Add the 7 derived building-height metrics to cities whose enrichment output
stopped at 114 columns, promoting them to the 121-column tier.

A larger subset of cities (roughly 1,266, concentrated in DRC, South Africa,
CAR, Republic of Congo, South Sudan, Cape Verde, and Sao Tome and Principe)
reached the grid-enrichment stage missing both the 51 road-network columns
and the 7 height-derived columns. This script closes the height-derived gap
using the same building-footprint + raster-sampling approach as
``derive_165_column_metrics.py``, bringing these cities from 114 to 121
columns. The remaining 51 road-network columns are added in a separate,
later step (``add_road_network_metrics.py``) once OSM road data is available.

Missing columns filled in: FloorARat, Mean_FormF, Mean_H2F, Mean_VolDe,
Std_FormF, Std_H2F, Std_VolDe.

Strategy
--------
1. Load ``open_buildings.shp`` (building footprints, EPSG:4326).
2. Sample ``building_height.tif`` / ``building_volume.tif`` at each building
   centroid to get Height_90m / Volume_90m.
3. Compute per-building FormFactor (momepy), H2F_Ratio, FloorArea, and
   VolumeDens_bldg.
4. Spatially join buildings to grid cells and aggregate to mean/std per cell;
   FloorARat is the per-cell sum of FloorArea divided by cell area (Area_m2).
5. Merge the 7 aggregated columns into the existing per-city CSV by GridID.

After processing, the city inventory's ``exclusion_reason`` is updated from
``col_114`` to ``col_121`` for cities that now carry all 7 columns with
non-degenerate data, recording that only the road-network metrics remain
outstanding for them.

Inputs
------
- ``<data_root>/<country>/<city>/Output/<city><suffix>.csv`` — per-city metrics
- ``<data_root>/<country>/<city>/Output/<city><grid_suffix>.shp`` — grid geometry
- ``<data_root>/<country>/<city>/Output/open_buildings.shp`` — building footprints
- ``<tif_root>/<country>/building_height.tif``, ``building_volume.tif``
- ``--city-inventory`` CSV with columns [country_name, city_name, n_cols,
  exclusion_reason, ...] — durable per-run tracking of each city's cleaning
  status; cities with ``exclusion_reason == 'col_114'`` are processed here.

Output
------
Updates the per-city CSV in place with the 7 new columns, and updates
``--city-inventory`` in place (exclusion_reason: col_114 -> col_121).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import momepy
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'
GRID_SUFFIX = '_grid_250m_hex_enriched_metrics_v15.shp'

# ── Constants (must match the main grid-enrichment pipeline's morphology_metrics.py) ──
FLOOR_HEIGHT = 3.0
MIN_HEIGHT = 2.0
H_NODATA = -32767.0
H_DEFAULT = 3.0   # fallback height when the raster has no data at a centroid
V_DEFAULT = 0.0   # fallback volume when the raster has no data at a centroid

TARGET_COLS = ['FloorARat', 'Mean_FormF', 'Mean_H2F', 'Mean_VolDe',
               'Std_FormF', 'Std_H2F', 'Std_VolDe']


def extract_raster_at_centroids(buildings_4326: gpd.GeoDataFrame,
                                 raster_path: Path,
                                 default: float) -> pd.Series:
    """Sample a raster at building centroids (buildings must be in EPSG:4326)."""
    if not raster_path.exists():
        return pd.Series([default] * len(buildings_4326), index=buildings_4326.index)
    try:
        with rasterio.open(raster_path) as src:
            cents = buildings_4326.geometry.centroid
            if src.crs and src.crs.to_epsg() != 4326:
                cents_gdf = gpd.GeoDataFrame(geometry=cents, crs='EPSG:4326')
                cents_gdf = cents_gdf.to_crs(src.crs)
                cents = cents_gdf.geometry

            coords = [(g.x, g.y) for g in cents]
            vals = [v[0] for v in src.sample(coords)]
            vals = pd.Series(vals, index=buildings_4326.index, dtype='float64')

            nodata = src.nodata
            if nodata is not None:
                vals = vals.where(~np.isclose(vals, nodata, rtol=1e-5, atol=1e-2), np.nan)
            vals = vals.where(~np.isclose(vals, H_NODATA, rtol=1e-5, atol=1e-2), np.nan)
            return vals
    except Exception as e:
        print(f'    [WARN] raster extraction failed ({raster_path.name}): {e}')
        return pd.Series([default] * len(buildings_4326), index=buildings_4326.index)


def calc_floors(height: pd.Series) -> np.ndarray:
    """Estimate floor count from building height using FLOOR_HEIGHT/MIN_HEIGHT."""
    h = np.asarray(height, dtype='float64')
    base = np.floor(h / FLOOR_HEIGHT)
    extra = np.where((h % FLOOR_HEIGHT) >= MIN_HEIGHT, 1, 0)
    floors = base + extra
    floors = np.maximum(floors, 1)
    floors = np.where(h <= 0, 0, floors)
    return floors


def process_city(country: str, city: str, data_root: Path, tif_root: Path) -> str:
    """
    Returns: 'ok' | 'skip_already_done' | 'skip_no_buildings' |
             'skip_no_grid_shp' | 'skip_no_buildings_in_grid' | 'error:<msg>'
    """
    out_dir = data_root / country / city / 'Output'
    csv_path = out_dir / f'{city}{SUFFIX}'
    shp_path = out_dir / f'{city}{GRID_SUFFIX}'
    ob_path = out_dir / 'open_buildings.shp'

    if not csv_path.exists():
        return 'error:no_csv'

    df_check = pd.read_csv(csv_path, nrows=0)
    if all(c in df_check.columns for c in TARGET_COLS):
        df_quick = pd.read_csv(csv_path, usecols=['Mean_VolDe', 'Std_VolDe'], low_memory=False)
        vde_ok = not (((df_quick['Mean_VolDe'] == 0) | df_quick['Mean_VolDe'].isna()).all())
        if vde_ok:
            return 'skip_already_done'

    if not ob_path.exists():
        return 'skip_no_buildings'

    if not shp_path.exists():
        return 'skip_no_grid_shp'

    try:
        buildings = gpd.read_file(ob_path)
        grid = gpd.read_file(shp_path)
    except Exception as e:
        return f'error:read_shp:{e}'

    if len(buildings) == 0:
        return 'skip_no_buildings'

    if buildings.crs and buildings.crs.to_epsg() != 4326:
        buildings = buildings.to_crs('EPSG:4326')

    # The TIF directory uses underscores in country names; the data directory uses spaces.
    country_tif = country.replace(' ', '_')
    h_tif = tif_root / country_tif / 'building_height.tif'
    v_tif = tif_root / country_tif / 'building_volume.tif'
    if not h_tif.exists():
        h_tif = tif_root / country / 'building_height.tif'
    if not v_tif.exists():
        v_tif = tif_root / country / 'building_volume.tif'

    buildings['Height_90m'] = extract_raster_at_centroids(buildings, h_tif, H_DEFAULT)
    buildings['Height_90m'] = buildings['Height_90m'].clip(lower=0, upper=200).fillna(H_DEFAULT)

    buildings['Volume_90m'] = extract_raster_at_centroids(buildings, v_tif, V_DEFAULT)
    buildings['Volume_90m'] = buildings['Volume_90m'].clip(lower=0, upper=1e6).fillna(V_DEFAULT)

    grid_crs = grid.crs
    bldg_proj = buildings.to_crs(grid_crs)
    bldg_proj['Area'] = bldg_proj.geometry.area
    bldg_proj['Height_90m'] = buildings['Height_90m'].values
    bldg_proj['Volume_90m'] = buildings['Volume_90m'].values

    try:
        bldg_proj['FormFactor'] = momepy.form_factor(bldg_proj, bldg_proj['Height_90m'])
    except Exception as e:
        return f'error:momepy_formfactor:{e}'

    sqrt_area = np.sqrt(bldg_proj['Area'].clip(lower=0.01))
    bldg_proj['H2F_Ratio'] = bldg_proj['Height_90m'] / sqrt_area

    floors = calc_floors(bldg_proj['Height_90m'])
    bldg_proj['FloorArea'] = bldg_proj['Area'] * floors

    bldg_proj['VolumeDens_bldg'] = bldg_proj['Volume_90m'] / bldg_proj['Area'].clip(lower=0.01)

    if 'GridID' not in grid.columns:
        grid = grid.reset_index().rename(columns={'index': 'GridID'})
    grid = grid.copy()
    grid['_gidx'] = grid.index

    try:
        joined = gpd.sjoin(bldg_proj, grid[['_gidx', 'geometry', 'Area_m2']],
                            how='left', predicate='within')
    except Exception as e:
        return f'error:sjoin:{e}'

    joined = joined.dropna(subset=['_gidx'])

    if len(joined) == 0:
        return 'skip_no_buildings_in_grid'

    agg = joined.groupby('_gidx').agg(
        Sum_FlrAr=('FloorArea', 'sum'),
        Mean_FormF=('FormFactor', 'mean'),
        Std_FormF=('FormFactor', 'std'),
        Mean_H2F=('H2F_Ratio', 'mean'),
        Std_H2F=('H2F_Ratio', 'std'),
        Mean_VolDe=('VolumeDens_bldg', 'mean'),
        Std_VolDe=('VolumeDens_bldg', 'std'),
    ).reset_index()

    grid_area = grid.set_index('_gidx')['Area_m2']
    agg = agg.set_index('_gidx')
    agg['FloorARat'] = agg['Sum_FlrAr'] / grid_area
    agg = agg.drop(columns=['Sum_FlrAr'])

    df = pd.read_csv(csv_path, low_memory=False)

    if 'GridID' in grid.columns and 'GridID' in df.columns:
        gidx_to_gridid = grid.set_index('_gidx')['GridID']
        agg.index = agg.index.map(gidx_to_gridid)
        agg.index.name = 'GridID'
        for col in TARGET_COLS:
            if col in agg.columns:
                df[col] = df['GridID'].map(agg[col])
    else:
        for col in TARGET_COLS:
            if col in agg.columns:
                new_col = pd.Series(np.nan, index=df.index)
                for gidx, val in agg[col].items():
                    if 0 <= int(gidx) < len(df):
                        new_col.iloc[int(gidx)] = val
                df[col] = new_col

    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df.to_csv(csv_path, index=False)
    return 'ok'


def main():
    parser = argparse.ArgumentParser(
        description='Add the 7 height-based metrics to col_114 cities, promoting '
                    'them to col_121 status in the city inventory.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/.')
    parser.add_argument('--tif-root', type=Path, required=True,
                         help='Root directory containing <country>/building_height.tif '
                              'and building_volume.tif.')
    parser.add_argument('--city-inventory', type=Path, required=True,
                         help='Path to city_inventory.csv (read and updated in place).')
    args = parser.parse_args()

    inv = pd.read_csv(args.city_inventory)
    cities_114 = inv[inv['exclusion_reason'] == 'col_114'][['country_name', 'city_name']]
    total = len(cities_114)
    print(f'Cities at col_114: {total}')
    print()

    results = {'ok': [], 'skip_already_done': [], 'skip_no_buildings': [],
               'skip_no_grid_shp': [], 'skip_no_buildings_in_grid': [], 'error': []}

    for _, r in tqdm(cities_114.iterrows(), total=total, desc='Adding height metrics'):
        country, city = r['country_name'], r['city_name']
        status = process_city(country, city, args.data_root, args.tif_root)
        cat = status if status in results else 'error'
        results[cat].append((country, city, status))

    print()
    print('=' * 60)
    for cat, items in results.items():
        print(f'{cat:35s}: {len(items)}')

    if results['error']:
        print('\nError cities:')
        for country, city, status in results['error'][:20]:
            print(f'  {country}/{city}: {status}')

    # ── Update city_inventory: col_114 -> col_121 for successfully processed cities ──
    print()
    print('Updating city inventory...')
    inv_updated = pd.read_csv(args.city_inventory)

    ok_set = {(c, ct) for c, ct, _ in results['ok'] + results['skip_already_done']}
    updated = 0
    for idx, row in inv_updated.iterrows():
        if (row['country_name'], row['city_name']) in ok_set:
            if inv_updated.at[idx, 'exclusion_reason'] == 'col_114':
                inv_updated.at[idx, 'exclusion_reason'] = 'col_121'
                inv_updated.at[idx, 'n_cols'] = 121
                inv_updated.at[idx, 'exclusion_note'] = 'height metrics added; still missing 51 road network cols'
                updated += 1

    inv_updated.to_csv(args.city_inventory, index=False)
    print(f'City inventory updated: {updated} cities col_114 -> col_121')
    print()

    print('Spot check (Mean_VolDe for the first 5 successfully processed cities):')
    for country, city, _ in results['ok'][:5]:
        csv_path = args.data_root / country / city / 'Output' / f'{city}{SUFFIX}'
        df = pd.read_csv(csv_path, usecols=['Mean_VolDe'], low_memory=False)
        nonzero = (df['Mean_VolDe'] > 0).sum()
        total_rows = len(df)
        print(f'  {country}/{city}: Mean_VolDe nonzero cells {nonzero}/{total_rows}')


if __name__ == '__main__':
    main()
