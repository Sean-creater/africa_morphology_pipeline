"""
Compute the 7 derived building-height metrics for cities whose enrichment
output stopped at 165 columns.

A subset of cities (roughly 473, concentrated in Cote d'Ivoire, Burkina
Faso, South Sudan, and elsewhere) reached the grid-enrichment stage with
open-building footprints and height/volume raster coverage available, but
their per-city output is missing the 7 metrics that are derived from
per-building height and volume rather than measured directly:
FloorARat, Mean_FormF, Mean_H2F, Mean_VolDe, Std_FormF, Std_H2F, Std_VolDe.

This script computes those 7 columns from the underlying building footprints
and country-level height/volume rasters and appends them to the existing
per-city CSV:

1. Load ``open_buildings.shp`` (building footprints, EPSG:4326).
2. Sample ``building_height.tif`` / ``building_volume.tif`` at each building
   centroid to get Height_90m / Volume_90m.
3. Compute per-building FormFactor (momepy), H2F_Ratio (height / sqrt(footprint
   area)), FloorArea (footprint area x estimated floor count), and
   VolumeDens_bldg (volume / footprint area).
4. Spatially join buildings to grid cells and aggregate to mean/std per cell;
   FloorARat is the per-cell sum of FloorArea divided by cell area (Area_m2).
5. Merge the 7 aggregated columns into the existing per-city CSV by GridID.
   All other columns are left untouched.

After processing, cities whose 7 target columns are present and pass a
quality check (present, not all-zero, and the existing TIF-derived and road
columns are not degenerate) are copied into the cleaned-output directory.

Inputs
------
- ``<data_root>/<country>/<city>/Output/<city><suffix>.csv`` — per-city metrics
- ``<data_root>/<country>/<city>/Output/<city><grid_suffix>.shp`` — grid geometry
- ``<data_root>/<country>/<city>/Output/open_buildings.shp`` — building footprints
- ``<tif_root>/<country>/building_height.tif``, ``building_volume.tif``
- ``--quality-summary`` CSV listing each city's current column count (``n_cols``);
  this is durable per-run reference data (not a throwaway diagnostic artifact),
  and cities with ``n_cols == 165`` are the ones processed here.

Output
------
Updates the per-city CSV in place with the 7 new columns, and copies cities
that pass the quality check into ``--cleaned-dir``.
"""

import argparse
import shutil
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
FLOOR_HEIGHT = 3.0   # metres per floor
MIN_HEIGHT = 2.0     # minimum remainder to count as an extra floor
H_NODATA = -32767.0
V_NODATA = -32767.0
H_DEFAULT = 3.0       # fallback height when the raster has no data at a centroid
V_DEFAULT = 0.0       # fallback volume when the raster has no data at a centroid

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
    floors = np.maximum(floors, 1)   # any building with h>0 has >=1 floor
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

    # Skip cities where the 7 columns are already present with real (non-degenerate) data.
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
        buildings = gpd.read_file(ob_path)   # EPSG:4326
        grid = gpd.read_file(shp_path)       # UTM CRS
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

    # Reproject to the grid's UTM CRS for area-based metrics.
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
        # Fall back to positional alignment when GridID is unavailable.
        for col in TARGET_COLS:
            if col in agg.columns:
                new_col = pd.Series(np.nan, index=df.index)
                for gidx, val in agg[col].items():
                    if 0 <= int(gidx) < len(df):
                        new_col.iloc[int(gidx)] = val
                df[col] = new_col

    # Grid cells with no buildings joined get 0 rather than NaN.
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df.to_csv(csv_path, index=False)
    return 'ok'


def main():
    parser = argparse.ArgumentParser(
        description='Derive the 7 height-based metrics for 165-column cities and '
                    'promote passing cities into the cleaned-output directory.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/.')
    parser.add_argument('--tif-root', type=Path, required=True,
                         help='Root directory containing <country>/building_height.tif '
                              'and building_volume.tif.')
    parser.add_argument('--quality-summary', type=Path, required=True,
                         help="CSV with columns [country_name, city_name, n_cols] "
                              "used to select cities with n_cols == 165.")
    parser.add_argument('--cleaned-dir', type=Path, required=True,
                         help='Directory to copy passing per-city CSVs into, '
                              'mirroring the <country>/<city><suffix>.csv layout.')
    args = parser.parse_args()

    qs = pd.read_csv(args.quality_summary)
    cities_165 = qs[qs['n_cols'] == 165][['country_name', 'city_name']]
    total = len(cities_165)
    print(f'Cities with 165 columns: {total}')
    print()

    results = {'ok': [], 'skip_already_done': [], 'skip_no_buildings': [],
               'skip_no_grid_shp': [], 'skip_no_buildings_in_grid': [], 'error': []}

    for _, r in tqdm(cities_165.iterrows(), total=total, desc='Deriving height metrics'):
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

    # ── Quality check + copy to the cleaned-output directory ──────────────────
    print()
    print('Running quality check and copying to cleaned-output directory...')
    TIF_COLS = ['LC_Built', 'H_max', 'V_sum', 'Pop2022', 'LCF_Built', 'LC_Shannon']
    ROAD_COLS = ['StreetDens', 'NodeDens']

    copied = 0
    failed_qc = []

    done_cities = results['ok'] + results['skip_already_done']
    for country, city, _ in done_cities:
        csv_path = args.data_root / country / city / 'Output' / f'{city}{SUFFIX}'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, low_memory=False)

        missing_targets = [c for c in TARGET_COLS if c not in df.columns]
        all_zero_targets = [c for c in TARGET_COLS
                             if c in df.columns and ((df[c] == 0) | df[c].isna()).all()]

        # LC_Built is exempt: some regions genuinely have no built-up land cover.
        tif_issues = [c for c in TIF_COLS
                      if c in df.columns and ((df[c] == 0) | df[c].isna()).all()
                      and c != 'LC_Built']

        road_issues = [c for c in ROAD_COLS
                       if c in df.columns and ((df[c] == 0) | df[c].isna()).all()]

        if missing_targets or tif_issues or road_issues:
            failed_qc.append({
                'country': country, 'city': city,
                'missing_targets': missing_targets,
                'tif_issues': tif_issues,
                'road_issues': road_issues,
            })
            continue

        dst_dir = args.cleaned_dir / country
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'{city}{SUFFIX}'
        shutil.copy2(csv_path, dst)
        copied += 1

    print(f'Copied to cleaned-output directory: {copied}')
    if failed_qc:
        print(f'Failed quality check: {len(failed_qc)}')
        for item in failed_qc[:10]:
            print(f"  {item['country']}/{item['city']}: "
                  f"missing={item['missing_targets']}, "
                  f"tif={item['tif_issues']}, road={item['road_issues']}")


if __name__ == '__main__':
    main()
