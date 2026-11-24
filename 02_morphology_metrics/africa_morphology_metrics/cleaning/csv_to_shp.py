"""
Join cleaned per-city CSV attributes back onto grid geometry to produce the
final SHP_cleaned output.

The cleaning pipeline (fix/derive/verify steps) operates entirely on the
CSV attribute tables for speed, since geometry is unchanged throughout
cleaning and re-reading/rewriting it at every step would be wasteful. This
final step re-attaches geometry exactly once: for each city, it takes the
authoritative (cleaned) attribute row set from ``--cleaned-dir`` and joins
it to that city's grid polygon geometry by GridID, then writes the combined
GeoDataFrame as a Shapefile.

Steps
-----
1. Read ``<cleaned-dir>/<country>/<city><suffix>.csv`` (cleaned attributes).
2. Read the source grid SHP for that city to get geometry (see "External
   input requirement" below).
3. Left-join geometry onto the CSV by GridID, with the CSV as the
   authoritative row set (a right join from geometry's perspective, so the
   cleaned CSV's row count is always preserved even if the source SHP has
   stray rows).
4. Rename columns using ``col_name_mapping.csv`` (only ThreeWayCnt ->
   ThreeWayCn is affected, since ESRI Shapefile field names are capped at
   10 characters).
5. Write to ``<out-root>/<country>/<city><suffix>.shp``.

External input requirement
---------------------------
Step 2 requires each city's source grid Shapefile, produced by the
upstream (pre-cleaning) grid-enrichment stage at
``<source-shp-root>/<country>/<city>/Output/<city><suffix>.shp``. This
working-pipeline geometry cache is not part of this repository (it is
large, per-run bulk data); ``--source-shp-root`` must be pointed at a
directory with that same ``<country>/<city>/Output/`` layout from your own
pipeline run before this script can produce output.

Supports multiprocessing (``--workers N``) and resumes automatically by
skipping cities whose output SHP already exists.
"""

import argparse
import logging
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15'


def load_col_rename(map_csv: Path) -> dict:
    """Read the column-rename table and return only the entries that actually change."""
    df = pd.read_csv(map_csv)
    return {row['csv_col']: row['shp_col']
            for _, row in df.iterrows()
            if row['csv_col'] != row['shp_col']}


def process_city(task) -> dict:
    """
    Process one city: read the cleaned CSV + source SHP, join, write output SHP.
    Returns {'city', 'country', 'status': 'ok'/'skip'/'error', 'msg'}.
    """
    csv_path, country, source_shp_root, out_root, col_rename = task
    city_stem = csv_path.stem
    city_name = city_stem.replace(SUFFIX, '')

    out_dir = out_root / country
    out_path = out_dir / (city_stem + '.shp')

    if out_path.exists():
        return {'city': city_name, 'country': country,
                'status': 'skip', 'msg': 'already_exists'}

    src_shp = source_shp_root / country / city_name / 'Output' / (city_stem + '.shp')
    if not src_shp.exists():
        return {'city': city_name, 'country': country,
                'status': 'skip', 'msg': f'src_shp_missing:{src_shp}'}

    try:
        df_csv = pd.read_csv(csv_path, low_memory=False)

        gdf_src = gpd.read_file(src_shp, columns=['GridID'])
        gdf_geom = gdf_src[['GridID', 'geometry']].copy()

        # GridID dtypes can differ between the CSV and the source SHP; normalize
        # to string for the join key.
        df_csv['GridID'] = df_csv['GridID'].astype(str)
        gdf_geom['GridID'] = gdf_geom['GridID'].astype(str)

        # Right join keeps the cleaned CSV as the authoritative row set.
        gdf_out = gdf_geom.merge(df_csv, on='GridID', how='right')

        gdf_out = gpd.GeoDataFrame(gdf_out, geometry='geometry', crs=gdf_src.crs)

        if col_rename:
            gdf_out.rename(columns=col_rename, inplace=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        gdf_out.to_file(out_path, driver='ESRI Shapefile', encoding='UTF-8')

        return {'city': city_name, 'country': country,
                'status': 'ok', 'msg': f'rows={len(gdf_out)}'}

    except Exception:
        return {'city': city_name, 'country': country,
                'status': 'error', 'msg': traceback.format_exc(limit=3)}


def main():
    parser = argparse.ArgumentParser(
        description='Join cleaned CSV attributes onto grid geometry to produce SHP_cleaned output.')
    parser.add_argument('--cleaned-dir', type=Path, required=True,
                         help='Root directory of cleaned per-city CSVs '
                              '(<cleaned-dir>/<country>/<city><suffix>.csv).')
    parser.add_argument('--source-shp-root', type=Path, required=True,
                         help='Root directory of the upstream pipeline\'s per-city '
                              'grid geometry cache: <source-shp-root>/<country>/<city>/'
                              'Output/<city><suffix>.shp. Not part of this repository; '
                              'point this at your own pipeline run\'s output directory.')
    parser.add_argument('--out-root', type=Path, required=True,
                         help='Directory to write SHP_cleaned output into.')
    parser.add_argument('--col-mapping', type=Path, required=True,
                         help='Path to col_name_mapping.csv (CSV column -> SHP field name).')
    parser.add_argument('--workers', type=int, default=8,
                         help='Number of worker processes (default: 8).')
    parser.add_argument('--log-path', type=Path, default=Path('csv_to_shp_progress.log'),
                         help='Path to write the run log to.')
    args = parser.parse_args()

    col_rename = load_col_rename(args.col_mapping)

    all_csvs = sorted(args.cleaned_dir.rglob(f'*{SUFFIX}.csv'))
    total = len(all_csvs)
    print(f'Cleaned CSVs found: {total}')

    tasks = [(csv_path, csv_path.parent.name, args.source_shp_root, args.out_root, col_rename)
             for csv_path in all_csvs]

    ok_count = 0
    skip_count = 0
    error_count = 0
    errors = []

    logging.basicConfig(
        filename=str(args.log_path),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    print(f'Using {args.workers} worker processes...')
    print(f'Output directory: {args.out_root}')
    print(f'Log: {args.log_path}')
    args.out_root.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_city, tasks, chunksize=4),
            total=total, desc='CSV->SHP', unit='city'
        ):
            status = result['status']
            city = result['city']
            country = result['country']
            msg = result['msg']

            if status == 'ok':
                ok_count += 1
                logging.info(f'OK {country}/{city} {msg}')
            elif status == 'skip':
                skip_count += 1
                logging.info(f'SKIP {country}/{city} {msg}')
            else:
                error_count += 1
                errors.append(result)
                logging.error(f'ERROR {country}/{city}\n{msg}')
                tqdm.write(f'  ERROR: {country}/{city}: {msg[:120]}')

    print(f'\n{"="*60}')
    print(f'Done: {ok_count}/{total}')
    print(f'Skipped: {skip_count}/{total}')
    print(f'Errors: {error_count}/{total}')

    if errors:
        err_path = args.out_root.parent / 'csv_to_shp_errors.csv'
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f'\nError details saved: {err_path}')
        print('First 5 errors:')
        for e in errors[:5]:
            print(f"  {e['country']}/{e['city']}: {e['msg'][:200]}")

    print(f'\nLog saved: {args.log_path}')


if __name__ == '__main__':
    os.environ.setdefault('PROJ_LIB', '/usr/share/proj')
    import warnings
    warnings.filterwarnings('ignore', message='.*PROJ.*')
    main()
