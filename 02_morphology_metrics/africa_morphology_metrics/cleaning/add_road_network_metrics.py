"""
Add the 51 road-network and street-connectivity columns to col_121 cities,
promoting passing cities to the final 172-column cleaned schema.

col_121 cities have land cover, WSF3D height/volume, population, and
building-morphology metrics (including the derived height metrics from
``add_height_metrics.py`` / ``derive_165_column_metrics.py``) but no
road-network signal, because that requires an OSM road extract per city
that is fetched separately from the raster/vector layers used earlier in
the pipeline.

Pipeline per city:
  Phase 1: Download/clip OSM highway data per country (batched, one road
           network extract cached per country to avoid repeated downloads).
  Phase 2: Compute street-network metrics (16 columns: density, width,
           orientation, building-street alignment, etc.) and connectivity
           metrics (35 columns: centrality, clustering, intersection
           topology, COINS stroke continuity, etc.).
  Phase 3: Merge the 51 new columns into the existing per-city CSV by
           GridID and run the standard non-degenerate-column quality check.
  Phase 4: Copy passing cities into the cleaned-output directory and update
           the city inventory (cleaned / excluded / still col_121).

A fixed list of cities with pre-existing WSF3D/land-cover data-quality
issues (``KNOWN_BAD``) still get road metrics computed (for completeness of
their raw per-city CSV) but are excluded from the cleaned-output directory,
since their non-road columns already fail quality checks independently of
the road-network addition.

Code dependencies
-----------------
The road, connectivity, OSM and projection implementations are imported from
the published ``africa_morphology_metrics`` package in this repository.

Inputs
------
- ``<data_root>/<country>/<city>/Output/<city><suffix>.csv`` — per-city metrics
- ``<data_root>/<country>/<city>/Output/<city><grid_suffix>.shp`` — grid geometry
- ``<data_root>/<country>/<city>/Output/open_buildings.shp`` — building footprints (optional)
- ``<data_root>/<country>/<city>/highways/<city>_highways.geojson`` — clipped OSM roads
- ``--agglomeration-file`` — GeoPackage with Agglomeration_Name -> ISO3 lookup,
  used to select the correct projected CRS per city
- ``--city-inventory`` CSV, filtered to ``exclusion_reason == 'col_121'``

Output
------
Updates each city's per-city CSV in place with the 51 new columns, copies
passing cities into ``--cleaned-dir``, and updates ``--city-inventory`` in
place (status/exclusion_reason/n_cols/exclusion_note).
"""

import argparse
import shutil
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import momepy
from tqdm import tqdm

from africa_morphology_metrics.streetnetwork_metrics import calculate_street_network_metrics_with_graph
from africa_morphology_metrics.connectivity import calculate_connectivity_metrics
from africa_morphology_metrics.osm_network import process_osm_highways_for_cities
from africa_morphology_metrics.projection import get_crs_for_city

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'
SHP_SUFFIX = '_grid_250m_hex_enriched_metrics_v15.shp'

# ── 51 target columns ──────────────────────────────────────────────────────────
TARGET_COLS = [
    'StreetDens', 'NodeDens', 'StreetsPN', 'SegmentDen', 'PlotDens', 'BldgDensN',
    'PlotFrag', 'StrWidth', 'StrOpen', 'StrWidthD', 'StrLinM', 'StrLinS',
    'StrOrientM', 'StrOrientS', 'BldgStrAM', 'BldgStrAS',
    'BtwCentM', 'BtwCentX', 'BtwCentS', 'ClsCentM', 'ClsCentX', 'ClsCentS',
    'StrCentM', 'StrCentX', 'StrCentS', 'NdNbDenM', 'NdNbDenS',
    'ClusterM', 'ClusterS', 'MeanDeg', 'StdDeg',
    'DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt',
    'Ratio3way', 'Ratio4way', 'CuldesacR', 'MeanNdDst',
    'Cyclomat', 'EdgeNdRat', 'GammaIdx', 'Meshedness',
    'EdgeBtwM', 'EdgeClsM', 'EdgeStrM', 'CuldesacL',
    'COINSCnt', 'COINSLen', 'DeadendDen', 'InterDens',
]

# ── Cities with pre-existing WSF3D/land-cover data-quality issues ──────────────
# Road metrics are still computed for these (their raw CSV stays complete),
# but they are excluded from the cleaned-output directory regardless of the
# road-metric outcome, since their non-road columns already fail QC.
KNOWN_BAD = {
    ('Chad', 'Tine Djangarba'),
    ('Chad', 'Youe'),
    ('Democratic Republic of the Congo', 'Bena-Musakatshi_Mulamba'),
    ('Democratic Republic of the Congo', 'Bossondjo'),
    ('Democratic Republic of the Congo', 'Elema-Impata'),
    ('Democratic Republic of the Congo', 'Kakoma'),
    ('Democratic Republic of the Congo', 'Mpoyi'),
    ('Democratic Republic of the Congo', 'Tshioji'),
    ('Eritrea', 'Adhi Bdo'),
    ('Mauritius', 'Port-Mathurin'),
    ('Seychelles', 'Victoria'),
    ('Sudan', 'Abu Shagar Refugee Camp'),
    ('Sudan', 'Baqbaqa New Settlement'),
    ('Sudan', 'Deirirro'),
    ('Sudan', 'Hamsh Kurib'),
    ('Sudan', 'Haskanita'),
    ('Sudan', 'Hilat al-Asira_Hilat Qubur ab Kiday'),
    ('Sudan', 'Jawri Refugee Camp'),
    ('Sudan', 'Kassab'),
    ('Sudan', 'Matamir'),
    ('Sudan', 'New Setllement Wad al Hulaywah'),
    ('Sudan', 'Qarya 6'),
    ('Sudan', 'Qarya 8'),
    ('Sudan', "Qarya Sabha'Arab"),
    ('Sudan', "Qarya Sitta'Arab"),
    ('Sudan', "Qarya Wahad 'Arab"),
    ('Sudan', 'Rifi Talkuk'),
    ('Sudan', 'Um Sanqur Refugee Camp'),
    ('Sudan', 'Wad al-Mahi 11'),
    ('Sudan', 'Waggar'),
    ('Sudan', 'al-Jamaya Refugee Camp'),
    ('Sudan', 'al-Radis 1'),
    ('Sudan', 'al-Radis 2 Refugee Camp'),
}

# ── Quality columns that must be non-zero for a city to be considered cleaned ──
QC_NONZERO = [
    'H_max', 'H_mean', 'H_std', 'H_sum', 'V_sum', 'V_mean', 'V_max', 'V_std',
    'Pop2022', 'LCF_Built', 'LC_Shannon',
    'FloorARat', 'Mean_FormF', 'Mean_H2F', 'Mean_VolDe',
    'Std_FormF', 'Std_H2F', 'Std_VolDe',
    'StreetDens', 'NodeDens',
]

COUNTRY_ISO3 = {
    'Algeria': 'DZA', 'Angola': 'AGO', 'Benin': 'BEN', 'Botswana': 'BWA',
    'Burkina Faso': 'BFA', 'Burundi': 'BDI', 'Cameroon': 'CMR', 'Cape Verde': 'CPV',
    'Central African Republic': 'CAF', 'Chad': 'TCD', 'Comoros': 'COM',
    'Democratic Republic of the Congo': 'COD', 'Egypt': 'EGY', 'Eritrea': 'ERI',
    'Ethiopia': 'ETH', 'Gabon': 'GAB', 'Gambia': 'GMB', 'Ghana': 'GHA',
    'Guinea': 'GIN', 'Guinea-Bissau': 'GNB', 'Ivory Coast': 'CIV',
    'Côte dIvoire': 'CIV', "Cote d'Ivoire": 'CIV',
    'Kenya': 'KEN', 'Lesotho': 'LSO', 'Liberia': 'LBR', 'Libya': 'LBY',
    'Madagascar': 'MDG', 'Malawi': 'MWI', 'Mali': 'MLI', 'Mauritania': 'MRT',
    'Mauritius': 'MUS', 'Morocco': 'MAR', 'Mozambique': 'MOZ', 'Namibia': 'NAM',
    'Niger': 'NER', 'Nigeria': 'NGA', 'Republic of Congo': 'COG',
    'Rwanda': 'RWA', 'Sao Tome and Principe': 'STP', 'Senegal': 'SEN',
    'Seychelles': 'SYC', 'Sierra Leone': 'SLE', 'Somalia': 'SOM',
    'South Africa': 'ZAF', 'South Sudan': 'SSD', 'Sudan': 'SDN',
    'Swaziland': 'SWZ', 'Tanzania': 'TZA', 'Togo': 'TGO', 'Tunisia': 'TUN',
    'Uganda': 'UGA', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE',
}


def get_iso3(city_name: str, agg_df: pd.DataFrame, country_name: str = None) -> str:
    """Get the ISO3 country code for a city.
    First tries an exact match against the agglomeration lookup table.
    Falls back to the country-name -> ISO3 table for merged or
    non-standard city names not present in the lookup table.
    """
    rows = agg_df[agg_df['Agglomeration_Name'] == city_name]
    if len(rows) > 0:
        return rows.iloc[0]['ISO3']
    if country_name and country_name in COUNTRY_ISO3:
        return COUNTRY_ISO3[country_name]
    return None


def process_city(country: str, city: str, iso3: str, data_root: Path,
                  calculate_street_network_metrics_with_graph,
                  calculate_connectivity_metrics,
                  get_crs_for_city, logger) -> str:
    """
    Compute road-network and connectivity metrics for one city.

    Returns: 'ok' | 'skip_already_done' | 'skip_no_highway' |
             'skip_no_shp' | 'skip_no_roads_in_area' | 'error:<msg>'
    """
    out_dir = data_root / country / city / 'Output'
    csv_path = out_dir / f'{city}{SUFFIX}'
    shp_path = out_dir / f'{city}{SHP_SUFFIX}'
    hwys_path = data_root / country / city / 'highways' / f'{city}_highways.geojson'

    if not csv_path.exists():
        return 'error:no_csv'
    if not shp_path.exists():
        return 'skip_no_shp'

    df_head = pd.read_csv(csv_path, nrows=0)
    if all(c in df_head.columns for c in TARGET_COLS):
        df_quick = pd.read_csv(csv_path, usecols=['StreetDens', 'NodeDens'], low_memory=False)
        if not (((df_quick['StreetDens'] == 0) | df_quick['StreetDens'].isna()).all()):
            return 'skip_already_done'

    if not hwys_path.exists():
        return 'skip_no_highway'

    try:
        grid = gpd.read_file(shp_path)
    except Exception as e:
        return f'error:read_shp:{e}'

    if 'GridID' not in grid.columns:
        grid = grid.reset_index().rename(columns={'index': 'GridID'})

    try:
        highways = gpd.read_file(hwys_path)
    except Exception as e:
        return f'error:read_hwy:{e}'

    if len(highways) == 0:
        return 'skip_no_roads_in_area'

    try:
        target_crs = get_crs_for_city(city, iso3)
    except Exception:
        target_crs = 'ESRI:102022'   # Africa Albers Equal Area Conic fallback

    # Buildings are optional context for BldgDensN and BldgStrAM/BldgStrAS.
    ob_path = out_dir / 'open_buildings.shp'
    buildings = None
    if ob_path.exists():
        try:
            buildings = gpd.read_file(ob_path)
            if len(buildings) == 0:
                buildings = None
            else:
                if buildings.crs and buildings.crs.to_epsg() != 4326:
                    buildings = buildings.to_crs('EPSG:4326')
                buildings = buildings.to_crs(target_crs)
                try:
                    buildings['Orientation'] = momepy.orientation(buildings)
                except Exception:
                    pass   # BldgStrAM/BldgStrAS will be NaN for this city
        except Exception:
            buildings = None

    if grid.crs is None or str(grid.crs) != target_crs:
        try:
            grid = grid.to_crs(target_crs)
        except Exception as e:
            return f'error:reproject_grid:{e}'

    try:
        grid_with_metrics, nodes_gdf, edges_gdf, graph = calculate_street_network_metrics_with_graph(
            roads=highways,
            cells=grid,
            buildings=buildings,
            cell_id_col='GridID',
            area_col='Area_m2',
            target_crs=target_crs,
        )
    except Exception as e:
        return f'error:streetnet:{e}'

    if graph is not None and nodes_gdf is not None and len(nodes_gdf) > 0:
        try:
            dummy_pkl = out_dir / 'dummy_for_cache.pkl'
            grid_with_metrics = calculate_connectivity_metrics(
                grid=grid_with_metrics,
                nodes_gdf=nodes_gdf,
                edges_gdf=edges_gdf,
                graph=graph,
                spatial_graphs_pkl=dummy_pkl,
                grid_id_col='GridID',
                skip_coins=False,
            )
        except Exception as e:
            return f'error:connectivity:{e}'

    new_cols = [c for c in TARGET_COLS if c in grid_with_metrics.columns]
    if not new_cols:
        return 'skip_no_roads_in_area'

    df = pd.read_csv(csv_path, low_memory=False)

    road_data = grid_with_metrics[['GridID'] + new_cols].copy()
    road_data = road_data.rename(columns={'GridID': '_GridID'})
    road_data = road_data.set_index('_GridID')

    for col in new_cols:
        df[col] = df['GridID'].map(road_data[col])

    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df.to_csv(csv_path, index=False)

    # Clean up the per-city subgraph cache created during connectivity computation.
    for f in out_dir.glob('subgraph_cache_*.pkl'):
        try:
            f.unlink()
        except Exception:
            pass
    dummy_pkl_path = out_dir / 'dummy_for_cache.pkl'
    if dummy_pkl_path.exists():
        try:
            dummy_pkl_path.unlink()
        except Exception:
            pass

    return 'ok'


def quality_check(country: str, city: str, data_root: Path) -> list:
    """Return the list of QC-failed columns for a city (empty list = pass)."""
    csv_path = data_root / country / city / 'Output' / f'{city}{SUFFIX}'
    if not csv_path.exists():
        return ['no_csv']
    df = pd.read_csv(csv_path, low_memory=False)
    failed = []
    for col in QC_NONZERO:
        if col in df.columns:
            if ((df[col] == 0) | df[col].isna()).all():
                failed.append(col)
    return failed


def main():
    parser = argparse.ArgumentParser(
        description='Add the 51 road-network/connectivity columns to col_121 cities.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/ '
                              'and <country>/<city>/highways/.')
    parser.add_argument('--city-inventory', type=Path, required=True,
                         help='Path to city_inventory.csv (read and updated in place).')
    parser.add_argument('--cleaned-dir', type=Path, required=True,
                         help='Directory to copy passing per-city CSVs into.')
    parser.add_argument('--agglomeration-file', type=Path, required=True,
                         help='GeoPackage with Agglomeration_Name -> ISO3 lookup '
                              '(used to pick each city\'s projected CRS).')
    parser.add_argument('--country-boundaries-file', type=Path, required=True,
                         help='Country boundaries GeoPackage used by the bundled OSM processor.')
    parser.add_argument('--grid-data-base', type=Path, required=True,
                         help='Base directory of country-level grid data used by '
                              'the OSM highway processor to determine city extents.')
    parser.add_argument('--log-path', type=Path, default=Path('add_road_network_metrics.log'),
                         help='Path to write the run log to.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(args.log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)

    inv = pd.read_csv(args.city_inventory)
    agg_df = gpd.read_file(args.agglomeration_file)[['Agglomeration_Name', 'ISO3']].copy()

    cities_121 = inv[inv['exclusion_reason'] == 'col_121'][['country_name', 'city_name']].copy()
    total = len(cities_121)
    logger.info(f'col_121 cities: {total}')

    # ── Phase 1: OSM highway download (batched per country) ────────────────
    logger.info('=' * 70)
    logger.info('Phase 1: OSM highway download')
    logger.info('=' * 70)

    country_groups = cities_121.groupby('country_name')['city_name'].apply(list).to_dict()
    hwy_results = {}

    for country, city_list in country_groups.items():
        cities_needing_hwy = []
        for city in city_list:
            hwys = args.data_root / country / city / 'highways' / f'{city}_highways.geojson'
            if hwys.exists():
                hwy_results[city] = hwys
            else:
                cities_needing_hwy.append(city)

        if not cities_needing_hwy:
            logger.info(f'  {country}: all cities already have highway files, skipping')
            continue

        logger.info(f'  {country}: downloading {len(cities_needing_hwy)} cities...')
        try:
            results = process_osm_highways_for_cities(
                city_names=cities_needing_hwy,
                base_output_dir=args.data_root,
                agglomeration_file=args.agglomeration_file,
                grid_data_base=args.grid_data_base,
                country_boundaries_file=args.country_boundaries_file,
                logger=logger,
                force_redownload=False,
                overwrite_existing=False,
            )
            hwy_results.update({k: v for k, v in results.items()})
            success_n = sum(1 for v in results.values() if v is not None)
            logger.info(f'  {country}: {success_n}/{len(cities_needing_hwy)} succeeded')
        except Exception as e:
            logger.error(f'  {country}: download error: {e}')

    # ── Phase 2: Compute road metrics per city ──────────────────────────────
    logger.info('=' * 70)
    logger.info('Phase 2: Computing road-network metrics')
    logger.info('=' * 70)

    results = {
        'ok': [], 'skip_already_done': [], 'skip_no_highway': [],
        'skip_no_shp': [], 'skip_no_roads_in_area': [], 'error': []
    }

    t0 = time.time()
    for i, (_, r) in enumerate(tqdm(cities_121.iterrows(), total=total, desc='Road metrics'), 1):
        country, city = r['country_name'], r['city_name']
        iso3 = get_iso3(city, agg_df, country)
        if iso3 is None:
            results['error'].append((country, city, 'error:no_iso3'))
            continue

        status = process_city(country, city, iso3, args.data_root,
                               calculate_street_network_metrics_with_graph,
                               calculate_connectivity_metrics,
                               get_crs_for_city, logger)
        cat = status if status in results else 'error'
        results[cat].append((country, city, status))

        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            logger.info(f'  [{i}/{total}] ok={len(results["ok"])} skip_hwy={len(results["skip_no_highway"])} '
                        f'err={len(results["error"])} ETA={eta/60:.0f}min')

    logger.info('')
    logger.info('=' * 70)
    logger.info('Phase 2 results:')
    for cat, items in results.items():
        logger.info(f'  {cat:<30}: {len(items)}')

    if results['error']:
        logger.info('')
        logger.info('Error cities (first 20):')
        for country, city, status in results['error'][:20]:
            logger.info(f'  {country}/{city}: {status}')

    # ── Phase 3: Quality check + copy to cleaned-output directory ───────────
    logger.info('')
    logger.info('=' * 70)
    logger.info('Phase 3: Quality check + copy to cleaned-output directory')
    logger.info('=' * 70)

    ok_cities = results['ok'] + results['skip_already_done']
    copied = 0
    qc_failed = []

    for country, city, _ in ok_cities:
        if (country, city) in KNOWN_BAD:
            continue

        failed_cols = quality_check(country, city, args.data_root)
        if failed_cols:
            qc_failed.append((country, city, failed_cols))
            logger.warning(f'  QC FAIL {country}/{city}: {failed_cols}')
            continue

        csv_path = args.data_root / country / city / 'Output' / f'{city}{SUFFIX}'
        dst_dir = args.cleaned_dir / country
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'{city}{SUFFIX}'
        shutil.copy2(csv_path, dst)
        copied += 1

    logger.info(f'Copied to cleaned-output directory: {copied}')
    logger.info(f'Failed QC: {len(qc_failed)}')

    # ── Phase 4: Update city inventory ───────────────────────────────────────
    logger.info('')
    logger.info('Phase 4: Updating city inventory')

    inv_updated = pd.read_csv(args.city_inventory)
    ok_set = {(c, ct) for c, ct, _ in ok_cities}
    qc_failed_set = {(c, ct) for c, ct, _ in qc_failed}

    updated_cleaned = 0
    updated_excluded = 0

    for idx, row in inv_updated.iterrows():
        key = (row['country_name'], row['city_name'])
        if row['exclusion_reason'] != 'col_121':
            continue

        if key in KNOWN_BAD:
            # Retains its pre-existing WSF3D/land-cover issue; leave as col_121.
            continue

        if key in qc_failed_set:
            inv_updated.at[idx, 'status'] = 'excluded'
            inv_updated.at[idx, 'n_cols'] = 172
            failed_detail = next((str(cols) for c, ct, cols in qc_failed if (c, ct) == key), 'qc_failed')
            inv_updated.at[idx, 'exclusion_note'] = f'road metrics added but QC failed: {failed_detail}'
            updated_excluded += 1

        elif key in ok_set and key not in qc_failed_set:
            inv_updated.at[idx, 'status'] = 'cleaned'
            inv_updated.at[idx, 'exclusion_reason'] = ''
            inv_updated.at[idx, 'n_cols'] = 172
            inv_updated.at[idx, 'exclusion_note'] = ''
            updated_cleaned += 1

    inv_updated.to_csv(args.city_inventory, index=False)
    logger.info(f'  City inventory updated: {updated_cleaned} -> cleaned, {updated_excluded} -> excluded')

    logger.info('')
    logger.info('=' * 70)
    logger.info('Final tally:')
    inv_final = pd.read_csv(args.city_inventory)
    logger.info(f"  cleaned: {(inv_final['status']=='cleaned').sum()}")
    logger.info(f"  excluded: {(inv_final['status']=='excluded').sum()}")
    logger.info(f"  col_121 residual: {(inv_final['exclusion_reason']=='col_121').sum()}")


if __name__ == '__main__':
    main()
