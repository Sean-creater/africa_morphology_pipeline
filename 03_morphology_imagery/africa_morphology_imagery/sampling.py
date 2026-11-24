#!/usr/bin/env python3
"""Two-stage sampling of hexagonal grid cells for imagery construction.

Stage 1, `select_candidate_cells` / `run_candidate_selection`: for each city,
reads the full per-city 250 m hex grid produced by
`../../02_morphology_metrics/` and draws an initial, uniform-random
pool of candidate cells (all cells if the city has <= `max_candidates`,
otherwise a random subset). This keeps the working set small enough for the
per-cell imagery pipeline while still leaving enough cells for the
morphology-aware stage below to stratify over. Writes one
`{City}_sampled_cells.shp` shapefile per city.

Stage 2, `stratified_sample_city` / `run_sampling`: reads those per-city
candidate shapefiles and draws the final stratified sample used for
rendering and annotation, so that the dataset covers the full range of
building and street morphology rather than whatever happens to be common in
each city.

Stratification dimensions (three morphology columns produced by the
companion pipeline):
  1. Mean_Squar  - mean building squareness (building shape regularity)
  2. BldgStrAM   - building-to-street alignment
  3. StrOrientS  - street orientation diversity (Shannon entropy of street
                   bearings)

Each dimension is split at its city-specific median into "low"/"high",
producing 2^3 = 8 strata per city (stratum IDs 0-7, encoded as
Mean_Squar_bit*4 + BldgStrAM_bit*2 + StrOrientS_bit). Samples are drawn from
each stratum in proportion to how many cells fall into it, so that rarer
morphological combinations are neither ignored nor swamped.

The total number of cells drawn from a city in Stage 2 depends on its size,
so that small cities are not proportionally over- or under-sampled relative
to large ones:

    size category   total grid cells   quota
    small           <= 500             55
    medium          501 - 3000         115
    large           > 3000             180

(the quota is capped at the city's total candidate-cell count if smaller).

Output of Stage 2:
  - `SampleGridCell.csv`: one row per sampled grid cell, with columns
    GridID, City, Country, CRS, Center_X, Center_Y, Stratum_ID,
    Stratum_Shape, Stratum_Alignment, Stratum_Diversity, Mean_Squar,
    BldgStrAM, StrOrientS.
  - `SamplingStatistics.csv`: one row per city, with total/sampled cell
    counts, sampling rate, and size category.
"""

import sys
import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# The three morphology dimensions used for stratification.
STRATIFICATION_DIMENSIONS = ['Mean_Squar', 'BldgStrAM', 'StrOrientS']

# Sampling quota by city-size category.
SIZE_CATEGORY_QUOTAS = {
    'small': 55,     # total grid cells <= 500
    'medium': 115,   # 501 - 3000
    'large': 180,    # > 3000
}

RANDOM_STATE = 42

# Stage 1 (candidate selection): cap on the number of candidate cells drawn
# per city before morphology-aware stratification.
DEFAULT_MAX_CANDIDATES = 200

# ISO3 country codes, used as a prefix in generated GridIDs when a sampled
# cell does not already carry a GridID.
COUNTRY_TO_ISO3 = {
    'Ghana': 'GHA', 'Tanzania': 'TZA', 'Senegal': 'SEN',
    'Togo': 'TGO', 'Benin': 'BEN', 'Niger': 'NER',
    'Côte dIvoire': 'CIV', "Côte d'Ivoire": 'CIV',
    'Nigeria': 'NGA', 'Kenya': 'KEN', 'Uganda': 'UGA',
    'Rwanda': 'RWA', 'Burundi': 'BDI', 'Ethiopia': 'ETH',
    'Mali': 'MLI', 'Burkina Faso': 'BFA', 'Cameroon': 'CMR',
    'Chad': 'TCD', 'South Sudan': 'SSD', 'Mozambique': 'MOZ',
    'Madagascar': 'MDG', 'Zambia': 'ZMB', 'Malawi': 'MWI',
    'Angola': 'AGO', 'Republic of Congo': 'COG', 'Gabon': 'GAB',
    'Algeria': 'DZA', 'Egypt': 'EGY', 'Tunisia': 'TUN',
    'Morocco': 'MAR', 'Botswana': 'BWA', 'Guinea': 'GIN',
    'Democratic Republic of the Congo': 'COD',
    'South Africa': 'ZAF',
}


# ============================================================================
# Stage 1: candidate cell selection (uniform random subsample per city)
# ============================================================================

def select_candidate_cells(grid_cells: gpd.GeoDataFrame, max_candidates: int,
                            random_state: int = RANDOM_STATE,
                            logger: logging.Logger = None) -> gpd.GeoDataFrame:
    """Draw a uniform-random candidate pool from a city's full hex grid.

    Takes every cell if the grid has at most `max_candidates` cells,
    otherwise draws `max_candidates` cells without replacement. This is
    the coarse first pass; morphology-aware stratification (Stage 2) then
    refines this candidate pool down to the final sample.
    """
    logger = logger or logging.getLogger(__name__)
    total_cells = len(grid_cells)

    if total_cells <= max_candidates:
        logger.info(f"  Taking all {total_cells} cells (<= max_candidates)")
        return grid_cells.copy()

    sampled_indices = np.random.choice(grid_cells.index, size=max_candidates, replace=False)
    logger.info(f"  Randomly sampled {max_candidates}/{total_cells} candidate cells")
    return grid_cells.loc[sampled_indices].copy()


def run_candidate_selection(cities_csv: Path, grid_glob_template: str, output_shp_dir: Path,
                             max_candidates: int = DEFAULT_MAX_CANDIDATES,
                             logger: logging.Logger = None) -> None:
    """Draw and save a candidate-cell shapefile for every city in `cities_csv`.

    Args:
        cities_csv: CSV with at least City, Country columns.
        grid_glob_template: format string with {city} and {country} placeholders
            resolving to each city's full hex-grid shapefile, e.g.
            "/path/to/{country}/{city}/Output/{city}_grid_250m_hex_enriched_metrics_v15.shp".
        output_shp_dir: directory to write `{City}_sampled_cells.shp` into.
        max_candidates: per-city candidate cap (see `select_candidate_cells`).
    """
    logger = logger or logging.getLogger(__name__)
    cities = pd.read_csv(cities_csv)
    logger.info(f"Loaded {len(cities)} cities from {cities_csv}")

    output_shp_dir.mkdir(parents=True, exist_ok=True)

    for _, row in cities.iterrows():
        city_name = row['City']
        country_name = row.get('Country', 'Unknown')
        grid_file = Path(grid_glob_template.format(city=city_name, country=country_name))

        if not grid_file.exists():
            logger.error(f"Grid file not found for {city_name}: {grid_file}")
            continue

        logger.info(f"Processing: {city_name}, {country_name}")
        grid_cells = gpd.read_file(grid_file)
        logger.info(f"  Loaded {len(grid_cells)} grid cells")

        candidates = select_candidate_cells(grid_cells, max_candidates, logger=logger)
        candidates['City'] = city_name
        candidates['Country'] = country_name

        output_shp = output_shp_dir / f"{city_name}_sampled_cells.shp"
        candidates.to_file(output_shp)
        logger.info(f"  Wrote {len(candidates)} candidate cells to {output_shp}")


# ============================================================================
# Stage 2: morphology-stratified sampling
# ============================================================================

def get_city_size_category(total_cells: int) -> str:
    """Classify a city by its total number of grid cells."""
    if total_cells <= 500:
        return 'small'
    elif total_cells <= 3000:
        return 'medium'
    else:
        return 'large'


def get_quota(total_cells: int) -> int:
    """Sampling quota for a city, capped at its total cell count."""
    category = get_city_size_category(total_cells)
    quota = SIZE_CATEGORY_QUOTAS[category]
    return min(quota, total_cells)


def stratified_sample_city(gdf: gpd.GeoDataFrame, quota: int,
                            dimensions=STRATIFICATION_DIMENSIONS,
                            random_state: int = RANDOM_STATE,
                            logger: logging.Logger = None) -> gpd.GeoDataFrame:
    """Draw a stratified sample of grid cells from one city.

    Splits each of `dimensions` at its city-specific median into low/high,
    forming 2**len(dimensions) strata, then samples from each stratum in
    proportion to its share of the (NaN-dropped) candidate cells. Falls
    back to uniform random sampling if any stratification column is
    missing, and to "take everything" if too few cells survive NaN
    filtering.

    Returns a GeoDataFrame with added columns `{dim}_stratum` (0/1 per
    dimension, renamed to `Stratum_Shape` / `Stratum_Alignment` /
    `Stratum_Diversity` for the three canonical dimensions) and `stratum`
    (the combined 0-7 stratum ID).
    """
    logger = logger or logging.getLogger(__name__)

    for dim in dimensions:
        if dim not in gdf.columns:
            logger.warning(f"Missing stratification column {dim}; falling back to uniform random sampling")
            return gdf.sample(n=quota, random_state=random_state)

    for dim in dimensions:
        gdf[dim] = pd.to_numeric(gdf[dim], errors='coerce')

    gdf_clean = gdf.dropna(subset=dimensions)

    if len(gdf_clean) < quota:
        logger.warning(f"Only {len(gdf_clean)} cells remain after dropping NaNs; sampling all of them")
        return gdf_clean

    # Split each dimension at its median into a 0/1 stratum bit.
    for dim in dimensions:
        median = gdf_clean[dim].median()
        gdf_clean[f'{dim}_stratum'] = (gdf_clean[dim] > median).astype(int)

    # Combine the three bits into a single 0-7 stratum ID.
    gdf_clean['stratum'] = (
        gdf_clean[f'{dimensions[0]}_stratum'] * 4
        + gdf_clean[f'{dimensions[1]}_stratum'] * 2
        + gdf_clean[f'{dimensions[2]}_stratum']
    )

    stratum_counts = gdf_clean['stratum'].value_counts()
    total_cells = len(gdf_clean)

    # Allocate the quota to each stratum in proportion to its size
    # (at least 1 cell per non-empty stratum).
    stratum_quotas = {}
    for stratum in range(8):
        if stratum in stratum_counts.index:
            proportion = stratum_counts[stratum] / total_cells
            stratum_quota = max(1, int(quota * proportion))
            stratum_quotas[stratum] = min(stratum_quota, stratum_counts[stratum])
        else:
            stratum_quotas[stratum] = 0

    # Rounding can leave the total off by a few cells; put the remainder
    # into the largest stratum.
    current_total = sum(stratum_quotas.values())
    if current_total != quota:
        diff = quota - current_total
        largest_stratum = max(stratum_quotas.items(), key=lambda x: x[1])[0]
        stratum_quotas[largest_stratum] += diff

    sampled_dfs = []
    for stratum in range(8):
        if stratum_quotas[stratum] > 0:
            stratum_data = gdf_clean[gdf_clean['stratum'] == stratum]
            actual_quota = min(stratum_quotas[stratum], len(stratum_data))
            sampled_dfs.append(stratum_data.sample(n=actual_quota, random_state=random_state))

    result = pd.concat(sampled_dfs, ignore_index=True)

    # Rename the per-dimension stratum bits to their semantic names.
    result = result.rename(columns={
        f'{dimensions[0]}_stratum': 'Stratum_Shape',       # Mean_Squar -> building shape
        f'{dimensions[1]}_stratum': 'Stratum_Alignment',   # BldgStrAM -> building-street alignment
        f'{dimensions[2]}_stratum': 'Stratum_Diversity',   # StrOrientS -> street orientation diversity
    })
    # `stratum` (the combined 0-7 ID) keeps its name.

    return result


def get_country_for_city(city_name: str, city_path_csv: Path) -> str:
    """Look up a city's country from the city/path reference CSV."""
    if city_path_csv.exists():
        df = pd.read_csv(city_path_csv)
        row = df[df['City'] == city_name]
        if len(row) > 0:
            return row.iloc[0]['Country']
    return 'Unknown'


def run_sampling(shp_dir: Path, output_csv: Path, output_stats_csv: Path,
                  city_path_csv: Path, dimensions=STRATIFICATION_DIMENSIONS,
                  random_state: int = RANDOM_STATE,
                  logger: logging.Logger = None) -> None:
    """Run stratified sampling for every `{City}_sampled_cells.shp` under `shp_dir`."""
    logger = logger or logging.getLogger(__name__)

    logger.info("Stratification dimensions: %s", ', '.join(dimensions))
    logger.info(
        "City-size quotas: small (<=500) -> %d, medium (501-3000) -> %d, large (>3000) -> %d",
        SIZE_CATEGORY_QUOTAS['small'], SIZE_CATEGORY_QUOTAS['medium'], SIZE_CATEGORY_QUOTAS['large'],
    )

    shp_files = sorted(shp_dir.glob('*_sampled_cells.shp'))
    if len(shp_files) == 0:
        logger.error(f"No shapefiles found under {shp_dir}")
        sys.exit(1)
    logger.info(f"Found {len(shp_files)} cities")

    all_sampled_cells = []
    sampling_stats = []

    for shp_file in shp_files:
        city_name = shp_file.stem.replace('_sampled_cells', '')
        logger.info(f"Processing: {city_name}")

        try:
            gdf = gpd.read_file(shp_file)
            total_cells = len(gdf)
            quota = get_quota(total_cells)
            category = get_city_size_category(total_cells)
            logger.info(f"  total cells={total_cells:,}, category={category}, quota={quota}")

            sampled_gdf = stratified_sample_city(gdf, quota, dimensions, random_state, logger)
            sampled_count = len(sampled_gdf)
            logger.info(f"  sampled {sampled_count} cells")

            country = get_country_for_city(city_name, city_path_csv)
            iso3 = COUNTRY_TO_ISO3.get(country, 'UNK')
            crs = gdf.crs.to_string() if gdf.crs else 'Unknown'

            for idx, row in sampled_gdf.iterrows():
                centroid = row['geometry'].centroid

                grid_id = row['GridID'] if 'GridID' in row and pd.notna(row['GridID']) else \
                    f"{iso3}_{city_name[:4].upper()}_{idx:06d}"

                all_sampled_cells.append({
                    'GridID': grid_id,
                    'City': city_name,
                    'Country': country,
                    'CRS': crs,
                    'Center_X': centroid.x,
                    'Center_Y': centroid.y,
                    'Stratum_ID': row.get('stratum', -1),
                    'Stratum_Shape': row.get('Stratum_Shape', -1),
                    'Stratum_Alignment': row.get('Stratum_Alignment', -1),
                    'Stratum_Diversity': row.get('Stratum_Diversity', -1),
                    'Mean_Squar': row.get('Mean_Squar', None),
                    'BldgStrAM': row.get('BldgStrAM', None),
                    'StrOrientS': row.get('StrOrientS', None),
                })

            sampling_stats.append({
                'City': city_name,
                'Country': country,
                'Total_Cells': total_cells,
                'Sampled_Cells': sampled_count,
                'Sampling_Rate': f"{sampled_count / total_cells * 100:.1f}%",
                'Size_Category': category,
                'Shapefile': shp_file.name,
            })

        except Exception as e:
            logger.error(f"  Failed to process {city_name}: {e}")
            continue

    df_cells = pd.DataFrame(all_sampled_cells)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_cells.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(df_cells):,} sampled grid cells to {output_csv}")

    df_stats = pd.DataFrame(sampling_stats).sort_values('City')
    df_stats.to_csv(output_stats_csv, index=False)
    logger.info(f"Wrote sampling statistics for {len(df_stats)} cities to {output_stats_csv}")

    logger.info(
        "Totals: %d cities, %d total grid cells, %d sampled (%.1f%% overall)",
        len(df_stats), df_stats['Total_Cells'].sum(), df_stats['Sampled_Cells'].sum(),
        df_stats['Sampled_Cells'].sum() / df_stats['Total_Cells'].sum() * 100,
    )
    for cat in ('small', 'medium', 'large'):
        cat_data = df_stats[df_stats['Size_Category'] == cat]
        if len(cat_data) > 0:
            logger.info(f"  {cat:8s}: {len(cat_data):2d} cities, {cat_data['Sampled_Cells'].sum():5,} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Two-stage sampling of hexagonal grid cells by building/street morphology',
    )
    subparsers = parser.add_subparsers(dest='stage', required=True)

    p1 = subparsers.add_parser('candidates', help='Stage 1: draw a uniform-random candidate pool per city')
    p1.add_argument('--cities-csv', type=Path, required=True, help='CSV with City, Country columns')
    p1.add_argument('--grid-glob-template', type=str, required=True,
                     help='Format string with {city}/{country} placeholders locating each city\'s full hex grid shapefile')
    p1.add_argument('--output-shp-dir', type=Path, required=True)
    p1.add_argument('--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES)
    p1.add_argument('--verbose', action='store_true')

    p2 = subparsers.add_parser('stratify', help='Stage 2: morphology-stratified sampling from the candidate pool')
    p2.add_argument('--shp-dir', type=Path, required=True,
                     help='Directory of {City}_sampled_cells.shp candidate-cell shapefiles (Stage 1 output)')
    p2.add_argument('--city-path-csv', type=Path, required=True,
                     help='CSV mapping City -> Country (and other per-city paths)')
    p2.add_argument('--output-csv', type=Path, default=Path('SampleGridCell.csv'))
    p2.add_argument('--output-stats-csv', type=Path, default=Path('SamplingStatistics.csv'))
    p2.add_argument('--random-state', type=int, default=RANDOM_STATE)
    p2.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    if args.stage == 'candidates':
        run_candidate_selection(
            cities_csv=args.cities_csv,
            grid_glob_template=args.grid_glob_template,
            output_shp_dir=args.output_shp_dir,
            max_candidates=args.max_candidates,
            logger=logger,
        )
    else:
        run_sampling(
            shp_dir=args.shp_dir,
            output_csv=args.output_csv,
            output_stats_csv=args.output_stats_csv,
            city_path_csv=args.city_path_csv,
            random_state=args.random_state,
            logger=logger,
        )


if __name__ == '__main__':
    main()
