"""
Generate the CSV-column-name -> SHP-field-name mapping table.

The ESRI Shapefile format limits field (column) names to 10 characters.
Every metrics CSV column name is 10 characters or fewer with one exception:
``ThreeWayCnt`` (11 characters), which must be truncated to ``ThreeWayCn``
when writing SHP output. This script derives the full mapping
programmatically from the canonical 172-column list, so the truncation rule
is defined once and reused consistently by ``csv_to_shp.py`` rather than
hand-maintained in two places.

Rule: a CSV column name of 10 characters or fewer is kept unchanged as the
SHP field name; a name longer than 10 characters is truncated to its first
10 characters. The geometry column is not part of this mapping since
geopandas handles it separately.

Output
------
A CSV with columns [csv_col, shp_col, csv_len, changed, description],
written to ``data/reference/col_name_mapping.csv`` by default.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'data' / 'reference' / 'col_name_mapping.csv'

# ── Canonical 172-column list (CSV column order) ──────────────────────────────
CSV_COLS = [
    'GridID', 'center_x', 'center_y', 'City', 'Area_m2',
    'Center_X_1', 'Center_Y_1',
    # Land cover
    'LC_Shrub', 'LC_Herb', 'LC_Crop', 'LC_Built', 'LC_Bare', 'LC_Snow',
    'LC_Water', 'LC_Wetlnd', 'LC_Moss',
    'LC_FCEvNd', 'LC_FCEvBr', 'LC_FCDcNd', 'LC_FCDcBr', 'LC_FCMix', 'LC_FCUnd',
    'LC_FOEvNd', 'LC_FOEvBr', 'LC_FODcNd', 'LC_FODcBr', 'LC_FOMix', 'LC_FOUnd',
    'LC_Ocean',
    # WSF3D height
    'H_max', 'H_mean', 'H_std', 'H_sum',
    'V_sum', 'V_mean', 'V_max', 'V_std',
    # Population
    'Pop2022',
    # LCF (fine land-cover fractions)
    'LCF_Bare', 'LCF_Built', 'LCF_Crop', 'LCF_Grass', 'LCF_Moss',
    'LCF_Water', 'LCF_SeaWtr', 'LCF_Shrub', 'LCF_Snow', 'LCF_Tree',
    'LC_Shannon',
    # Building morphology - area
    'Mean_Area', 'Max_Area', 'Std_Area',
    'Mean_Perim', 'Max_Perim', 'Std_Perim',
    'Mean_Court', 'Std_Court',
    'Mean_Longe', 'Max_Longe', 'Std_Longe',
    'Mean_Circu', 'Std_Circu',
    'Mean_Squ_1', 'Std_Squ_1',
    'Mean_Compa', 'Std_Compa',
    'Mean_Equiv', 'Std_Equiv',
    'Mean_Recta', 'Std_Recta',
    'Mean_Conve', 'Std_Conve',
    'Mean_Elong', 'Std_Elong',
    'Mean_Squar', 'Std_Squar',
    'Mean_Centr', 'Std_Centr',
    'Mean_CentS', 'Std_CentS',
    'Mean_CrtYd', 'Std_CrtYd',
    'Mean_Facad', 'Std_Facad',
    'Mean_Fract', 'Std_Fract',
    'Mean_ShpId', 'Std_ShpId',
    'Mean_FormF', 'Std_FormF',
    'Mean_H2F', 'Std_H2F',
    'Mean_VolDe', 'Std_VolDe',
    'Mean_Corne', 'Max_Corne', 'Std_Corne',
    'Mean_Orien', 'Std_Orien',
    'Mean_Share', 'Max_Share', 'Std_Share',
    # Building counts/density
    'BldgCount', 'BldgDens', 'CoverRatio', 'OpenRatio',
    'PerimDens', 'CourtDens', 'ShareDens', 'FloorARat',
    # Alignment/proximity
    'Mean_Align', 'Std_Align',
    'Mean_IBDis', 'Std_IBDis',
    'Mean_NbDis', 'Std_NbDis',
    'Mean_Neigh', 'Max_Neigh', 'Std_Neigh',
    'Mean_BAdj', 'Std_BAdj',
    # Road network - density
    'StreetDens', 'NodeDens', 'StreetsPN', 'SegmentDen',
    'PlotDens', 'BldgDensN', 'PlotFrag',
    # Road network - width/openness
    'StrWidth', 'StrOpen', 'StrWidthD',
    # Road network - linearity/orientation
    'StrLinM', 'StrLinS', 'StrOrientM', 'StrOrientS',
    # Road network - building-street alignment
    'BldgStrAM', 'BldgStrAS',
    # Centrality metrics
    'BtwCentM', 'BtwCentX', 'BtwCentS',
    'ClsCentM', 'ClsCentX', 'ClsCentS',
    'StrCentM', 'StrCentX', 'StrCentS',
    # Network topology
    'NdNbDenM', 'NdNbDenS',
    'ClusterM', 'ClusterS',
    'MeanDeg', 'StdDeg',
    # Intersection counts
    'DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt',
    # Intersection ratios
    'Ratio3way', 'Ratio4way', 'CuldesacR',
    'MeanNdDst',
    # Graph indices
    'Cyclomat', 'EdgeNdRat', 'GammaIdx', 'Meshedness',
    # Edge metrics
    'EdgeBtwM', 'EdgeClsM', 'EdgeStrM', 'CuldesacL',
    # COINS
    'COINSCnt', 'COINSLen',
    # Final densities
    'DeadendDen', 'InterDens',
]

DESCRIPTIONS = {
    'GridID': 'Hex grid cell ID',
    'center_x': 'Grid cell center X (original CRS)',
    'center_y': 'Grid cell center Y (original CRS)',
    'City': 'City name',
    'Area_m2': 'Grid cell area (m2)',
    'Center_X_1': 'Grid cell center longitude (WGS84)',
    'Center_Y_1': 'Grid cell center latitude (WGS84)',
    'ThreeWayCnt': '3-way intersection count (truncated to ThreeWayCn in SHP)',
}


def build_mapping(csv_cols: list) -> pd.DataFrame:
    mapping_rows = []
    shp_names_used = set()

    for csv_col in csv_cols:
        shp_col = csv_col if len(csv_col) <= 10 else csv_col[:10]

        assert shp_col not in shp_names_used, f'Collision: {csv_col} -> {shp_col}'
        shp_names_used.add(shp_col)

        changed = csv_col != shp_col
        mapping_rows.append({
            'csv_col': csv_col,
            'shp_col': shp_col,
            'csv_len': len(csv_col),
            'changed': changed,
            'description': '',
        })

    df = pd.DataFrame(mapping_rows)
    for k, v in DESCRIPTIONS.items():
        df.loc[df['csv_col'] == k, 'description'] = v
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate the CSV-column -> SHP-field name mapping table.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                         help=f'Output CSV path (default: {DEFAULT_OUTPUT}).')
    args = parser.parse_args()

    df = build_mapping(CSV_COLS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    changed_cols = df[df['changed']]
    print(f'Total columns: {len(df)}')
    print(f'Renamed columns: {len(changed_cols)}')
    if len(changed_cols):
        print(changed_cols[['csv_col', 'shp_col']].to_string(index=False))
    print(f'\nMapping saved to: {args.output}')


if __name__ == '__main__':
    main()
