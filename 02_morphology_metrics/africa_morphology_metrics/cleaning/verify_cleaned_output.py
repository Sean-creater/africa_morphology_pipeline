"""
Run a comprehensive pass/fail QC sweep over every city in the cleaned-output
directory, as the gate before generating SHP_cleaned.

Checks performed per city:
1. Column count must be exactly 172 (the full enriched schema).
2. TIF-derived columns (H_max, H_mean, H_std, H_sum, V_sum, V_mean, V_max,
   V_std, Pop2022, LCF_Built, LC_Shannon) must not be all-zero/all-NaN.
   (LC_Built is intentionally excluded: some cities genuinely have no
   built-up land cover.)
3. Height-derived columns (FloorARat, Mean_FormF, Std_FormF, Mean_H2F,
   Std_H2F, Mean_VolDe, Std_VolDe) must not be all-zero/all-NaN.
4. Road-network columns are checked at two levels of strictness:
   - Core columns (StreetDens, NodeDens) must not be all-zero/all-NaN at the
     whole-city level.
   - The remaining connectivity columns are only flagged if all-NaN; all-zero
     is acceptable, since small cities can have legitimately empty edge
     grid cells for connectivity metrics.
5. GridID must have no duplicates within a city.

Input
-----
Per-city metrics CSVs at ``<cleaned-dir>/<country>/<city><suffix>.csv``.

Output
------
- A summary text file with overall pass/fail counts and an issue-type breakdown.
- A CSV listing every city with at least one issue.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'

EXPECTED_NCOLS = 172

# TIF-derived columns (LC_Built excluded)
TIF_COLS = [
    'H_max', 'H_mean', 'H_std', 'H_sum',
    'V_sum', 'V_mean', 'V_max', 'V_std',
    'Pop2022', 'LCF_Built', 'LC_Shannon',
]

# Height-derived columns
HGT_COLS = [
    'FloorARat', 'Mean_FormF', 'Std_FormF',
    'Mean_H2F', 'Std_H2F', 'Mean_VolDe', 'Std_VolDe',
]

# Core road-network columns (all-zero or all-NaN both flagged)
ROAD_CORE = ['StreetDens', 'NodeDens']

# Remaining road-network columns (only all-NaN flagged; all-zero accepted)
ROAD_OTHER = [
    'StreetsPN', 'SegmentDen', 'PlotDens', 'BldgDensN', 'PlotFrag',
    'StrWidth', 'StrOpen', 'StrWidthD', 'StrLinM', 'StrLinS',
    'StrOrientM', 'StrOrientS', 'BldgStrAM', 'BldgStrAS',
    'BtwCentM', 'BtwCentX', 'BtwCentS',
    'ClsCentM', 'ClsCentX', 'ClsCentS',
    'StrCentM', 'StrCentX', 'StrCentS',
    'NdNbDenM', 'NdNbDenS', 'ClusterM', 'ClusterS',
    'MeanDeg', 'StdDeg',
    'DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt',
    'Ratio3way', 'Ratio4way', 'CuldesacR', 'MeanNdDst',
    'Cyclomat', 'EdgeNdRat', 'GammaIdx', 'Meshedness',
    'EdgeBtwM', 'EdgeClsM', 'EdgeStrM', 'CuldesacL',
    'COINSCnt', 'COINSLen', 'DeadendDen', 'InterDens',
]


def check_city(csv_path: Path):
    """Returns (country, city, issues_list); an empty list means the city passed."""
    parts = csv_path.parts
    country = parts[-2]
    city = csv_path.stem.replace('_grid_250m_hex_enriched_metrics_v15', '')
    issues = []

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return country, city, [f'read_error:{e}']

    ncols = len(df.columns)
    if ncols != EXPECTED_NCOLS:
        issues.append(f'ncols={ncols}')

    if 'GridID' in df.columns and df['GridID'].duplicated().any():
        issues.append('duplicate_GridID')

    for col in TIF_COLS:
        if col not in df.columns:
            issues.append(f'missing_col:{col}')
        elif ((df[col] == 0) | df[col].isna()).all():
            issues.append(f'all_zero_nan:{col}')

    for col in HGT_COLS:
        if col not in df.columns:
            issues.append(f'missing_col:{col}')
        elif ((df[col] == 0) | df[col].isna()).all():
            issues.append(f'all_zero_nan:{col}')

    for col in ROAD_CORE:
        if col not in df.columns:
            issues.append(f'missing_col:{col}')
        elif ((df[col] == 0) | df[col].isna()).all():
            issues.append(f'road_core_all_zero_nan:{col}')

    for col in ROAD_OTHER:
        if col not in df.columns:
            issues.append(f'missing_col:{col}')
        elif df[col].isna().all():
            issues.append(f'road_all_nan:{col}')

    return country, city, issues


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive QC sweep of the cleaned-output directory.')
    parser.add_argument('--cleaned-dir', type=Path, required=True,
                         help='Root directory containing <country>/<city><suffix>.csv.')
    parser.add_argument('--output-dir', type=Path, required=True,
                         help='Directory to write verify_cleaned_summary.txt and '
                              'verify_cleaned_issues.csv into.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(args.cleaned_dir.rglob(f'*{SUFFIX}'))
    total = len(all_csvs)
    print(f'Cities in cleaned-output directory: {total}')

    ok_count = 0
    issue_rows = []

    for csv_path in tqdm(all_csvs, desc='Verifying', unit='city'):
        country, city, issues = check_city(csv_path)
        if issues:
            issue_rows.append({'country': country, 'city': city,
                                'n_issues': len(issues), 'issues': '; '.join(issues)})
        else:
            ok_count += 1

    issues_df = pd.DataFrame(issue_rows)

    print('\n' + '=' * 60)
    print(f'Passed QC: {ok_count}/{total}')
    print(f'Issues:    {len(issue_rows)}/{total}')

    ncols_issues = tif_issues = hgt_issues = road_core_iss = road_nan_iss = dup_issues = missing_issues = pd.DataFrame()
    if len(issues_df) > 0:
        ncols_issues = issues_df[issues_df['issues'].str.contains('ncols=')]
        tif_issues = issues_df[issues_df['issues'].str.contains('all_zero_nan:(H_|V_|Pop|LCF_Built|LC_Shannon)')]
        hgt_issues = issues_df[issues_df['issues'].str.contains('all_zero_nan:(FloorARat|Mean_FormF|Std_FormF|Mean_H2F|Std_H2F|Mean_VolDe|Std_VolDe)')]
        road_core_iss = issues_df[issues_df['issues'].str.contains('road_core_all_zero_nan')]
        road_nan_iss = issues_df[issues_df['issues'].str.contains('road_all_nan')]
        dup_issues = issues_df[issues_df['issues'].str.contains('duplicate_GridID')]
        missing_issues = issues_df[issues_df['issues'].str.contains('missing_col')]

        print('\nIssue breakdown:')
        print(f'  ncols != 172:               {len(ncols_issues)}')
        print(f'  TIF columns all-0/NaN:      {len(tif_issues)}')
        print(f'  Height columns all-0/NaN:   {len(hgt_issues)}')
        print(f'  Core road cols all-0/NaN:   {len(road_core_iss)}')
        print(f'  Other road cols all-NaN:    {len(road_nan_iss)}')
        print(f'  Duplicate GridID:           {len(dup_issues)}')
        print(f'  Missing columns:            {len(missing_issues)}')

        print('\nFirst 20 problem cities:')
        print(issues_df.head(20).to_string(index=False))

    summary_path = args.output_dir / 'verify_cleaned_summary.txt'
    issues_path = args.output_dir / 'verify_cleaned_issues.csv'

    with open(summary_path, 'w') as f:
        f.write(f'Cities in cleaned-output directory: {total}\n')
        f.write(f'Passed QC: {ok_count}/{total}\n')
        f.write(f'Issues:    {len(issue_rows)}/{total}\n')
        if len(issues_df) > 0:
            f.write('\nIssue breakdown:\n')
            f.write(f'  ncols != 172:               {len(ncols_issues)}\n')
            f.write(f'  TIF columns all-0/NaN:      {len(tif_issues)}\n')
            f.write(f'  Height columns all-0/NaN:   {len(hgt_issues)}\n')
            f.write(f'  Core road cols all-0/NaN:   {len(road_core_iss)}\n')
            f.write(f'  Other road cols all-NaN:    {len(road_nan_iss)}\n')
            f.write(f'  Duplicate GridID:           {len(dup_issues)}\n')
            f.write(f'  Missing columns:            {len(missing_issues)}\n')

    if len(issues_df) > 0:
        issues_df.to_csv(issues_path, index=False)
        print(f'\nIssue list saved: {issues_path}')

    print(f'Summary saved: {summary_path}')


if __name__ == '__main__':
    main()
