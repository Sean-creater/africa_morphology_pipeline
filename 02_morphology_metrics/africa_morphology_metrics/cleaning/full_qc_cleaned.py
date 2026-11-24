"""
Run an exhaustive structural, range, and internal-consistency QC audit over
every city in the cleaned-output directory.

This is the most thorough check in the pipeline, applied to data that has
already passed ``verify_cleaned_output.py``'s coarser sweep. It checks:

A. Structural integrity
   A1. Column count == 172.
   A2. Column names match the standard 172-column schema exactly (no extra,
       no missing, no renames).
   A3. GridID has no duplicates.
   A4. GridID has no NaN.
   A5. At least one row.

B. City-level non-degeneracy (whole column must not be all-zero/all-NaN)
   B1. WSF3D/population: H_max, H_mean, H_std, H_sum, V_sum, V_mean, V_max,
       V_std, Pop2022.
   B2. Land cover: LCF_Built, LC_Shannon.
   B3. Building morphology basics: BldgCount, BldgDens, CoverRatio, Mean_Area.
   B4. Height-derived: FloorARat, Mean_FormF, Std_FormF, Mean_H2F, Std_H2F,
       Mean_VolDe, Std_VolDe.
   B5. Core road network: StreetDens, NodeDens.

C. Per-cell range checks
   C1. Area_m2 > 0.
   C2. BldgCount >= 0.
   C3. BldgDens in [0, 1).
   C4. CoverRatio in [0, 1].
   C5. OpenRatio in [0, 1].
   C6. LCF_* columns (11 columns, expressed as percentages) are each >= 0,
       and their per-row sum is within [95, 105] for at least 95% of rows.
   C7. LC_Shannon >= 0 (an entropy measure, non-negative by definition).
   C8. WSF3D columns are non-negative.
   C9. Pop2022 > 0 for at least half of a city's rows (edge cells can
       legitimately have zero population; a city included in the dataset
       is expected to have population somewhere).
   C10. StreetDens, NodeDens >= 0.
   C11. CoverRatio + OpenRatio is within [0.9, 1.1] tolerance for at least
        90% of rows with both values present.
   C12/C13. Center_Y_1 / Center_X_1, when expressed as lon/lat (not a
        projected coordinate), fall within the African continental bounding
        box (lat in [-35, 38], lon in [-18, 52]).

D. Internal consistency
   D1. Where BldgCount == 0, building-derived columns (BldgDens, CoverRatio,
       Mean_Area, Mean_Perim) should be 0 or NaN for at least 95% of those rows.
   D2. Where StreetDens == 0, NodeDens should also be 0 or NaN (no roads
       implies no network nodes).
   D3. Where BldgCount > 0, Mean_Area should not be 0/NaN for more than 5%
       of those rows.

E. Road-network column rules (some columns can legitimately be all-NaN for
   very small cities, but negative values are never valid)
   E1. No road-network column may contain negative values, except
       Meshedness, GammaIdx, Cyclomat, and EdgeNdRat, which are graph-theory
       indices that can be mathematically negative.
   E2. Count columns (DeadendCnt, InterCnt, ThreeWayCnt, FourWayCnt) must be
       >= 0.
   E3. Ratio3way + Ratio4way must not exceed 1.

Input
-----
Per-city metrics CSVs at ``<cleaned-dir>/<country>/<city><suffix>.csv``.

Output
------
- ``full_qc_summary.txt`` — overall pass/fail counts and per-check trigger counts.
- ``full_qc_issues.csv`` — every city with at least one issue, tagged ERROR or WARNING.
- ``full_qc_colstats.csv`` — per-column summary statistics (mean/quantiles/anomaly
  counts) aggregated across all cities.
"""

import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'

# ── Standard 172-column schema, in the same order as the source pipeline output ──
STANDARD_COLS = [
    'GridID', 'center_x', 'center_y', 'City', 'Area_m2', 'Center_X_1', 'Center_Y_1',
    'LC_Shrub', 'LC_Herb', 'LC_Crop', 'LC_Built', 'LC_Bare', 'LC_Snow', 'LC_Water',
    'LC_Wetlnd', 'LC_Moss', 'LC_FCEvNd', 'LC_FCEvBr', 'LC_FCDcNd', 'LC_FCDcBr',
    'LC_FCMix', 'LC_FCUnd', 'LC_FOEvNd', 'LC_FOEvBr', 'LC_FODcNd', 'LC_FODcBr',
    'LC_FOMix', 'LC_FOUnd', 'LC_Ocean',
    'H_max', 'H_mean', 'H_std', 'H_sum', 'V_sum', 'V_mean', 'V_max', 'V_std', 'Pop2022',
    'LCF_Bare', 'LCF_Built', 'LCF_Crop', 'LCF_Grass', 'LCF_Moss', 'LCF_Water',
    'LCF_SeaWtr', 'LCF_Shrub', 'LCF_Snow', 'LCF_Tree', 'LC_Shannon',
    'Mean_Area', 'Max_Area', 'Std_Area', 'Mean_Perim', 'Max_Perim', 'Std_Perim',
    'Mean_Court', 'Std_Court', 'Mean_Longe', 'Max_Longe', 'Std_Longe',
    'Mean_Circu', 'Std_Circu', 'Mean_Squ_1', 'Std_Squ_1', 'Mean_Compa', 'Std_Compa',
    'Mean_Equiv', 'Std_Equiv', 'Mean_Recta', 'Std_Recta', 'Mean_Conve', 'Std_Conve',
    'Mean_Elong', 'Std_Elong', 'Mean_Squar', 'Std_Squar', 'Mean_Centr', 'Std_Centr',
    'Mean_CentS', 'Std_CentS', 'Mean_CrtYd', 'Std_CrtYd', 'Mean_Facad', 'Std_Facad',
    'Mean_Fract', 'Std_Fract', 'Mean_ShpId', 'Std_ShpId',
    'Mean_FormF', 'Std_FormF', 'Mean_H2F', 'Std_H2F', 'Mean_VolDe', 'Std_VolDe',
    'Mean_Corne', 'Max_Corne', 'Std_Corne', 'Mean_Orien', 'Std_Orien',
    'Mean_Share', 'Max_Share', 'Std_Share',
    'BldgCount', 'BldgDens', 'CoverRatio', 'OpenRatio', 'PerimDens', 'CourtDens', 'ShareDens',
    'FloorARat',
    'Mean_Align', 'Std_Align', 'Mean_IBDis', 'Std_IBDis', 'Mean_NbDis', 'Std_NbDis',
    'Mean_Neigh', 'Max_Neigh', 'Std_Neigh', 'Mean_BAdj', 'Std_BAdj',
    'StreetDens', 'NodeDens', 'StreetsPN', 'SegmentDen', 'PlotDens', 'BldgDensN', 'PlotFrag',
    'StrWidth', 'StrOpen', 'StrWidthD', 'StrLinM', 'StrLinS', 'StrOrientM', 'StrOrientS',
    'BldgStrAM', 'BldgStrAS',
    'BtwCentM', 'BtwCentX', 'BtwCentS', 'ClsCentM', 'ClsCentX', 'ClsCentS',
    'StrCentM', 'StrCentX', 'StrCentS', 'NdNbDenM', 'NdNbDenS', 'ClusterM', 'ClusterS',
    'MeanDeg', 'StdDeg', 'DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt',
    'Ratio3way', 'Ratio4way', 'CuldesacR', 'MeanNdDst', 'Cyclomat', 'EdgeNdRat',
    'GammaIdx', 'Meshedness', 'EdgeBtwM', 'EdgeClsM', 'EdgeStrM', 'CuldesacL',
    'COINSCnt', 'COINSLen', 'DeadendDen', 'InterDens',
]
STANDARD_SET = set(STANDARD_COLS)

LCF_COLS = ['LCF_Bare', 'LCF_Built', 'LCF_Crop', 'LCF_Grass', 'LCF_Moss',
            'LCF_Water', 'LCF_SeaWtr', 'LCF_Shrub', 'LCF_Snow', 'LCF_Tree']

ROAD_51 = [
    'StreetDens', 'NodeDens', 'StreetsPN', 'SegmentDen', 'PlotDens', 'BldgDensN', 'PlotFrag',
    'StrWidth', 'StrOpen', 'StrWidthD', 'StrLinM', 'StrLinS', 'StrOrientM', 'StrOrientS',
    'BldgStrAM', 'BldgStrAS',
    'BtwCentM', 'BtwCentX', 'BtwCentS', 'ClsCentM', 'ClsCentX', 'ClsCentS',
    'StrCentM', 'StrCentX', 'StrCentS', 'NdNbDenM', 'NdNbDenS', 'ClusterM', 'ClusterS',
    'MeanDeg', 'StdDeg', 'DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt',
    'Ratio3way', 'Ratio4way', 'CuldesacR', 'MeanNdDst', 'Cyclomat', 'EdgeNdRat',
    'GammaIdx', 'Meshedness', 'EdgeBtwM', 'EdgeClsM', 'EdgeStrM', 'CuldesacL',
    'COINSCnt', 'COINSLen', 'DeadendDen', 'InterDens',
]
COUNT_COLS = ['DeadendCnt', 'InterCnt', 'ThreeWayCnt', 'FourWayCnt']

# Per-city (mean, min, max, nan_pct, neg_cnt, zero_pct) records, keyed by column.
col_accumulator = defaultdict(list)


def check_city(csv_path: Path):
    """
    Run the full audit on one city CSV.
    Returns (country, city, n_rows, issues_dict) where issues_dict maps
    check_id -> human-readable description.
    """
    parts = csv_path.parts
    country = parts[-2]
    city = csv_path.stem.replace('_grid_250m_hex_enriched_metrics_v15', '')
    issues = {}

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        issues['A0_read_error'] = str(e)
        return country, city, 0, issues

    n = len(df)

    # ═══ A. Structural integrity ═══
    if len(df.columns) != 172:
        issues['A1_ncols'] = f'ncols={len(df.columns)} (expected 172)'

    actual_set = set(df.columns)
    missing = STANDARD_SET - actual_set
    extra = actual_set - STANDARD_SET
    if missing:
        issues['A2_missing_cols'] = f'{sorted(missing)}'
    if extra:
        issues['A2_extra_cols'] = f'{sorted(extra)}'

    if 'GridID' in df.columns:
        dup = df['GridID'].duplicated().sum()
        if dup > 0:
            issues['A3_gridid_dup'] = f'{dup} duplicate GridIDs'

    if 'GridID' in df.columns and df['GridID'].isna().any():
        issues['A4_gridid_nan'] = f'{df["GridID"].isna().sum()} NaN GridIDs'

    if n == 0:
        issues['A5_empty'] = 'no rows'
        return country, city, n, issues

    # ═══ B. City-level all-zero/all-NaN checks ═══
    def all_zero_nan(col):
        if col not in df.columns:
            return False
        s = df[col]
        return ((s == 0) | s.isna()).all()

    for col in ['H_max', 'H_mean', 'H_std', 'H_sum', 'V_sum', 'V_mean', 'V_max', 'V_std', 'Pop2022']:
        if all_zero_nan(col):
            issues[f'B1_all_zero_nan_{col}'] = f'{col} entire column is 0/NaN'

    for col in ['LCF_Built', 'LC_Shannon']:
        if all_zero_nan(col):
            issues[f'B2_all_zero_nan_{col}'] = f'{col} entire column is 0/NaN'

    for col in ['BldgCount', 'BldgDens', 'CoverRatio', 'Mean_Area']:
        if all_zero_nan(col):
            issues[f'B3_all_zero_nan_{col}'] = f'{col} entire column is 0/NaN'

    for col in ['FloorARat', 'Mean_FormF', 'Std_FormF', 'Mean_H2F', 'Std_H2F', 'Mean_VolDe', 'Std_VolDe']:
        if all_zero_nan(col):
            issues[f'B4_all_zero_nan_{col}'] = f'{col} entire column is 0/NaN'

    for col in ['StreetDens', 'NodeDens']:
        if all_zero_nan(col):
            issues[f'B5_all_zero_nan_{col}'] = f'{col} entire column is 0/NaN'

    # ═══ C. Per-cell range checks ═══
    if 'Area_m2' in df.columns:
        bad = (df['Area_m2'] <= 0).sum()
        if bad > 0:
            issues['C1_area_le0'] = f'{bad} rows Area_m2<=0'

    if 'BldgCount' in df.columns:
        bad = (df['BldgCount'] < 0).sum()
        if bad > 0:
            issues['C2_bldgcount_neg'] = f'{bad} rows BldgCount<0'

    if 'BldgDens' in df.columns:
        v = df['BldgDens'].dropna()
        bad_neg = (v < 0).sum()
        bad_hi = (v >= 1).sum()
        if bad_neg:
            issues['C3_bldgdens_neg'] = f'{bad_neg} rows BldgDens<0'
        if bad_hi:
            issues['C3_bldgdens_ge1'] = f'{bad_hi} rows BldgDens>=1'

    if 'CoverRatio' in df.columns:
        v = df['CoverRatio'].dropna()
        bad_neg = (v < 0).sum()
        bad_hi = (v > 1).sum()
        if bad_neg:
            issues['C4_coverratio_neg'] = f'{bad_neg} rows CoverRatio<0'
        if bad_hi:
            issues['C4_coverratio_gt1'] = f'{bad_hi} rows CoverRatio>1'

    if 'OpenRatio' in df.columns:
        v = df['OpenRatio'].dropna()
        bad_neg = (v < 0).sum()
        bad_hi = (v > 1).sum()
        if bad_neg:
            issues['C5_openratio_neg'] = f'{bad_neg} rows OpenRatio<0'
        if bad_hi:
            issues['C5_openratio_gt1'] = f'{bad_hi} rows OpenRatio>1'

    lcf_present = [c for c in LCF_COLS if c in df.columns]
    if lcf_present:
        for col in lcf_present:
            neg = (df[col].fillna(0) < 0).sum()
            if neg > 0:
                issues[f'C6_lcf_neg_{col}'] = f'{neg} rows {col}<0'
        lcf_sum = df[lcf_present].fillna(0).sum(axis=1)
        # LCF columns are percentages (0-100); row sum should be ~100.
        bad_sum = ((lcf_sum < 95) | (lcf_sum > 105)).sum()
        if bad_sum > n * 0.05:
            issues['C6_lcf_sum_off'] = f'{bad_sum}/{n} rows LCF% sum not in [95,105]'

    if 'LC_Shannon' in df.columns:
        bad = (df['LC_Shannon'].dropna() < 0).sum()
        if bad > 0:
            issues['C7_shannon_neg'] = f'{bad} rows LC_Shannon<0'

    for col in ['H_max', 'H_mean', 'H_std', 'H_sum', 'V_sum', 'V_mean', 'V_max', 'V_std']:
        if col in df.columns:
            bad = (df[col].dropna() < 0).sum()
            if bad > 0:
                issues[f'C8_wsf3d_neg_{col}'] = f'{bad} rows {col}<0'

    if 'Pop2022' in df.columns:
        bad = (df['Pop2022'].dropna() <= 0).sum()
        if bad > n * 0.5:
            issues['C9_pop_le0'] = f'{bad}/{n} rows Pop2022<=0 (>50%)'

    for col in ['StreetDens', 'NodeDens']:
        if col in df.columns:
            bad = (df[col].dropna() < 0).sum()
            if bad > 0:
                issues[f'C10_road_neg_{col}'] = f'{bad} rows {col}<0'

    if 'CoverRatio' in df.columns and 'OpenRatio' in df.columns:
        s = df['CoverRatio'].fillna(0) + df['OpenRatio'].fillna(0)
        mask = df['CoverRatio'].notna() & df['OpenRatio'].notna()
        if mask.sum() > 0:
            bad = ((s[mask] < 0.9) | (s[mask] > 1.1)).sum()
            if bad > mask.sum() * 0.1:
                issues['C11_cover_open_sum'] = f'{bad}/{mask.sum()} rows CoverRatio+OpenRatio not~1 (>10%)'

    # center_x/center_y are projected coordinates (metres) and are not range-checked;
    # Center_X_1/Center_Y_1 are lon/lat and are checked only when they look like degrees.
    if 'Center_Y_1' in df.columns:
        v = df['Center_Y_1'].dropna()
        if v.abs().max() < 200:
            bad = ((v < -35) | (v > 38)).sum()
            if bad > 0:
                issues['C12_lat_out_africa'] = f'{bad} rows Center_Y_1 out of Africa range [-35,38]'

    if 'Center_X_1' in df.columns:
        v = df['Center_X_1'].dropna()
        if v.abs().max() < 200:
            bad = ((v < -18) | (v > 52)).sum()
            if bad > 0:
                issues['C13_lon_out_africa'] = f'{bad} rows Center_X_1 out of Africa range [-18,52]'

    # ═══ D. Internal consistency ═══
    if 'BldgCount' in df.columns:
        no_bldg = df['BldgCount'].fillna(0) == 0
        if no_bldg.sum() > 0:
            for col in ['BldgDens', 'CoverRatio', 'Mean_Area', 'Mean_Perim']:
                if col in df.columns:
                    bad = (df.loc[no_bldg, col].fillna(0) > 0).sum()
                    if bad > no_bldg.sum() * 0.05:
                        issues[f'D1_nobldg_nonzero_{col}'] = f'{bad} rows {col}>0 when BldgCount=0'

    if 'StreetDens' in df.columns and 'NodeDens' in df.columns:
        no_road = df['StreetDens'].fillna(0) == 0
        if no_road.sum() > 0:
            bad = (df.loc[no_road, 'NodeDens'].fillna(0) > 0).sum()
            if bad > 0:
                issues['D2_noroad_nodedens'] = f'{bad} rows NodeDens>0 when StreetDens=0'

    if 'BldgCount' in df.columns and 'Mean_Area' in df.columns:
        has_bldg = df['BldgCount'].fillna(0) > 0
        if has_bldg.sum() > 0:
            bad = (df.loc[has_bldg, 'Mean_Area'].fillna(0) == 0).sum()
            if bad > has_bldg.sum() * 0.05:
                issues['D3_bldg_no_meanarea'] = f'{bad}/{has_bldg.sum()} rows Mean_Area=0 when BldgCount>0'

    # ═══ E. Road-network column rules ═══
    # Meshedness/GammaIdx/Cyclomat/EdgeNdRat are graph-theory indices that can be
    # mathematically negative and are exempt from the non-negativity check.
    ROAD_NEG_EXEMPT = {'Meshedness', 'GammaIdx', 'Cyclomat', 'EdgeNdRat'}
    for col in ROAD_51:
        if col in df.columns and col not in ROAD_NEG_EXEMPT:
            bad = (df[col].dropna() < 0).sum()
            if bad > 0:
                issues[f'E1_road_neg_{col}'] = f'{bad} rows {col}<0'

    for col in COUNT_COLS:
        if col in df.columns:
            bad = (df[col].dropna() < 0).sum()
            if bad > 0:
                issues[f'E2_count_neg_{col}'] = f'{bad} rows {col}<0'

    if 'Ratio3way' in df.columns and 'Ratio4way' in df.columns:
        mask = df['Ratio3way'].notna() & df['Ratio4way'].notna()
        if mask.sum() > 0:
            s = df.loc[mask, 'Ratio3way'] + df.loc[mask, 'Ratio4way']
            bad = (s > 1.01).sum()
            if bad > 0:
                issues['E3_ratio_sum_gt1'] = f'{bad} rows Ratio3way+Ratio4way>1'

    for col in STANDARD_COLS:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, 'float64', 'int64']:
            v = df[col].dropna()
            if len(v) > 0:
                col_accumulator[col].append({
                    'mean': float(v.mean()),
                    'p25': float(v.quantile(0.25)),
                    'p75': float(v.quantile(0.75)),
                    'min': float(v.min()),
                    'max': float(v.max()),
                    'nan_pct': float(df[col].isna().mean() * 100),
                    'neg_cnt': int((v < 0).sum()),
                    'zero_pct': float((v == 0).mean() * 100),
                })

    return country, city, n, issues


def main():
    parser = argparse.ArgumentParser(
        description='Exhaustive structural, range, and consistency QC audit of the '
                    'cleaned-output directory.')
    parser.add_argument('--cleaned-dir', type=Path, required=True,
                         help='Root directory containing <country>/<city><suffix>.csv.')
    parser.add_argument('--output-dir', type=Path, required=True,
                         help='Directory to write full_qc_summary.txt, '
                              'full_qc_issues.csv, and full_qc_colstats.csv into.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(args.cleaned_dir.rglob(f'*{SUFFIX}'))
    total = len(all_csvs)
    print(f'Cities in cleaned-output directory: {total}')

    ok_count = 0
    issue_rows = []
    total_rows = 0

    for csv_path in tqdm(all_csvs, desc='Full QC', unit='city'):
        country, city, n_rows, issues = check_city(csv_path)
        total_rows += n_rows
        if not issues:
            ok_count += 1
        else:
            severity = 'WARNING'
            for k in issues:
                if k.startswith('A') or k.startswith('B') or k.startswith('C1') or k.startswith('C8') or k.startswith('C12') or k.startswith('C13'):
                    severity = 'ERROR'
                    break
            issue_rows.append({
                'country': country,
                'city': city,
                'n_rows': n_rows,
                'severity': severity,
                'n_issues': len(issues),
                'checks': '; '.join(f'{k}:{v}' for k, v in issues.items()),
            })

    issues_df = pd.DataFrame(issue_rows) if issue_rows else pd.DataFrame()

    print('\n' + '=' * 65)
    print(f'Passed all checks: {ok_count:,}/{total:,}')
    print(f'Cities with issues: {len(issue_rows):,}/{total:,}')
    print(f'Total grid cells:   {total_rows:,}')

    check_counts = {}
    if len(issues_df) > 0:
        errors = issues_df[issues_df['severity'] == 'ERROR']
        warns = issues_df[issues_df['severity'] == 'WARNING']
        print(f'\n  ERROR   (structural/severe value issues): {len(errors)}')
        print(f'  WARNING (consistency/range flags):         {len(warns)}')

        for row in issue_rows:
            for part in row['checks'].split('; '):
                cid = part.split(':')[0]
                check_counts[cid] = check_counts.get(cid, 0) + 1
        print('\nPer-check trigger counts (number of cities):')
        for cid, cnt in sorted(check_counts.items(), key=lambda x: -x[1]):
            print(f'  {cid:<35}: {cnt}')

        if len(errors) > 0:
            print('\nFirst 30 ERROR cities:')
            print(errors.head(30)[['country', 'city', 'n_rows', 'checks']].to_string(index=False))

    colstats_rows = []
    for col in STANDARD_COLS:
        if col not in col_accumulator:
            colstats_rows.append({'col': col, 'n_cities': 0})
            continue
        records = col_accumulator[col]
        n_cities = len(records)
        means = [r['mean'] for r in records]
        mins = [r['min'] for r in records]
        maxs = [r['max'] for r in records]
        nan_pcts = [r['nan_pct'] for r in records]
        neg_cnts = [r['neg_cnt'] for r in records]
        zero_pcts = [r['zero_pct'] for r in records]
        colstats_rows.append({
            'col': col,
            'n_cities': n_cities,
            'global_mean': round(float(np.mean(means)), 6),
            'global_min': round(float(np.min(mins)), 6),
            'global_max': round(float(np.max(maxs)), 6),
            'avg_nan_pct': round(float(np.mean(nan_pcts)), 2),
            'total_neg_rows': int(np.sum(neg_cnts)),
            'avg_zero_pct': round(float(np.mean(zero_pcts)), 2),
        })
    colstats_df = pd.DataFrame(colstats_rows)

    summary_path = args.output_dir / 'full_qc_summary.txt'
    issues_path = args.output_dir / 'full_qc_issues.csv'
    colstats_path = args.output_dir / 'full_qc_colstats.csv'

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Cities in cleaned-output directory: {total}\n')
        f.write(f'Passed all checks:  {ok_count}\n')
        f.write(f'Cities with issues: {len(issue_rows)}\n')
        f.write(f'Total grid cells:   {total_rows}\n')
        if len(issues_df) > 0:
            errors = issues_df[issues_df['severity'] == 'ERROR']
            warns = issues_df[issues_df['severity'] == 'WARNING']
            f.write(f'ERROR:   {len(errors)}\n')
            f.write(f'WARNING: {len(warns)}\n')
            f.write('\nPer-check trigger counts (number of cities):\n')
            for cid, cnt in sorted(check_counts.items(), key=lambda x: -x[1]):
                f.write(f'  {cid:<35}: {cnt}\n')

    if len(issues_df) > 0:
        issues_df.to_csv(issues_path, index=False, encoding='utf-8')
        print(f'\nIssue list:   {issues_path}')

    colstats_df.to_csv(colstats_path, index=False, encoding='utf-8')
    print(f'Column stats: {colstats_path}')
    print(f'Summary:      {summary_path}')


if __name__ == '__main__':
    main()
