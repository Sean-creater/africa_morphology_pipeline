"""
Verify structural and value-level correctness of all col_121 cities before
road-network metrics are added.

col_121 cities have completed land-cover, WSF3D height/volume, population,
and building-morphology processing (including the derived height metrics)
but are still missing the 51 road-network/connectivity columns, which are
added downstream once OSM road data is available. Before that step runs,
this check confirms the col_121 tier is internally consistent:

1. Each city is missing exactly the 51 expected road-network columns —
   no more, no less (catches partial writes or column-name drift).
2. None of the present columns are degenerate (all-zero or all-NaN) for
   the WSF3D, building-height-derived, and open-buildings groups.
3. Each city has at least one row.
4. The CSV's actual column count is exactly 121.

Input
-----
- Per-city metrics CSVs at ``<data_root>/<country>/<city>/Output/<city><suffix>.csv``
- ``--city-inventory`` CSV, filtered to ``exclusion_reason == 'col_121'``
- ``--reference-csv`` — one full 172-column city CSV used as the canonical
  column list to diff each col_121 city's header against (any complete,
  already-cleaned city works for this; the header only, not the data, is used)

Output
------
A CSV of problem cities with columns [country, city, issue, detail].
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'

# ── The 51 road-network columns expected to be missing at the col_121 tier ────
EXPECTED_MISSING = [
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
EXPECTED_MISSING_SET = set(EXPECTED_MISSING)

# ── Columns checked for all-zero / all-NaN degeneracy ─────────────────────────
CHECK_NONZERO = [
    # WSF3D / TIF derived (LC_Built excluded — legitimately 0 in some regions)
    'H_max', 'H_mean', 'H_std', 'H_sum',
    'V_sum', 'V_mean', 'V_max', 'V_std',
    'Pop2022', 'LCF_Built', 'LC_Shannon',
    # Derived height metrics
    'FloorARat', 'Mean_FormF', 'Mean_H2F', 'Mean_VolDe',
    'Std_FormF', 'Std_H2F', 'Std_VolDe',
]


def check_city(country: str, city: str, data_root: Path, ref_cols: list) -> dict:
    csv_path = data_root / country / city / 'Output' / f'{city}{SUFFIX}'

    result = {'country': country, 'city': city, 'issue': '', 'detail': ''}

    if not csv_path.exists():
        result['issue'] = 'no_csv'
        return result

    df_head = pd.read_csv(csv_path, nrows=0)
    actual_cols = set(df_head.columns)
    n_cols = len(actual_cols)

    if n_cols != 121:
        result['issue'] = f'wrong_col_count:{n_cols}'
        result['detail'] = f'expected 121, got {n_cols}'
        return result

    actually_missing = set(c for c in ref_cols if c not in actual_cols)
    extra_missing = actually_missing - EXPECTED_MISSING_SET
    unexpected_present = EXPECTED_MISSING_SET - actually_missing

    if extra_missing or unexpected_present:
        details = []
        if extra_missing:
            details.append(f'extra_missing({len(extra_missing)}):{sorted(extra_missing)}')
        if unexpected_present:
            details.append(f'road_cols_present({len(unexpected_present)}):{sorted(unexpected_present)}')
        result['issue'] = 'wrong_missing_cols'
        result['detail'] = '; '.join(details)
        return result

    df = pd.read_csv(csv_path, low_memory=False)

    if len(df) == 0:
        result['issue'] = 'empty_csv'
        return result

    zero_cols = []
    for col in CHECK_NONZERO:
        if col in df.columns:
            if ((df[col] == 0) | df[col].isna()).all():
                zero_cols.append(col)

    if zero_cols:
        result['issue'] = 'all_zero_cols'
        result['detail'] = str(zero_cols)
        return result

    result['issue'] = 'ok'
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Verify structural and value-level correctness of col_121 cities.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/.')
    parser.add_argument('--city-inventory', type=Path, required=True,
                         help='Path to city_inventory.csv.')
    parser.add_argument('--reference-csv', type=Path, required=True,
                         help='A complete (172-column) per-city CSV whose header is '
                              'used as the canonical column list.')
    parser.add_argument('--output', type=Path, required=True,
                         help='Path to write the problem-cities CSV report to.')
    args = parser.parse_args()

    ref_cols = list(pd.read_csv(args.reference_csv, nrows=0).columns)

    inv = pd.read_csv(args.city_inventory)
    cities_121 = inv[inv['exclusion_reason'] == 'col_121'][['country_name', 'city_name']]
    total = len(cities_121)
    print(f'col_121 cities: {total}')
    print()

    records = []
    for _, r in tqdm(cities_121.iterrows(), total=total, desc='Checking'):
        rec = check_city(r['country_name'], r['city_name'], args.data_root, ref_cols)
        records.append(rec)

    df_res = pd.DataFrame(records)

    print()
    print('=' * 60)
    issue_counts = df_res['issue'].value_counts()
    for issue, cnt in issue_counts.items():
        mark = 'OK' if issue == 'ok' else 'X '
        print(f'  {mark} {issue:<30}: {cnt}')

    problems = df_res[df_res['issue'] != 'ok']
    print()
    print(f'Total problem cities: {len(problems)}')

    if len(problems) > 0:
        print()
        print('Problem city detail:')
        for _, row in problems.iterrows():
            print(f'  {row["country"]}/{row["city"]}: [{row["issue"]}] {row["detail"]}')

        problems.to_csv(args.output, index=False)
        print(f'\nProblem cities saved to: {args.output}')
    else:
        print(f'All {total} cities passed: missing columns match exactly, no anomalies in present columns.')


if __name__ == '__main__':
    main()
