"""
Flag cities with no WSF3D (World Settlement Footprint 3D) building-height coverage.

The WSF3D height/volume rasters do not cover every settlement in the study
area (coverage gaps occur in sparsely built or desert/arid regions). A city
whose height and volume columns are entirely zero or NaN has no usable
building-height signal and should be reviewed for exclusion rather than
treated as a city with (accurately) zero building height.

Judgment rule: a city is 'no_coverage' when all 8 of
H_max, H_mean, H_std, H_sum, V_sum, V_mean, V_max, V_std
are zero or NaN across every grid cell; 'partial' when only some of them are;
'ok' otherwise.

Input
-----
Per-city metrics CSVs at ``<data_root>/<country>/<city>/Output/<city><suffix>.csv``.

Output
------
A CSV with columns [country_name, city_name, wsf3d_status, note], where
wsf3d_status is one of no_coverage / partial / ok / csv_missing / error.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'

H_V_COLS = ['H_max', 'H_mean', 'H_std', 'H_sum', 'V_sum', 'V_mean', 'V_max', 'V_std']


def check_city(csv_path: Path) -> str:
    """
    Returns:
      'no_coverage' — all 8 columns are zero/NaN
      'partial'     — some but not all of the 8 columns are zero/NaN
      'ok'          — no anomaly
      'col_missing' — none of the 8 columns are present
    """
    df = pd.read_csv(csv_path, usecols=lambda c: c in H_V_COLS, low_memory=False)

    present = [c for c in H_V_COLS if c in df.columns]
    if len(present) == 0:
        return 'col_missing'

    all_zero_null = [((df[c] == 0) | df[c].isna()).all() for c in present]

    if all(all_zero_null):
        return 'no_coverage'
    elif any(all_zero_null):
        return 'partial'
    else:
        return 'ok'


def main():
    parser = argparse.ArgumentParser(
        description='Scan per-city metrics CSVs for missing WSF3D height/volume coverage.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/ '
                              'per-city metrics CSVs.')
    parser.add_argument('--output', type=Path, required=True,
                         help='Path to write the wsf3d_status CSV report to.')
    args = parser.parse_args()

    records = []
    countries = sorted(p.name for p in args.data_root.iterdir())
    total_countries = len(countries)

    for i, country in enumerate(countries, 1):
        country_path = args.data_root / country
        if not country_path.is_dir():
            continue

        cities = sorted(p.name for p in country_path.iterdir())
        no_cov = 0

        for city in tqdm(cities, desc=f'[{i:>2}/{total_countries}] {country:<35}', leave=False):
            city_path = country_path / city
            if not city_path.is_dir():
                continue

            csv_path = city_path / 'Output' / f'{city}{SUFFIX}'
            if not csv_path.exists():
                records.append({'country_name': country, 'city_name': city,
                                 'wsf3d_status': 'csv_missing', 'note': ''})
                continue

            try:
                status = check_city(csv_path)
            except Exception as e:
                status = 'error'
                records.append({'country_name': country, 'city_name': city,
                                 'wsf3d_status': status, 'note': str(e)})
                continue

            if status != 'ok':
                no_cov += 1
            records.append({'country_name': country, 'city_name': city,
                             'wsf3d_status': status, 'note': ''})

        country_records = [r for r in records if r['country_name'] == country]
        total_c = len(country_records)
        no_cov_c = sum(1 for r in country_records if r['wsf3d_status'] == 'no_coverage')
        partial_c = sum(1 for r in country_records if r['wsf3d_status'] == 'partial')
        print(f'[{i:>2}/{total_countries}] {country:<35} total={total_c:>4}  '
              f'no_coverage={no_cov_c:>4}  partial={partial_c:>3}')

    df = pd.DataFrame(records, columns=['country_name', 'city_name', 'wsf3d_status', 'note'])
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print()
    print('=' * 60)
    print(f'Total cities:   {len(df):>6}')
    print(f'no_coverage:    {(df.wsf3d_status=="no_coverage").sum():>6}')
    print(f'partial:        {(df.wsf3d_status=="partial").sum():>6}')
    print(f'ok:             {(df.wsf3d_status=="ok").sum():>6}')
    print(f'csv_missing:    {(df.wsf3d_status=="csv_missing").sum():>6}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
