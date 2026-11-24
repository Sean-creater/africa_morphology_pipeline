"""
Flag cities where LC_Built, LCF_Built, or LC_Shannon are entirely zero or NaN.

These three columns come from the land-cover classification and derived
land-cover-fraction/diversity layers. A city where any one of them is
uniformly zero/NaN across all grid cells indicates the land-cover source
had no usable signal for that city (rather than the city genuinely having
zero built-up land cover and zero land-cover diversity, which is not
plausible for a populated settlement) and should be reviewed for exclusion.

Input
-----
Per-city metrics CSVs at ``<data_root>/<country>/<city>/Output/<city><suffix>.csv``.

Output
------
A CSV with columns [country_name, city_name, n_abnormal, abnormal_col_names]
listing only the cities with at least one abnormal column.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'

CHECK_COLS = ['LC_Built', 'LCF_Built', 'LC_Shannon']


def main():
    parser = argparse.ArgumentParser(
        description='Scan per-city metrics CSVs for degenerate LC_Built / LCF_Built / LC_Shannon columns.')
    parser.add_argument('--data-root', type=Path, required=True,
                         help='Root directory containing <country>/<city>/Output/ '
                              'per-city metrics CSVs.')
    parser.add_argument('--output', type=Path, required=True,
                         help='Path to write the abnormal-cities CSV report to.')
    args = parser.parse_args()

    records = []
    countries = sorted(p.name for p in args.data_root.iterdir())

    for i, country in enumerate(countries, 1):
        country_path = args.data_root / country
        if not country_path.is_dir():
            continue

        cities = sorted(p.name for p in country_path.iterdir())
        bad = 0

        for city in tqdm(cities, desc=f'[{i:>2}/{len(countries)}] {country:<35}', leave=False):
            csv_path = country_path / city / 'Output' / f'{city}{SUFFIX}'
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, usecols=lambda c: c in CHECK_COLS, low_memory=False)
            except Exception:
                continue

            abnormal = []
            for col in CHECK_COLS:
                if col in df.columns:
                    if ((df[col] == 0) | df[col].isna()).all():
                        abnormal.append(col)

            if abnormal:
                bad += 1
                records.append({
                    'country_name': country,
                    'city_name': city,
                    'n_abnormal': len(abnormal),
                    'abnormal_col_names': '; '.join(abnormal),
                })

        print(f'[{i:>2}/{len(countries)}] {country:<35} cities={len(cities):>4}  abnormal={bad:>4}')

    df_out = pd.DataFrame(records, columns=['country_name', 'city_name', 'n_abnormal', 'abnormal_col_names'])
    df_out.to_csv(args.output, index=False, encoding='utf-8-sig')

    print()
    print('=' * 60)
    print(f'Total abnormal cities: {len(df_out)}')
    print()
    print('By country:')
    print(df_out['country_name'].value_counts().to_string())
    print()
    print('By abnormal column combination:')
    print(df_out['abnormal_col_names'].value_counts().to_string())
    print(f'\nOutput: {args.output}')


if __name__ == '__main__':
    main()
