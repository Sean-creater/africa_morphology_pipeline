"""Inventory per-city column counts and missing or all-zero raster indicators.

Its output provides the ``n_cols`` input to ``derive_165_column_metrics.py``.
"""

import argparse
from pathlib import Path

import pandas as pd


ABNORMAL_CHECK_COLS = [
    'LC_Built', 'LC_Crop', 'H_max', 'H_mean', 'H_std', 'H_sum',
    'V_sum', 'V_mean', 'V_max', 'V_std', 'Pop2022', 'LCF_Built',
    'LCF_Grass', 'LCF_Tree', 'LC_Shannon',
]
SUFFIX = '_grid_250m_hex_enriched_metrics_v15.csv'


def check_city(csv_path: Path) -> dict:
    """Return the original inventory fields for a single city CSV."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        return {'n_cols': None, 'n_abnormal_cols': None, 'note': str(exc)}

    abnormal = sum(
        col not in df or ((df[col] == 0) | df[col].isna()).all()
        for col in ABNORMAL_CHECK_COLS
    )
    return {'n_cols': len(df.columns), 'n_abnormal_cols': abnormal, 'note': ''}


def build_summary(data_root: Path) -> pd.DataFrame:
    records = []
    for country_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for city_dir in sorted(p for p in country_dir.iterdir() if p.is_dir()):
            output_dir = city_dir / 'Output'
            if not output_dir.is_dir():
                continue
            csv_path = output_dir / f'{city_dir.name}{SUFFIX}'
            if not csv_path.exists():
                candidates = sorted(output_dir.glob('*.csv'))
                if not candidates:
                    result = {'n_cols': None, 'n_abnormal_cols': None,
                              'note': 'No CSV file found'}
                else:
                    result = check_city(candidates[0])
            else:
                result = check_city(csv_path)
            records.append({'country_name': country_dir.name,
                            'city_name': city_dir.name, **result})
    return pd.DataFrame(records, columns=[
        'country_name', 'city_name', 'n_cols', 'n_abnormal_cols', 'note'
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Wrote {len(summary)} city records to {args.output}')


if __name__ == '__main__':
    main()
