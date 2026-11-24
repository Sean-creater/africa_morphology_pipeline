#!/usr/bin/env python3
"""Step 5: validate the generated comparison-pair dataset.

Checks, in order: total pair count against the 5000 target, per-type count
against each type's target, per-layer participation balance, per-city
participation balance, absence of duplicate pairs, and (on a 100-pair
sample) that referenced image files actually exist on disk.
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict

EXPECTED_TOTAL = 5000
EXPECTED_TYPE_COUNTS = {
    'anchor_anchor_same_layer': 960,
    'anchor_anchor_cross_layer': 1440,
    'anchor_rep_same_layer_same_city': 450,
    'anchor_rep_same_layer_diff_city': 450,
    'anchor_rep_cross_layer_same_city': 450,
    'anchor_rep_cross_layer_diff_city': 450,
    'rep_rep_within_city': 400,
    'rep_rep_between_cities': 400,
}


def validate_pairs(pairs_file: Path) -> None:
    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))

    print(f"Total pairs: {len(pairs)}")

    # Check 1: total count.
    print("\nCheck 1: Total Count")
    actual = len(pairs)
    diff = actual - EXPECTED_TOTAL
    status = "OK" if abs(diff) < 150 else "WARN"
    print(f"[{status}] expected ~{EXPECTED_TOTAL}, actual {actual} ({diff:+d}, {diff / EXPECTED_TOTAL * 100:+.1f}%)")

    # Check 2: per-type distribution.
    print("\nCheck 2: Pair Type Distribution")
    type_counts = defaultdict(int)
    for pair in pairs:
        type_counts[pair['pair_type']] += 1
    for ptype in sorted(type_counts.keys()):
        actual_t = type_counts[ptype]
        expected_t = EXPECTED_TYPE_COUNTS.get(ptype, 0)
        diff_t = actual_t - expected_t
        status = "OK" if abs(diff_t) < 50 else "WARN"
        pct = (diff_t / expected_t * 100) if expected_t else 0
        print(f"[{status}] {ptype:40s}: {actual_t:4d} (expected {expected_t}, {diff_t:+4d}, {pct:+5.1f}%)")

    # Check 3: layer participation balance.
    print("\nCheck 3: Layer Balance")
    layer_participation = defaultdict(int)
    for pair in pairs:
        layer_participation[pair['layer1']] += 1
        layer_participation[pair['layer2']] += 1
    avg_participation = sum(layer_participation.values()) / 8
    for layer in sorted(layer_participation.keys(), key=int):
        count = layer_participation[layer]
        status = "OK" if abs(count - avg_participation) < avg_participation * 0.15 else "WARN"
        print(f"[{status}] layer {layer}: {count:4d} (avg {avg_participation:.0f})")

    # Check 4: city participation balance.
    print("\nCheck 4: City Participation Balance")
    city_participation = defaultdict(int)
    for pair in pairs:
        city_participation[pair['city1']] += 1
        city_participation[pair['city2']] += 1
    participations = list(city_participation.values())
    avg_city = sum(participations) / len(participations)
    min_city, max_city = min(participations), max(participations)
    print(f"avg={avg_city:.1f}, min={min_city}, max={max_city}")

    # Check 5: duplicate pairs.
    print("\nCheck 5: Duplicate Pairs")
    pair_set = set()
    duplicates = 0
    for pair in pairs:
        key = tuple(sorted([pair['img1_path'], pair['img2_path']]))
        if key in pair_set:
            duplicates += 1
        else:
            pair_set.add(key)
    print(f"[{'OK' if duplicates == 0 else 'WARN'}] duplicate pairs: {duplicates}")

    # Check 6: file existence, sampled.
    print("\nCheck 6: File Existence (sample of 100 pairs)")
    sample_size = min(100, len(pairs))
    sampled_pairs = random.sample(pairs, sample_size)
    missing_files = [
        pair[key] for pair in sampled_pairs for key in ('img1_path', 'img2_path')
        if not Path(pair[key]).exists()
    ]
    print(f"[{'OK' if not missing_files else 'WARN'}] checked {sample_size * 2}, missing {len(missing_files)}")
    if missing_files:
        for mf in missing_files[:5]:
            print(f"  missing: {mf}")

    # Summary.
    checks = [
        abs(actual - EXPECTED_TOTAL) < 150,
        duplicates == 0,
        len(missing_files) == 0,
        max_city - min_city < avg_city * 1.5,
    ]
    print(f"\n{sum(checks)}/{len(checks)} checks passed")


def main():
    parser = argparse.ArgumentParser(description='Validate the generated comparison-pair dataset')
    parser.add_argument('--pairs', type=Path, default=Path('../outputs/comparison_pairs.csv'))
    args = parser.parse_args()
    validate_pairs(args.pairs)


if __name__ == "__main__":
    main()
