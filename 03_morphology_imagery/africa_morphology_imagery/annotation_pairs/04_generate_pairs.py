#!/usr/bin/env python3
"""Step 4: generate the anchor-based comparison-pair dataset.

Builds 5000 image pairs for human similarity annotation, across three
pair types:

- **Type 1: Anchor x Anchor (2400 pairs)** - the densely-connected core of
  the network.
  - 1a. Same-layer (960 = 8 layers x 120/layer, forced equal across
    layers): 70% cross-city / 30% same-city, to establish within-layer
    similarity structure without letting one city dominate a layer.
  - 1b. Cross-layer (1440 = 28 layer-pair combinations): allocated
    proportional to each combination's number of possible anchor pairs,
    clamped to [35, 70] pairs per combination so no single layer pairing
    is starved or over-represented; defines cross-dimension relationships.
- **Type 2: Anchor x Representative (1800 pairs)** - propagates the anchor
  skeleton's structure out to the wider representative pool, in four
  same/cross-layer x same/different-city subtypes (450 pairs each, with
  their own per-layer quotas to keep participation even).
- **Type 3: Representative x Representative (800 pairs)** - adds network
  flexibility without relying on anchors: 400 within-city pairs (captures
  intra-city gradients) and 400 between-city pairs (400).

All pairs are deduplicated globally (order-independent) as they are
generated, and again in a final pass.

Output: `comparison_pairs.csv`, columns: pair_id, img1_path, img2_path,
pair_type, layer1, layer2, city1, city2, is_anchor_pair.
"""

import argparse
import json
import random
import csv
from pathlib import Path
from collections import defaultdict
from itertools import combinations

RANDOM_SEED = 42

TARGETS = {
    'anchor_anchor_same_layer': 960,             # Type 1a: 8 layers x 120/layer
    'anchor_anchor_cross_layer': 1440,           # Type 1b: 28 combinations x ~51-52/combo
    'anchor_rep_same_layer_same_city': 450,      # Type 2a
    'anchor_rep_same_layer_diff_city': 450,      # Type 2b
    'anchor_rep_cross_layer_same_city': 450,     # Type 2c
    'anchor_rep_cross_layer_diff_city': 450,     # Type 2d
    'rep_rep_within_city': 400,                  # Type 3a
    'rep_rep_between_cities': 400,                # Type 3b
}
# 2400 + 1800 + 800 = 5000

ANCHORS_PER_LAYER = {str(i): 30 for i in range(8)}  # must match 03_select_anchors.py

# Per-layer quotas for each Type 2 subtype, keeping every layer's
# participation roughly even instead of letting larger layers dominate.
LAYER_QUOTA_TYPE2A = {'0': 56, '1': 56, '2': 56, '3': 56, '4': 56, '5': 56, '6': 58, '7': 56}  # 450
LAYER_QUOTA_TYPE2B = {'0': 56, '1': 56, '2': 56, '3': 56, '4': 56, '5': 56, '6': 58, '7': 56}  # 450
LAYER_QUOTA_TYPE2C = {'0': 56, '1': 56, '2': 56, '3': 56, '4': 56, '5': 56, '6': 58, '7': 56}  # 450 (keyed by anchor layer)
LAYER_QUOTA_TYPE2D = {'0': 56, '1': 56, '2': 56, '3': 56, '4': 56, '5': 56, '6': 58, '7': 56}  # 450 (keyed by anchor layer)

CROSS_LAYER_QUOTA_MIN = 35  # -31% from the ~51/combo mean
CROSS_LAYER_QUOTA_MAX = 70  # +37% from the mean

GLOBAL_PAIRS_SET = set()  # normalized (sorted path pair) tuples, for global deduplication


def load_data(anchors_file, city_reps_file, city_index_file):
    with open(anchors_file, 'r') as f:
        anchors_data = json.load(f)
    with open(city_reps_file, 'r') as f:
        city_reps = json.load(f)
    with open(city_index_file, 'r') as f:
        city_index = json.load(f)
    return anchors_data, city_reps, city_index


def extract_metadata(path):
    """Recover (city, layer) from a rendered image path, {City}/..._{Layer}.png."""
    p = Path(path)
    city = p.parent.name
    parts = p.stem.split('_')
    layer = parts[-1] if parts else None
    return city, layer


def normalize_pair(p1, p2):
    return tuple(sorted([p1, p2]))


def is_duplicate(p1, p2, global_set):
    return normalize_pair(p1, p2) in global_set


def add_pair(p1, p2, global_set):
    normalized = normalize_pair(p1, p2)
    if normalized in global_set:
        return False
    global_set.add(normalized)
    return True


def generate_anchor_anchor_same_layer(anchors_data, n_target):
    """Type 1a: anchor x anchor, same layer, forced equal across all 8 layers."""
    print(f"\nGenerating {n_target} anchor-anchor same-layer pairs...")
    pairs = []
    anchors_by_layer = anchors_data['by_layer']
    metadata = anchors_data['metadata']

    base_per_layer = n_target // 8
    remainder = n_target % 8
    layer_targets = {
        layer: base_per_layer + (1 if idx < remainder else 0)
        for idx, layer in enumerate(sorted(anchors_by_layer.keys(), key=int))
    }

    for layer in sorted(anchors_by_layer.keys(), key=int):
        layer_anchors = anchors_by_layer[layer]
        n_per_layer = layer_targets[layer]

        max_possible = (len(layer_anchors) * (len(layer_anchors) - 1)) // 2
        if max_possible < n_per_layer:
            print(f"  Layer {layer}: can only generate {max_possible}/{n_per_layer} pairs")

        all_pairs = list(combinations(layer_anchors, 2))
        random.shuffle(all_pairs)

        cross_city, same_city = [], []
        for p1, p2 in all_pairs:
            if is_duplicate(p1, p2, GLOBAL_PAIRS_SET):
                continue
            (cross_city if metadata[p1]['city'] != metadata[p2]['city'] else same_city).append((p1, p2))

        # 70% cross-city / 30% same-city.
        count = 0
        for p1, p2 in cross_city + same_city:
            if count >= n_per_layer:
                break
            if add_pair(p1, p2, GLOBAL_PAIRS_SET):
                c1, l1 = extract_metadata(p1)
                c2, l2 = extract_metadata(p2)
                pairs.append({'img1': p1, 'img2': p2, 'type': 'anchor_anchor_same_layer',
                              'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2, 'is_anchor_pair': True})
                count += 1
        print(f"  Layer {layer}: {count} pairs (target: {n_per_layer})")

    print(f"  Total generated: {len(pairs)} pairs")
    return pairs


def generate_anchor_anchor_cross_layer(anchors_data, n_target):
    """Type 1b: anchor x anchor, cross layer, quota-clamped per layer combination."""
    print(f"\nGenerating {n_target} anchor-anchor cross-layer pairs...")
    pairs = []
    anchors_by_layer = anchors_data['by_layer']
    metadata = anchors_data['metadata']

    layer_combos = [(str(i), str(j)) for i in range(8) for j in range(i + 1, 8)]
    combo_weights = {
        (l1, l2): ANCHORS_PER_LAYER[l1] * ANCHORS_PER_LAYER[l2] for l1, l2 in layer_combos
    }
    total_weight = sum(combo_weights.values())

    combo_targets = {}
    for l1, l2 in layer_combos:
        proportional = int(n_target * combo_weights[(l1, l2)] / total_weight)
        combo_targets[(l1, l2)] = max(CROSS_LAYER_QUOTA_MIN, min(CROSS_LAYER_QUOTA_MAX, proportional))

    total_allocated = sum(combo_targets.values())
    if total_allocated != n_target:
        diff = n_target - total_allocated
        flexible = [c for c in layer_combos if CROSS_LAYER_QUOTA_MIN < combo_targets[c] < CROSS_LAYER_QUOTA_MAX]
        if flexible:
            per_combo, remainder = diff // len(flexible), diff % len(flexible)
            for idx, combo in enumerate(flexible):
                combo_targets[combo] += per_combo + (1 if idx < remainder and diff > 0 else 0)

    total_generated = 0
    for l1, l2 in layer_combos:
        anchors_l1, anchors_l2 = anchors_by_layer.get(l1, []), anchors_by_layer.get(l2, [])
        if not anchors_l1 or not anchors_l2:
            print(f"  Layer {l1}-{l2}: missing anchors")
            continue

        n_per_combo = combo_targets[(l1, l2)]
        all_pairs = [(a1, a2) for a1 in anchors_l1 for a2 in anchors_l2]
        random.shuffle(all_pairs)

        cross_city, same_city = [], []
        for p1, p2 in all_pairs:
            if is_duplicate(p1, p2, GLOBAL_PAIRS_SET):
                continue
            (cross_city if metadata[p1]['city'] != metadata[p2]['city'] else same_city).append((p1, p2))

        count = 0
        for p1, p2 in cross_city + same_city:  # prioritize cross-city
            if count >= n_per_combo:
                break
            if add_pair(p1, p2, GLOBAL_PAIRS_SET):
                c1, layer1 = extract_metadata(p1)
                c2, layer2 = extract_metadata(p2)
                pairs.append({'img1': p1, 'img2': p2, 'type': 'anchor_anchor_cross_layer',
                              'layer1': layer1, 'layer2': layer2, 'city1': c1, 'city2': c2, 'is_anchor_pair': True})
                count += 1
        total_generated += count
        print(f"  Layer {l1}-{l2}: {count} pairs (target: {n_per_combo})")

    print(f"  Total generated: {total_generated} pairs (target: {n_target})")
    return pairs


def generate_anchor_rep_pairs(anchors_data, city_reps, targets, city_index=None):
    """Type 2: anchor x representative, 4 subtypes with per-layer quotas.

    The four subtypes are intentionally written out separately (rather than
    factored into one shared loop) because each draws from the shared
    `random` module in a specific sequence; the exact sequence of shuffles
    and samples is what the fixed random seed reproduces, so the four
    blocks below mirror the original control flow line for line rather
    than being deduplicated into a parameterized helper.
    """
    print("\nGenerating anchor-representative pairs...")
    pairs = []
    anchors_by_layer = anchors_data['by_layer']
    metadata = anchors_data['metadata']
    anchor_set = set(anchors_data['anchors'])

    rep_pool_by_layer = defaultdict(lambda: defaultdict(list))  # layer -> city -> [paths]
    for city, data in city_reps.items():
        for layer, paths in data['by_layer'].items():
            for path in paths:
                if path not in anchor_set:
                    rep_pool_by_layer[layer][city].append(path)

    full_city_pool = None
    if city_index:
        full_city_pool = defaultdict(lambda: defaultdict(list))
        for city, layers_data in city_index.items():
            for layer, paths in layers_data.items():
                for path in paths:
                    if path not in anchor_set:
                        full_city_pool[layer][city].append(path)

    all_layers = sorted(anchors_by_layer.keys(), key=int)

    # --- 2a: same layer, same city (uses the full city index, not just representatives) ---
    print("  Generating same-layer, same-city pairs (with layer quotas)...")
    count, attempts = 0, 0
    max_attempts = targets['anchor_rep_same_layer_same_city'] * 20
    layer_counts_2a = {str(i): 0 for i in range(8)}
    source_pool = full_city_pool if full_city_pool else rep_pool_by_layer

    for layer in sorted(source_pool.keys(), key=int):
        city_paths = source_pool[layer]
        layer_quota = LAYER_QUOTA_TYPE2A.get(layer, 60)
        if count >= targets['anchor_rep_same_layer_same_city']:
            break

        layer_anchors = anchors_by_layer.get(layer, [])
        random.shuffle(layer_anchors)

        for anchor_path in layer_anchors:
            if (count >= targets['anchor_rep_same_layer_same_city']
                    or layer_counts_2a[layer] >= layer_quota or attempts >= max_attempts):
                break

            anchor_city = metadata[anchor_path]['city']
            same_city_samples = city_paths.get(anchor_city, [])

            if same_city_samples:
                random.shuffle(same_city_samples)
                for sample_path in same_city_samples:
                    attempts += 1
                    if attempts >= max_attempts:
                        break
                    if not is_duplicate(anchor_path, sample_path, GLOBAL_PAIRS_SET):
                        if add_pair(anchor_path, sample_path, GLOBAL_PAIRS_SET):
                            c1, l1 = extract_metadata(anchor_path)
                            c2, l2 = extract_metadata(sample_path)
                            pairs.append({'img1': anchor_path, 'img2': sample_path,
                                          'type': 'anchor_rep_same_layer_same_city',
                                          'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2,
                                          'is_anchor_pair': True})
                            count += 1
                            layer_counts_2a[layer] += 1
                            break
        print(f"    Layer {layer}: {layer_counts_2a[layer]} pairs (quota: {layer_quota})")
    print(f"    Total generated: {count} pairs (target: {targets['anchor_rep_same_layer_same_city']})")

    # --- 2b: same layer, different city ---
    print("  Generating same-layer, diff-city pairs (with layer quotas)...")
    count, attempts = 0, 0
    max_attempts = targets['anchor_rep_same_layer_diff_city'] * 20
    layer_counts_2b = {str(i): 0 for i in range(8)}

    for layer in sorted(rep_pool_by_layer.keys(), key=int):
        city_paths = rep_pool_by_layer[layer]
        layer_quota = LAYER_QUOTA_TYPE2B.get(layer, 56)
        if count >= targets['anchor_rep_same_layer_diff_city']:
            break

        layer_anchors = anchors_by_layer.get(layer, [])
        random.shuffle(layer_anchors)

        for anchor_path in layer_anchors:
            if (count >= targets['anchor_rep_same_layer_diff_city']
                    or layer_counts_2b[layer] >= layer_quota or attempts >= max_attempts):
                break

            anchor_city = metadata[anchor_path]['city']
            other_cities = [c for c in city_paths.keys() if c != anchor_city]

            if other_cities:
                random.shuffle(other_cities)
                for other_city in other_cities:
                    if (count >= targets['anchor_rep_same_layer_diff_city']
                            or layer_counts_2b[layer] >= layer_quota or attempts >= max_attempts):
                        break
                    other_city_reps = city_paths[other_city]
                    if other_city_reps:
                        random.shuffle(other_city_reps)
                        for rep_path in other_city_reps:
                            attempts += 1
                            if attempts >= max_attempts:
                                break
                            if not is_duplicate(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                                if add_pair(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                                    c1, l1 = extract_metadata(anchor_path)
                                    c2, l2 = extract_metadata(rep_path)
                                    pairs.append({'img1': anchor_path, 'img2': rep_path,
                                                  'type': 'anchor_rep_same_layer_diff_city',
                                                  'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2,
                                                  'is_anchor_pair': True})
                                    count += 1
                                    layer_counts_2b[layer] += 1
                                    break
        print(f"    Layer {layer}: {layer_counts_2b[layer]} pairs (quota: {layer_quota})")
    print(f"    Total generated: {count} pairs (target: {targets['anchor_rep_same_layer_diff_city']})")

    # --- 2c: cross layer, same city (quota tracked by anchor layer) ---
    print("  Generating cross-layer, same-city pairs (with layer quotas)...")
    count, attempts = 0, 0
    max_attempts = targets['anchor_rep_cross_layer_same_city'] * 20
    layer_counts_2c = {str(i): 0 for i in range(8)}

    for anchor_layer in all_layers:
        layer_quota = LAYER_QUOTA_TYPE2C.get(anchor_layer, 56)
        if count >= targets['anchor_rep_cross_layer_same_city']:
            break

        layer_anchors = anchors_by_layer[anchor_layer]
        random.shuffle(layer_anchors)

        for anchor_path in layer_anchors:
            if (count >= targets['anchor_rep_cross_layer_same_city']
                    or layer_counts_2c[anchor_layer] >= layer_quota or attempts >= max_attempts):
                break

            anchor_city = metadata[anchor_path]['city']
            other_layers = [l for l in all_layers if l != anchor_layer]
            if not other_layers:
                continue
            random.shuffle(other_layers)

            for rep_layer in other_layers:
                if (count >= targets['anchor_rep_cross_layer_same_city']
                        or layer_counts_2c[anchor_layer] >= layer_quota or attempts >= max_attempts):
                    break
                same_city_reps = rep_pool_by_layer[rep_layer].get(anchor_city, [])
                if same_city_reps:
                    random.shuffle(same_city_reps)
                    for rep_path in same_city_reps:
                        attempts += 1
                        if attempts >= max_attempts:
                            break
                        if not is_duplicate(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                            if add_pair(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                                c1, l1 = extract_metadata(anchor_path)
                                c2, l2 = extract_metadata(rep_path)
                                pairs.append({'img1': anchor_path, 'img2': rep_path,
                                              'type': 'anchor_rep_cross_layer_same_city',
                                              'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2,
                                              'is_anchor_pair': True})
                                count += 1
                                layer_counts_2c[anchor_layer] += 1
                                break
        print(f"    Anchor layer {anchor_layer}: {layer_counts_2c[anchor_layer]} pairs (quota: {layer_quota})")
    print(f"    Total generated: {count} pairs (target: {targets['anchor_rep_cross_layer_same_city']})")

    # --- 2d: cross layer, different city (quota tracked by anchor layer) ---
    print("  Generating cross-layer, diff-city pairs (with layer quotas)...")
    count, attempts = 0, 0
    max_attempts = targets['anchor_rep_cross_layer_diff_city'] * 20
    layer_counts_2d = {str(i): 0 for i in range(8)}

    for anchor_layer in all_layers:
        layer_quota = LAYER_QUOTA_TYPE2D.get(anchor_layer, 56)
        if count >= targets['anchor_rep_cross_layer_diff_city']:
            break

        layer_anchors = anchors_by_layer[anchor_layer]
        random.shuffle(layer_anchors)

        for anchor_path in layer_anchors:
            if (count >= targets['anchor_rep_cross_layer_diff_city']
                    or layer_counts_2d[anchor_layer] >= layer_quota or attempts >= max_attempts):
                break

            anchor_city = metadata[anchor_path]['city']
            other_layers = [l for l in all_layers if l != anchor_layer]
            if not other_layers:
                continue
            random.shuffle(other_layers)

            for rep_layer in other_layers:
                if (count >= targets['anchor_rep_cross_layer_diff_city']
                        or layer_counts_2d[anchor_layer] >= layer_quota or attempts >= max_attempts):
                    break
                other_cities = [c for c in rep_pool_by_layer[rep_layer].keys() if c != anchor_city]
                if other_cities:
                    random.shuffle(other_cities)
                    for other_city in other_cities:
                        if (count >= targets['anchor_rep_cross_layer_diff_city']
                                or layer_counts_2d[anchor_layer] >= layer_quota or attempts >= max_attempts):
                            break
                        other_city_reps = rep_pool_by_layer[rep_layer][other_city]
                        if other_city_reps:
                            random.shuffle(other_city_reps)
                            for rep_path in other_city_reps:
                                attempts += 1
                                if attempts >= max_attempts:
                                    break
                                if not is_duplicate(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                                    if add_pair(anchor_path, rep_path, GLOBAL_PAIRS_SET):
                                        c1, l1 = extract_metadata(anchor_path)
                                        c2, l2 = extract_metadata(rep_path)
                                        pairs.append({'img1': anchor_path, 'img2': rep_path,
                                                      'type': 'anchor_rep_cross_layer_diff_city',
                                                      'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2,
                                                      'is_anchor_pair': True})
                                        count += 1
                                        layer_counts_2d[anchor_layer] += 1
                                        break
        print(f"    Anchor layer {anchor_layer}: {layer_counts_2d[anchor_layer]} pairs (quota: {layer_quota})")
    print(f"    Total generated: {count} pairs (target: {targets['anchor_rep_cross_layer_diff_city']})")

    return pairs


def generate_rep_rep_pairs(city_reps, anchors_data, targets):
    """Type 3: representative x representative, within-city and between-city."""
    print("\nGenerating representative-representative pairs...")
    pairs = []
    anchor_set = set(anchors_data['anchors'])

    rep_pool_by_city = {}
    all_non_anchor_reps = []
    for city, data in city_reps.items():
        rep_pool_by_city[city] = {}
        for layer, paths in data['by_layer'].items():
            non_anchor_paths = [p for p in paths if p not in anchor_set]
            rep_pool_by_city[city][layer] = non_anchor_paths
            all_non_anchor_reps.extend([(p, city, layer) for p in non_anchor_paths])

    # 3a: within-city.
    print("  Generating within-city pairs...")
    count, attempts = 0, 0
    max_attempts = targets['rep_rep_within_city'] * 20
    cities = list(rep_pool_by_city.keys())
    random.shuffle(cities)

    for city in cities:
        if count >= targets['rep_rep_within_city'] or attempts >= max_attempts:
            break
        city_reps_list = [p for paths in rep_pool_by_city[city].values() for p in paths]
        if len(city_reps_list) < 2:
            continue
        city_pairs = list(combinations(city_reps_list, 2))
        random.shuffle(city_pairs)

        for p1, p2 in city_pairs[:min(10, len(city_pairs))]:
            if count >= targets['rep_rep_within_city'] or attempts >= max_attempts:
                break
            attempts += 1
            if not is_duplicate(p1, p2, GLOBAL_PAIRS_SET) and add_pair(p1, p2, GLOBAL_PAIRS_SET):
                c1, l1 = extract_metadata(p1)
                c2, l2 = extract_metadata(p2)
                pairs.append({'img1': p1, 'img2': p2, 'type': 'rep_rep_within_city',
                              'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2, 'is_anchor_pair': False})
                count += 1
    print(f"    Generated {count} pairs (target: {targets['rep_rep_within_city']})")

    # 3b: between cities.
    print("  Generating between-city pairs...")
    count, attempts = 0, 0
    max_attempts = targets['rep_rep_between_cities'] * 20
    random.shuffle(all_non_anchor_reps)

    while count < targets['rep_rep_between_cities'] and attempts < max_attempts and len(all_non_anchor_reps) >= 2:
        attempts += 1
        (p1, c1, l1), (p2, c2, l2) = random.sample(all_non_anchor_reps, 2)
        if c1 == c2 or is_duplicate(p1, p2, GLOBAL_PAIRS_SET):
            continue
        if add_pair(p1, p2, GLOBAL_PAIRS_SET):
            pairs.append({'img1': p1, 'img2': p2, 'type': 'rep_rep_between_cities',
                          'layer1': l1, 'layer2': l2, 'city1': c1, 'city2': c2, 'is_anchor_pair': False})
            count += 1
    print(f"    Generated {count} pairs (target: {targets['rep_rep_between_cities']})")

    return pairs


def generate_pairs(anchors_file: Path, city_reps_file: Path, city_index_file: Path,
                    output_file: Path, random_seed: int = RANDOM_SEED):
    random.seed(random_seed)

    print(f"Generating comparison pairs (target: {sum(TARGETS.values())} pairs)")
    for ptype, count in TARGETS.items():
        print(f"  {ptype:40s}: {count}")

    anchors_data, city_reps, city_index = load_data(anchors_file, city_reps_file, city_index_file)

    all_pairs = []
    all_pairs += generate_anchor_anchor_same_layer(anchors_data, TARGETS['anchor_anchor_same_layer'])
    all_pairs += generate_anchor_anchor_cross_layer(anchors_data, TARGETS['anchor_anchor_cross_layer'])
    all_pairs += generate_anchor_rep_pairs(anchors_data, city_reps, TARGETS, city_index=city_index)
    all_pairs += generate_rep_rep_pairs(city_reps, anchors_data, TARGETS)

    print("\nFinal deduplication check...")
    unique_pairs, seen, duplicates_found = [], set(), 0
    for pair in all_pairs:
        normalized = normalize_pair(pair['img1'], pair['img2'])
        if normalized not in seen:
            seen.add(normalized)
            unique_pairs.append(pair)
        else:
            duplicates_found += 1
    print(f"Original: {len(all_pairs)}, duplicates removed: {duplicates_found}, unique: {len(unique_pairs)}")
    all_pairs = unique_pairs

    random.shuffle(all_pairs)
    for i, pair in enumerate(all_pairs):
        pair['pair_id'] = f"pair_{i:05d}"

    print(f"\nSaving {len(all_pairs)} pairs to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['pair_id', 'img1_path', 'img2_path', 'pair_type', 'layer1', 'layer2', 'city1', 'city2', 'is_anchor_pair']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in all_pairs:
            writer.writerow({
                'pair_id': pair['pair_id'], 'img1_path': pair['img1'], 'img2_path': pair['img2'],
                'pair_type': pair['type'], 'layer1': pair['layer1'], 'layer2': pair['layer2'],
                'city1': pair['city1'], 'city2': pair['city2'], 'is_anchor_pair': pair['is_anchor_pair'],
            })

    type_counts = defaultdict(int)
    for pair in all_pairs:
        type_counts[pair['type']] += 1
    print("\nPairs by type:")
    for ptype in sorted(type_counts.keys()):
        print(f"  {ptype}: {type_counts[ptype]}")
    print(f"\nTotal: {len(all_pairs)} pairs")


def main():
    parser = argparse.ArgumentParser(description='Generate the anchor-based comparison-pair dataset')
    parser.add_argument('--anchors', type=Path, default=Path('../outputs/global_anchors.json'))
    parser.add_argument('--city-reps', type=Path, default=Path('../outputs/city_representatives.json'))
    parser.add_argument('--city-index', type=Path, default=Path('../outputs/city_index.json'))
    parser.add_argument('--output', type=Path, default=Path('../outputs/comparison_pairs.csv'))
    args = parser.parse_args()
    generate_pairs(args.anchors, args.city_reps, args.city_index, args.output)


if __name__ == "__main__":
    main()
