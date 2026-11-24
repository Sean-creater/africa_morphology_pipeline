#!/usr/bin/env python3
"""Step 3: select global anchor cells from the city representatives.

Selects 240 anchor cells (30 per morphological layer, uniformly) from the
representative pool built in step 2. Anchors are the "skeleton" of the
comparison-pair network: densely cross-connected in step 4 so that
similarity judgments propagate across the whole dataset rather than staying
confined to isolated per-city clusters.

Within each layer, anchors are drawn with a target regional balance
(North/West/East/Central/Southern Africa) so no single region dominates a
layer's anchor set.

Output: `global_anchors.json`,
`{"anchors": [path, ...], "by_layer": {layer: [path, ...]},
  "metadata": {path: {"city", "layer", "region"}}}`.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

ANCHORS_PER_LAYER = 30  # uniform across all 8 layers; total = 240
RANDOM_SEED = 42

REGIONS = {
    'North Africa': ['Cairo', 'Algiers', 'Tunis', 'Oran', 'Sfax', 'Aswan'],
    'West Africa': ['Lagos', 'Accra', 'Abidjan', 'Dakar', 'Ouagadougou',
                    'Sokode', 'Kara', 'Ziguinchor', 'Korhogo',
                    'Parakou', 'Bolgatanga', 'Ho', 'Wa', 'Bafoussam'],
    'East Africa': ['Nairobi', 'Addis Ababa', 'Dar es Salaam',
                    'Kigali', 'Arusha', 'Mbarara', 'Bahir Dar', 'Iringa',
                    'Bukoba', 'Mwanza'],
    'Central Africa': ['Bukavu', 'Pointe Noire', 'Kinshasa'],
    'Southern Africa': ['Johannesburg', 'Durban', 'Francistown', 'Maun', 'Livingstone', 'Luanda', 'Lusaka'],
}
CITY_TO_REGION = {city: region for region, cities in REGIONS.items() for city in cities}

# Target share of each layer's anchors coming from each region.
REGIONAL_PROPORTIONS = {
    'North Africa': 6 / 30,
    'West Africa': 7 / 30,
    'East Africa': 8 / 30,
    'Central Africa': 4 / 30,
    'Southern Africa': 5 / 30,
}


def select_anchors(city_reps_file: Path, output_file: Path,
                    anchors_per_layer: int = ANCHORS_PER_LAYER,
                    random_seed: int = RANDOM_SEED) -> dict:
    random.seed(random_seed)

    with open(city_reps_file, 'r') as f:
        city_reps = json.load(f)

    print(f"Selecting anchors from {len(city_reps)} cities...")

    layer_pool = defaultdict(list)  # layer -> [(path, city), ...]
    for city, data in city_reps.items():
        for layer, paths in data['by_layer'].items():
            for path in paths:
                layer_pool[layer].append((path, city))

    anchors_by_layer = {}
    all_anchors = []
    metadata = {}

    for layer in sorted(layer_pool.keys(), key=int):
        available = layer_pool[layer]
        target = anchors_per_layer
        print(f"\nLayer {layer}: {len(available)} candidates, target={target}")

        if len(available) < target:
            print(f"  Only {len(available)} available, taking all")
            selected = available
        else:
            layer_regional_targets = {
                region: max(1, round(target * prop)) for region, prop in REGIONAL_PROPORTIONS.items()
            }
            total_allocated = sum(layer_regional_targets.values())
            if total_allocated != target:
                largest_region = max(layer_regional_targets, key=layer_regional_targets.get)
                layer_regional_targets[largest_region] += (target - total_allocated)

            by_region = defaultdict(list)
            for path, city in available:
                by_region[CITY_TO_REGION.get(city, 'Unknown')].append((path, city))

            selected = []
            for region, region_target in layer_regional_targets.items():
                candidates = by_region[region]
                if not candidates:
                    print(f"  {region}: no candidates available")
                    continue
                n_select = min(region_target, len(candidates))
                selected.extend(random.sample(candidates, n_select))
                print(f"  {region}: {n_select}/{region_target} selected")

            if len(selected) < target:
                remaining = [x for x in available if x not in selected]
                n_needed = target - len(selected)
                if remaining:
                    extra = random.sample(remaining, min(n_needed, len(remaining)))
                    selected.extend(extra)
                    print(f"  Added {len(extra)} extra anchors from the remaining pool")

        layer_anchors = [path for path, city in selected]
        anchors_by_layer[layer] = layer_anchors
        all_anchors.extend(layer_anchors)
        for path, city in selected:
            metadata[path] = {"city": city, "layer": layer, "region": CITY_TO_REGION.get(city, 'Unknown')}
        print(f"  Selected {len(layer_anchors)} anchors for layer {layer}")

    print(f"\nTotal anchors: {len(all_anchors)}")

    output = {"anchors": all_anchors, "by_layer": anchors_by_layer, "metadata": metadata}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Select global anchor cells, balanced by layer and region')
    parser.add_argument('--city-reps', type=Path, default=Path('../outputs/city_representatives.json'))
    parser.add_argument('--output', type=Path, default=Path('../outputs/global_anchors.json'))
    parser.add_argument('--anchors-per-layer', type=int, default=ANCHORS_PER_LAYER)
    args = parser.parse_args()
    select_anchors(args.city_reps, args.output, args.anchors_per_layer)


if __name__ == "__main__":
    main()
