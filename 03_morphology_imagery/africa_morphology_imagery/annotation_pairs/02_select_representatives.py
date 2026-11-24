#!/usr/bin/env python3
"""Step 2: select per-city representative cells, layer-balanced.

For each city, selects up to `REPS_PER_CITY_TARGET` representative cells,
allocated across the 8 morphological layers in proportion to a fixed
target layer distribution (`TARGET_LAYER_PROPORTIONS`, derived from the
morphology-stratified sample's overall layer composition) rather than
equally across layers. Allocating equally would let a city's chronically
under-populated layers (e.g. layer 1, 6) pull representatives away from
better-populated ones every time, so target proportions are held constant
across cities instead.

Output: `city_representatives.json`,
`{city: {"representatives": [path, ...], "by_layer": {layer: [path, ...]}}}`.
"""

import argparse
import json
import random
from pathlib import Path

# Target share of representatives coming from each layer, matching each
# layer's overall share of the full stratified sample.
TARGET_LAYER_PROPORTIONS = {
    '0': 0.152, '1': 0.083, '2': 0.122, '3': 0.153,
    '4': 0.148, '5': 0.116, '6': 0.082, '7': 0.143,
}

REPS_PER_CITY_TARGET = 12
RANDOM_SEED = 42


def select_representatives(city_index_file: Path, output_file: Path,
                            reps_per_city_target: int = REPS_PER_CITY_TARGET,
                            random_seed: int = RANDOM_SEED) -> dict:
    random.seed(random_seed)

    with open(city_index_file, 'r') as f:
        city_index = json.load(f)

    print(f"Selecting representatives for {len(city_index)} cities...")

    city_representatives = {}
    total_reps = 0

    for city, layer_data in city_index.items():
        layer_counts = {layer: len(files) for layer, files in layer_data.items()}
        total_available = sum(layer_counts.values())
        if total_available == 0:
            print(f"  {city}: no samples available, skipping")
            continue

        n_reps = total_available if total_available < reps_per_city_target else reps_per_city_target

        available_layers = [l for l, c in layer_counts.items() if c > 0]
        n_layers = len(available_layers)
        if n_layers == 0:
            print(f"  {city}: no valid layers, skipping")
            continue

        # Allocate n_reps across available layers, proportional to each
        # layer's target share (renormalized over only the layers this
        # city actually has).
        layer_allocation = {}
        total_target_prop = sum(TARGET_LAYER_PROPORTIONS.get(l, 0.125) for l in available_layers)

        for layer in available_layers:
            layer_prop = TARGET_LAYER_PROPORTIONS.get(layer, 0.125)
            allocated = round((layer_prop / total_target_prop) * n_reps)
            if allocated == 0 and n_reps >= n_layers:
                allocated = 1
            allocated = min(allocated, layer_counts[layer])
            if allocated > 0:
                layer_allocation[layer] = allocated

        # Rounding can leave the total off by a few; add to layers with
        # spare capacity, or remove from layers with more than 1.
        total_allocated = sum(layer_allocation.values())
        diff = n_reps - total_allocated
        if diff > 0:
            for layer in sorted(available_layers, key=int):
                if diff == 0:
                    break
                if layer_allocation.get(layer, 0) < layer_counts[layer]:
                    layer_allocation[layer] = layer_allocation.get(layer, 0) + 1
                    diff -= 1
        elif diff < 0:
            for layer in sorted(available_layers, key=lambda l: layer_allocation.get(l, 0), reverse=True):
                if diff == 0:
                    break
                if layer_allocation.get(layer, 0) > 1:
                    layer_allocation[layer] -= 1
                    diff += 1

        selected_by_layer = {}
        all_selected = []
        for layer, n_select in layer_allocation.items():
            selected = random.sample(layer_data[layer], n_select)
            selected_by_layer[layer] = selected
            all_selected.extend(selected)

        city_representatives[city] = {"representatives": all_selected, "by_layer": selected_by_layer}
        total_reps += len(all_selected)
        print(f"  {city}: {len(all_selected)} reps, layers={ {l: len(f) for l, f in selected_by_layer.items()} }")

    print(f"\nTotal: {len(city_representatives)} cities, {total_reps} representatives")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(city_representatives, f, indent=2)
    print(f"Saved to: {output_file}")

    return city_representatives


def main():
    parser = argparse.ArgumentParser(description='Select layer-balanced representative cells per city')
    parser.add_argument('--city-index', type=Path, default=Path('../outputs/city_index.json'))
    parser.add_argument('--output', type=Path, default=Path('../outputs/city_representatives.json'))
    parser.add_argument('--reps-per-city', type=int, default=REPS_PER_CITY_TARGET)
    args = parser.parse_args()
    select_representatives(args.city_index, args.output, args.reps_per_city)


if __name__ == "__main__":
    main()
