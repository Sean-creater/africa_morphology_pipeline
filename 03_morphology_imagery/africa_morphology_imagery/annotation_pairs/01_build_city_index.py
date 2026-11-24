#!/usr/bin/env python3
"""Step 1: index rendered grid cells by city and morphological layer.

Scans the `png_integrated/{City}/*.png` images (the DINOv2/DINOv3-style
render, used as the source image set for annotation pairs) produced by
`render.py`, and groups them by city and by morphological "layer" (0-7),
which is the `Stratum_ID` value encoded as the last underscore-separated
token in each filename, e.g. `GHA_0063_000107_5.png` -> layer 5. This
layer id is the same 8-way morphology stratum computed in `sampling.py`.

Output: `city_index.json`, `{city: {layer: [file_path, ...]}}`.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def build_city_index(city_list_file: Path, png_base_dir: Path, output_file: Path) -> dict:
    with open(city_list_file, 'r') as f:
        cities = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(cities)} cities...")

    city_index = {}
    total_cells = 0

    for city in cities:
        city_dir = png_base_dir / city
        if not city_dir.exists():
            print(f"  Skipping {city}: directory not found")
            continue

        layer_groups = defaultdict(list)
        for png_file in city_dir.glob("*.png"):
            # Filename format: {Country}_{CityCode}_{GridID}_{Layer}.png
            parts = png_file.stem.split('_')
            if len(parts) >= 4:
                try:
                    layer = int(parts[-1])
                    if 0 <= layer <= 7:
                        layer_groups[str(layer)].append(str(png_file))
                except ValueError:
                    print(f"  Cannot parse layer from {png_file.name}")

        city_index[city] = dict(layer_groups)
        city_total = sum(len(files) for files in layer_groups.values())
        total_cells += city_total
        print(f"  {city}: {city_total} cells, layers={ {l: len(f) for l, f in layer_groups.items()} }")

    print(f"\nTotal: {len(city_index)} cities, {total_cells} cells")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(city_index, f, indent=2)
    print(f"Saved to: {output_file}")

    layer_totals = defaultdict(int)
    for city_data in city_index.values():
        for layer, files in city_data.items():
            layer_totals[layer] += len(files)
    print("\nCells per layer:")
    for layer in sorted(layer_totals.keys(), key=int):
        print(f"  Layer {layer}: {layer_totals[layer]}")

    return city_index


def main():
    parser = argparse.ArgumentParser(description='Index rendered grid cells by city and morphological layer')
    parser.add_argument('--city-list', type=Path, default=Path('../city_list.txt'))
    parser.add_argument('--png-dir', type=Path, required=True, help='png_integrated/ directory from render.py')
    parser.add_argument('--output', type=Path, default=Path('../outputs/city_index.json'))
    args = parser.parse_args()
    build_city_index(args.city_list, args.png_dir, args.output)


if __name__ == "__main__":
    main()
