#!/usr/bin/env python3
"""Step 6: package comparison pairs for a human annotation interface.

For each pair in `comparison_pairs.csv`, creates a folder named
`pair_{id}_{city1}-L{layer1}_vs_{city2}-L{layer2}/` containing the two
source images, then groups all pair folders into fixed-size batches (200
pairs/batch) as JSON files ready for an annotation tool to iterate over,
plus a master index and an annotator-facing README describing the 1-5
similarity scale.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict

BATCH_SIZE = 200


def prepare_labeling_data(pairs_file: Path, output_dir: Path, batch_size: int = BATCH_SIZE) -> None:
    pairs_dir = output_dir / 'pairs'
    batches_dir = output_dir / 'batches'
    pairs_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    copied_count = missing_count = 0
    pair_folders = []

    for pair in pairs:
        pair_id, city1, city2 = pair['pair_id'], pair['city1'], pair['city2']
        layer1, layer2 = pair['layer1'], pair['layer2']
        img1_path, img2_path = pair['img1_path'], pair['img2_path']

        folder_name = f"{pair_id}_{city1}-L{layer1}_vs_{city2}-L{layer2}"
        pair_folder = pairs_dir / folder_name
        pair_folder.mkdir(exist_ok=True)

        src1, src2 = Path(img1_path), Path(img2_path)
        if not src1.exists():
            print(f"  Missing: {img1_path}")
            missing_count += 1
            continue
        if not src2.exists():
            print(f"  Missing: {img2_path}")
            missing_count += 1
            continue

        filename1, filename2 = src1.name, src2.name
        dst1 = pair_folder / f"img1_{city1}_{filename1}"
        dst2 = pair_folder / f"img2_{city2}_{filename2}"

        if not dst1.exists():
            shutil.copy2(src1, dst1)
            copied_count += 1
        if not dst2.exists():
            shutil.copy2(src2, dst2)
            copied_count += 1

        pair_folders.append({
            'pair_id': pair_id,
            'folder_name': folder_name,
            'folder_path': f"pairs/{folder_name}",
            'img1_relative': f"pairs/{folder_name}/img1_{city1}_{filename1}",
            'img2_relative': f"pairs/{folder_name}/img2_{city2}_{filename2}",
            'metadata': {
                'pair_type': pair['pair_type'], 'layer1': layer1, 'layer2': layer2,
                'city1': city1, 'city2': city2, 'is_anchor_pair': pair['is_anchor_pair'],
            },
        })

    print(f"Created {len(pair_folders)} pair folders, copied {copied_count} images"
          + (f", {missing_count} missing" if missing_count else ""))

    # Batches.
    n_batches = (len(pair_folders) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch_pair_folders = pair_folders[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_data = {
            "batch_id": batch_idx + 1,
            "total_pairs": len(batch_pair_folders),
            "pairs": [
                {
                    "id": pf['pair_id'], "folder_name": pf['folder_name'], "folder_path": pf['folder_path'],
                    "image1": pf['img1_relative'], "image2": pf['img2_relative'], "metadata": pf['metadata'],
                    "similarity_score": None, "notes": "",
                }
                for pf in batch_pair_folders
            ],
        }
        batch_file = batches_dir / f"batch_{batch_idx + 1:03d}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        print(f"  Batch {batch_idx + 1}/{n_batches}: {len(batch_pair_folders)} pairs -> {batch_file}")

    # Master index.
    all_cities = set()
    type_counts = defaultdict(int)
    for pf in pair_folders:
        all_cities.add(pf['metadata']['city1'])
        all_cities.add(pf['metadata']['city2'])
        type_counts[pf['metadata']['pair_type']] += 1

    master_index = {
        "project": "Urban Morphology Similarity Annotation",
        "total_pairs": len(pair_folders),
        "total_batches": n_batches,
        "batch_size": batch_size,
        "batches": [
            {"batch_id": i + 1, "file": f"batches/batch_{i + 1:03d}.json",
             "pairs_count": batch_size if i < n_batches - 1 else len(pair_folders) - i * batch_size}
            for i in range(n_batches)
        ],
        "statistics": {
            "total_pair_folders": len(pair_folders), "total_cities": len(all_cities),
            "pair_types": dict(type_counts),
        },
    }
    with open(output_dir / 'master_index.json', 'w') as f:
        json.dump(master_index, f, indent=2)

    # Annotator README.
    readme_content = f"""# Urban Morphology Similarity Annotation

## Overview

This dataset contains {len(pair_folders)} image pairs from {len(all_cities)} African cities.
The task is to rate the morphological similarity between pairs of urban grid cells.

## Structure

- `pairs/`: {len(pair_folders)} folders, one per comparison pair, each with exactly 2 images
- `batches/`: {n_batches} batch files, each with ~{batch_size} pairs
- `master_index.json`: overview of all batches

## Folder Naming Convention

Example: `pair_00001_Cairo-L3_vs_Lagos-L5/` compares Cairo layer 3 with Lagos layer 5.
Each folder contains `img1_{{city}}_{{filename}}.png` and `img2_{{city}}_{{filename}}.png`.

## Similarity Scale (1-5)

- **1**: Completely different - numerous highly inconsistent characteristics or extreme differences in key dimensions
- **2**: Somewhat dissimilar - many characteristics differ substantially, yet still exhibit some overall similarity
- **3**: Moderately similar - several moderately different features, or one pronounced difference while most other characteristics remain similar
- **4**: Quite similar - same overall character with variations, but no single dimension shows extreme differences
- **5**: Very similar - nearly identical patterns with highly consistent characteristics; minor detail differences are acceptable

## What to Consider

1. Building density: how closely packed are the buildings?
2. Street network: regular grid vs. organic/irregular?
3. Building types: size, shape, arrangement
4. Land use mix: residential, commercial, industrial, mixed?
5. Urban form: block structure, plot sizes

## Layer Information (L0-L7)

Images are classified into 8 morphological layers (0-7), the same stratum IDs computed in
`sampling.py`, based on building squareness, building-street alignment, and street
orientation diversity. The layer number is the filename's final digit (e.g. `_3.png` = layer 3).

Fill in each pair's `similarity_score` field (in the batch JSON) with a value from 1-5.
"""
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print(f"\nDone. {len(pair_folders)} pair folders, {n_batches} batches, output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Package comparison pairs for a human annotation interface')
    parser.add_argument('--pairs', type=Path, default=Path('../outputs/comparison_pairs.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('../labeling_data'))
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    prepare_labeling_data(args.pairs, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
