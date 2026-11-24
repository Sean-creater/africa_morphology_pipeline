# Anchor-Based Comparison-Pair Generation

Generates image pairs for human urban-morphology similarity annotation,
using an anchor-based sampling strategy over the rendered cells from
`render.py`.

Rather than uniform sampling across all rendered grid cells, this uses a
two-tier structure: a smaller set of per-city representative cells, and a
subset of those forming a densely-connected "anchor" skeleton of the
morphological space. This keeps the pair count computationally tractable
while giving structural coverage across cities and morphology strata. See
the manuscript's Methods and Supplementary Information for the exact
sampling quotas and pair-type composition.

## Pipeline steps

1. `01_build_city_index.py` — groups rendered images by city and
   morphology stratum.
2. `02_select_representatives.py` — selects representative cells per city.
3. `03_select_anchors.py` — selects the global anchor set.
4. `04_generate_pairs.py` — generates the comparison-pair list.
5. `05_validate_pairs.py` — checks pair counts, balance, and duplicates.
6. `06_prepare_for_labeling.py` — packages pairs for an annotation
   interface.

## Usage

```bash
cd africa_morphology_imagery/annotation_pairs
bash run_pipeline.sh /path/to/png_integrated
```

## Requirements

Python 3.7+, standard library only.
