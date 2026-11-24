# Example / Illustrative Images

A small, reproducible slice of the imagery used in the study — not the full
image sets, which are too large to check into a code repository.

- `val_pair_images/` — a subset of composite validation-pair images,
  letting the image-extraction code path (`extract.py` / `extract_triplet.py`)
  be exercised end-to-end on real data. `eval_all_methods.py` reproduces
  `final_ranking.csv` from the bundled `outputs/` embeddings directly and
  does not require these images.
- `annotation_sample/` — a handful of representative images illustrating
  the composite satellite + building/road-overlay format used throughout
  the study. Illustrative only, not a usable training or evaluation subset.
