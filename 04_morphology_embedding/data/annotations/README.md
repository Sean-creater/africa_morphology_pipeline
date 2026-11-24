# Annotation Data

Expert pairwise-similarity ratings used to train the teacher model and the
ablation alternatives, and their train/validation split.

| File | Description |
|------|-------------|
| `pairs_all_rounds.csv` | Consolidated annotation dataset (inter-annotator filtered, averaged) |
| `pairs_all_rounds_detailed.csv` | Same, with per-expert ratings retained |
| `train.csv` / `val.csv` | Fixed 90/10 split, used identically throughout |

## Format

`Image_A_gridID, Image_B_gridID, Image_A, Image_B, Rating` — a rated pair
of grid cells, identified by grid ID and image filename. Images are the
satellite + building/road overlay render produced by
`../../../03_morphology_imagery/`, the same style used as model input
for the teacher model.
Rating is the mean expert similarity score (1.0 = completely dissimilar,
5.0 = very similar).

See the manuscript's Methods and Supplementary Information for the
inter-annotator agreement methodology.
