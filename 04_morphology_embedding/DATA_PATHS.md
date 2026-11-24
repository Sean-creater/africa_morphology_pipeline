# Data Paths Reference

All data required to reproduce the ablation comparison is bundled
inside this directory, with one external dependency: the full-Africa
morphological indicator table, produced by `../02_morphology_metrics/`.
It is not bundled here and is only needed to apply the trained
embedding model beyond this repository's evaluation set — it is not
required to reproduce the ablation ranking table. Point the
`MORPHOLOGY_CLEANED_CSV_ROOT` environment variable at your own checkout of
that pipeline's output if it differs from the documented default.

## Directory Layout

```
africa_morphology_embedding/
├── data/
│   ├── annotations/    # expert similarity-judgment dataset, train/val split
│   ├── reference/      # evaluation set and ablation ranking table
│   ├── features/       # morphology feature CSVs
│   └── example/        # illustrative images
├── backbone/dinov3/     # vendored, pruned vision backbone (weights not bundled)
├── methods/             # the teacher-student approach and five alternatives
├── shared/               # shared dataset/path utilities
├── eval_all_methods.py   # ablation evaluation script
└── outputs/              # final checkpoints/embeddings
```

## Python Environment

See `environment.yml` / `requirements.txt`. Training and embedding
extraction require PyTorch + CUDA for the image-based stages; the
indicator-only stages do not.
