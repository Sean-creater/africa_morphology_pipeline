# Africa Morphology Embedding

Produces a compact, continent-scale embedding of African urban morphology,
capturing perceptual similarity between neighborhoods, for downstream use
in constructing a data-driven typology of African urban form.

## Approach

A vision model, trained on urban imagery with expert similarity judgments
as supervision, produces the reference embedding space (the "teacher"
stage). A lightweight model is then trained to reproduce that embedding
space from morphology indicators alone (the "student" stage), so the
published embedding requires only tabular features at inference and can be
computed for every grid cell on the continent.

This teacher-student design was selected via an ablation comparing six
approaches to producing the embedding — see [Ablation](#ablation).

## Installation

```bash
conda env create -f environment.yml
conda activate africa_morphology_embedding
# or: pip install -r requirements.txt
```

PyTorch with CUDA is required for the image-based stages.

## Data requirements

Everything needed to reproduce the ablation ranking table is bundled in
this repository: morphology feature CSVs (`data/features/`), the
expert-annotation dataset (`data/annotations/`), the evaluation set and
ranking table (`data/reference/`), a small illustrative image subset
(`data/example/`), the trained checkpoints and embeddings, and a vendored,
pruned copy of the vision backbone (`backbone/dinov3/`; pretrained weights
are not bundled — see `DATA_PATHS.md`).

Images are produced by `../03_morphology_imagery/`. The full-Africa
morphological indicator table used to apply the trained model beyond
this repository's evaluation set is produced by `../02_morphology_metrics/`
and is not bundled here; see `DATA_PATHS.md`.

### Full-Africa outputs

The morphological indicator table and embeddings for all African grid
cells (**Continent-scale morphology and embedding data**) are published
as release assets of this repository.

- `africa_morphology_indicators.parquet` — morphological indicator table, one row per grid cell.
- `africa_embeddings.npz` / `.parquet` — learned embedding table, one row per grid cell.

Both are joined on `GridID`, consistent with the hex-grid and
morphology-metrics modules, and cover the identical set of grid cells:
47 countries, 7,764 cities, 1,000,207 grid cells. See the manuscript's
Methods and Supplementary Information for construction and validation
details.

Countries covered: Algeria, Angola, Benin, Botswana, Burkina Faso, Burundi,
Cameroon, Cape Verde, Central African Republic, Côte dIvoire, Democratic
Republic of the Congo, Egypt, Equatorial Guinea, Eritrea, Ethiopia, Gabon,
Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar,
Malawi, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria,
Republic of Congo, Rwanda, Sao Tome and Principe, Senegal, Sierra Leone,
Somalia, South Africa, South Sudan, Sudan, Swaziland, Tanzania, Togo,
Tunisia, Uganda, Zambia, Zimbabwe.

The release is private, visible only to repository collaborators.

## Ablation

Six approaches to producing the embedding (`methods/`) were evaluated
against the same held-out expert pairwise-similarity judgments:
PCA on raw indicators, a supervised MLP on indicators, a frozen vision
backbone, a fine-tuned vision backbone (the teacher stage above), and two
distillation variants including the teacher-student design used
throughout this repository. Ranking table: `data/reference/final_ranking.csv`,
reproduced by `eval_all_methods.py`. See the manuscript's Methods and
Supplementary Information for each method's design and the evaluation
metrics used.

## Usage

```bash
# Train and extract the teacher (image-based) embedding
python methods/m4_finetuned_vit/train_triplet.py
python methods/m4_finetuned_vit/extract_triplet.py

# Train and extract the student (indicator-based) embedding
python methods/m6_triplet_distillation/train.py
python methods/m6_triplet_distillation/extract.py

# Reproduce the ablation ranking table from the bundled outputs/
python eval_all_methods.py
```

See `methods/` for the remaining alternatives evaluated in the ablation.

## Repository layout

```
africa_morphology_embedding/
├── README.md, DATA_PATHS.md, environment.yml, requirements.txt, pyproject.toml
├── eval_all_methods.py
├── methods/                         # the teacher-student approach and five alternatives
├── shared/                          # shared path/dataset utilities
├── backbone/dinov3/                 # vendored, pruned vision backbone (weights not bundled)
├── data/
│   ├── annotations/                  # expert similarity-judgment dataset
│   ├── reference/                    # evaluation set + ablation ranking table
│   ├── features/                     # morphology feature CSVs
│   └── example/                      # illustrative images
└── outputs/                          # final checkpoints + embeddings
```

## Citation

If you use this embedding model, please also cite the
morphology-extraction pipeline, `../02_morphology_metrics/`, which
produces the indicator inputs used throughout.

## License

No license file was found in the source project this repository was derived
from. Until a license is chosen, all rights are reserved by default; add a
`LICENSE` file before distributing this code publicly. The vendored
`backbone/dinov3/` dependency is the one exception: it carries Meta's own
DINOv3 license (see `backbone/dinov3/LICENSE.md`), and its pretrained
weights (not bundled) are subject to the same license's terms.
