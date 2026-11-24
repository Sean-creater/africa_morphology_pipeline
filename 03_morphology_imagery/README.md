# Africa Morphology Imagery

Constructs an image dataset for training and human-annotating an urban
morphology embedding model, from African cities' 250 m hexagonal grid cells.
For a stratified sample of grid cells per city, this downloads Sentinel-2
satellite imagery and renders visualizations combining building/road
footprints with the satellite background, then builds an anchor-based
pairwise comparison dataset used to collect human similarity judgments.

## Upstream dependency

This pipeline reads per-city 250 m hexagonal grid cells with morphology
metrics already computed, produced by `../02_morphology_metrics/`. It
does not compute morphology metrics itself; prepare its output first.

## Downstream use

The rendered images and comparison-pair dataset feed `../04_morphology_embedding/`'s
embedding model training and comparison study. This directory's pipeline
does not depend on it and stops at producing the image dataset and pair
list.

## Pipeline stages

1. **Candidate cell selection** (`sampling.py`) — a manageable per-city
   subset of grid cells.
2. **Morphology-stratified sampling** (`sampling.py`) — resamples the
   candidate pool by morphology so the dataset covers the range of
   building/street form.
3. **Satellite imagery download** (`satellite_download.py`) — Sentinel-2
   imagery via Google Earth Engine, per sampled cell.
4. **Rendering** (`render.py`) — building/road outline and satellite-backed
   visualization styles per cell.
5. **Annotation-pair construction** (`annotation_pairs/`) — an anchor-based
   pairwise comparison dataset for human similarity annotation.

See the manuscript's Methods and Supplementary Information for the
sampling and rendering methodology, including the stratification
dimensions, quotas, and rendering design choices.

## Installation

```bash
conda env create -f environment.yml
conda activate africa_morphology_imagery
# or: pip install -r requirements.txt
pip install -e .
```

Google Earth Engine access requires a one-time interactive authentication
(`python -c "import ee; ee.Authenticate(auth_mode='notebook')"`) before
`satellite_download.py` or `render.py` can run.

## Usage

```bash
python -m africa_morphology_imagery.sampling candidates \
    --cities-csv data/reference/SampleCityPath.csv \
    --grid-glob-template "/path/to/{country}/{city}/Output/{city}_grid_250m_hex_enriched_metrics_v15.shp" \
    --output-shp-dir /path/to/output/shp

python -m africa_morphology_imagery.sampling stratify \
    --shp-dir /path/to/output/shp \
    --city-path-csv data/reference/SampleCityPath.csv \
    --output-csv /path/to/output/SampleGridCell.csv \
    --output-stats-csv /path/to/output/SamplingStatistics.csv

python -m africa_morphology_imagery.render \
    --city-path-csv data/reference/SampleCityPath.csv \
    --grid-cells-dir /path/to/output/shp \
    --metadata-csv /path/to/output/SampleGridCell.csv \
    --output-dir /path/to/output/images

cd africa_morphology_imagery/annotation_pairs
bash run_pipeline.sh /path/to/output/images/png_integrated
```

## Output schema

- `SampleGridCell.csv` — one row per sampled cell, with `GridID`, morphology
  stratum, and raw morphology values.
- Rendered image directories under `{output-dir}/` per style.
- `comparison_pairs.csv` — the annotation-pair list.

See `data/example/` for real example images and one annotation pair, and
`annotation_pairs/README.md` for the pair-type breakdown.

## Repository layout

```
africa_morphology_imagery/
  README.md
  environment.yml / requirements.txt
  pyproject.toml
  africa_morphology_imagery/
    sampling.py              # candidate selection + stratified sampling
    satellite_download.py    # Sentinel-2 download via Earth Engine
    render.py                # rendered image styles
    annotation_pairs/        # anchor-based pair-generation pipeline
  data/
    reference/                # small durable config
    example/                  # real example images and one annotation pair
```

## License

No LICENSE file is included. The source project this was extracted from
does not carry one either, so none is asserted here.
