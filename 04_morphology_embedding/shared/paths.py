"""Shared path constants for all methods."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

ANNOTATIONS_DIR       = ROOT / "data" / "annotations"
TRAIN_CSV             = ANNOTATIONS_DIR / "train.csv"
VAL_CSV               = ANNOTATIONS_DIR / "val.csv"
ANNOTATION_IMAGES_DIR = ROOT / "data" / "images" / "annotation_set"
ANALYSIS_IMAGES_DIR   = ROOT / "data" / "images" / "analysis_set"

# Bundled validation-pair images (small illustrative/reproducible subset —
# see data/example/README.md for exactly which grid cells are covered).
VAL_PAIRS_IMAGES_DIR  = ROOT / "data" / "example" / "val_pair_images"

FEATURES_CSV          = ROOT / "data" / "features" / "features_analysis_set.csv"
FEATURES_VAL_CSV      = ROOT / "data" / "features" / "features_val_set.csv"

BACKBONE_DIR  = ROOT / "backbone" / "dinov3"
WEIGHTS_PATH  = ROOT / "backbone" / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

EVAL_DIR          = ROOT / "data" / "reference"
EVAL_PAIRS_CSV    = EVAL_DIR / "eval_pairs.csv"      # 535 high-consensus pairs (round2 + anchor)
EVAL_TRIPLETS_CSV = EVAL_DIR / "eval_triplets.csv"   # 546 triplets derived from anchor pairs

OUTPUTS_DIR = ROOT / "outputs"

# 02_morphology_metrics/ output: cleaned, per-city 250m-hex morphology
# CSVs. Not bundled here (external, large). Default assumes this
# pipeline's standard directory layout. Override with
# MORPHOLOGY_CLEANED_CSV_ROOT if your checkout lives elsewhere.
CLEANED_CSV_ROOT = Path(os.environ.get(
    "MORPHOLOGY_CLEANED_CSV_ROOT",
    str(ROOT / "../02_morphology_metrics/outputs/Cleaned"),
))
