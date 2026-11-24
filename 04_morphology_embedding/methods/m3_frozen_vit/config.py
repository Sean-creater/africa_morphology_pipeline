"""M3: Frozen DINOv3 — configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import (
    WEIGHTS_PATH, ANNOTATION_IMAGES_DIR, ANALYSIS_IMAGES_DIR,
    TRAIN_CSV, VAL_CSV, OUTPUTS_DIR,
)

# Backbone
BACKBONE_EMBED_DIM = 1024
PROJECTION_DIM = 256
IMG_SIZE = 224
DEVICE = "cuda"

# Output paths
OUTPUT_DIR    = OUTPUTS_DIR / "m3_frozen_vit"
BASELINE_DIR  = OUTPUT_DIR / "baseline"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

# Stage 1 hyperparameters
BASELINE_BATCH_SIZE = 32
BASELINE_LR = 1e-4
BASELINE_EPOCHS = 50

# Stage 2 hyperparameters
CLUSTER_SCALES = [10, 20, 30]
CLUSTERING_LOSS_WEIGHT = 1.0
DC_BATCH_SIZE = 32
DC_LR = 3e-5
DC_EPOCHS = 50
DC_WEIGHT_DECAY = 1e-4
DC_WARMUP_EPOCHS = 3
DC_MIN_LR = 1e-7
TARGET_UPDATE_INTERVAL = 1
ARI_THRESHOLD = 0.85
ARI_PATIENCE = 5
