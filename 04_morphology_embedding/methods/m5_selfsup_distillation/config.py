"""M5: Self-supervised distillation — configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import OUTPUTS_DIR, FEATURES_CSV

# Teacher embeddings come from M3 (frozen DINOv3)
M3_EMBEDDINGS = OUTPUTS_DIR / "m3_frozen_vit" / "embeddings" / "all_embeddings.npz"

# Output
OUTPUT_DIR = OUTPUTS_DIR / "m5_selfsup_distillation"
MODEL_DIR  = OUTPUT_DIR / "models"
LOG_DIR    = OUTPUT_DIR / "logs"

# Model architecture (student MLP)
TEACHER_DIM = 1024   # M3 outputs 1024-dim (raw backbone, L2-normalized)
HIDDEN_DIMS = [512, 512]
OUTPUT_DIM  = 1024   # match teacher dimension
DROPOUT     = 0.2
BATCH_NORM  = True

# Training
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
WARMUP_EPOCHS = 5
MIN_LR        = 1e-6
PATIENCE      = 15
MIN_DELTA     = 1e-4

# Loss weights: L = w_mse*MSE + w_cos*(1-cos)
LOSS_MSE_WEIGHT = 1.0
LOSS_COS_WEIGHT = 0.5

# Cross-validation
N_FOLDS     = 5
RANDOM_SEED = 42
