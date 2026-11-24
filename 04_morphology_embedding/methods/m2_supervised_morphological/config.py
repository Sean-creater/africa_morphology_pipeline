"""M2: Supervised morphological metric learning — configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import OUTPUTS_DIR, FEATURES_CSV, TRAIN_CSV, VAL_CSV

# Output
OUTPUT_DIR = OUTPUTS_DIR / "m2_supervised_morphological"
MODEL_DIR  = OUTPUT_DIR / "models"
LOG_DIR    = OUTPUT_DIR / "logs"

# Model
INPUT_DIM   = 165   # inferred at runtime from features_analysis_set.csv
HIDDEN_DIMS = (512, 256)
OUTPUT_DIM  = 256
DROPOUT     = 0.2

# Training
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
WARMUP_EPOCHS = 5
MIN_LR        = 1e-6
PATIENCE      = 15
MIN_DELTA     = 1e-4

RANDOM_SEED = 42
DEVICE      = "cuda"
