"""M6 (triplet_distillation): distillation from the triplet-trained M4 teacher — configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import OUTPUTS_DIR, FEATURES_CSV

# Teacher embeddings: M4 Triplet-trained projection head
M4_EMBEDDINGS = OUTPUTS_DIR / "m4_finetuned_vit" / "embeddings" / "all_embeddings_triplet.npz"

# Output directories
OUTPUT_DIR = OUTPUTS_DIR / "m6_triplet_distillation"
MODEL_DIR  = OUTPUT_DIR / "models"
LOG_DIR    = OUTPUT_DIR / "logs"

# Student MLP: 165 → 256
HIDDEN_DIMS = [512, 512]
OUTPUT_DIM  = 256
DROPOUT     = 0.2
BATCH_NORM  = True

BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
WARMUP_EPOCHS = 5
MIN_LR        = 1e-6
PATIENCE      = 15
MIN_DELTA     = 1e-4

LOSS_PAIR_WEIGHT = 1.0

N_FOLDS     = 5
RANDOM_SEED = 42
