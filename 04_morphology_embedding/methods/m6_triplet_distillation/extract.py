"""
M6-Triplet: Extract student embeddings (best fold model).

Embeds all analysis-set and validation-set grid cells using the best
cross-validation fold (lowest val_loss). Scaler is taken from the saved
checkpoint (fit on the training split only — no leakage).

Output: outputs/m6_triplet_distillation/embeddings/all_embeddings.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dataset import load_morphology_features
from shared.paths import FEATURES_CSV, FEATURES_VAL_CSV

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_DIR, OUTPUT_DIR
from models import StudentMLP


def load_best_fold(model_dir):
    best = None
    for p in sorted(model_dir.glob("student_fold*.pth")):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if best is None or ckpt["best_val_loss"] < best[1]:
            best = (p, ckpt["best_val_loss"], ckpt)
    print(f"Best fold: {best[0].name}  (val_loss={best[1]:.4f})")
    return best[2]


def extract():
    print("=" * 70)
    print("M6-Triplet: Extracting student embeddings")
    print("=" * 70)

    emb_dir = OUTPUT_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt   = load_best_fold(MODEL_DIR)
    scaler = ckpt["scaler"]
    model  = StudentMLP(
        ckpt["input_dim"], ckpt["hidden_dims"],
        ckpt["dropout"], ckpt["batch_norm"], ckpt["output_dim"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    feats_a, gids_a, _, _ = load_morphology_features(FEATURES_CSV)
    feats_v, gids_v, _, _ = load_morphology_features(FEATURES_VAL_CSV)

    existing = set(gids_a)
    mask     = [i for i, g in enumerate(gids_v) if g not in existing]
    feats_v  = feats_v[mask]
    gids_v   = [gids_v[i] for i in mask]
    print(f"  analysis_set: {len(gids_a)}  val_set (new): {len(gids_v)}")

    features = np.vstack([feats_a, feats_v])
    grid_ids = gids_a + gids_v
    X = scaler.transform(features).clip(-10, 10).astype(np.float32)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.tensor(X[i:i + 512]).to(device)
            embeddings.append(model(batch).cpu().numpy())

    embeddings = np.vstack(embeddings)
    np.savez_compressed(
        emb_dir / "all_embeddings.npz",
        embeddings=embeddings,
        grid_ids=np.array(grid_ids),
    )
    print(f"\nTotal: {len(embeddings)} embeddings {embeddings.shape}")
    print(f"Saved to: {emb_dir}")


if __name__ == "__main__":
    extract()
