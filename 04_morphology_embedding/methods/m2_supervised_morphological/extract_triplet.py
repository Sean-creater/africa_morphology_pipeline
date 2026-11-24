"""M2 (Triplet): Extract embeddings from triplet-trained model."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dataset import load_morphology_features
from shared.paths import FEATURES_CSV, FEATURES_VAL_CSV

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_DIR, OUTPUT_DIR, HIDDEN_DIMS, DROPOUT
from model import MorphologyMetricModel


def extract():
    emb_dir = OUTPUT_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt   = torch.load(MODEL_DIR / "best_model_triplet.pth", map_location="cpu")
    scaler = ckpt["scaler"]
    model  = MorphologyMetricModel(
        ckpt["input_dim"], HIDDEN_DIMS, ckpt["output_dim"], DROPOUT
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    feats_a, gids_a, _, _ = load_morphology_features(FEATURES_CSV)
    feats_v, gids_v, _, _ = load_morphology_features(FEATURES_VAL_CSV)
    existing = set(gids_a)
    new_idx  = [i for i, g in enumerate(gids_v) if g not in existing]
    features = np.vstack([feats_a, feats_v[new_idx]])
    grid_ids = gids_a + [gids_v[i] for i in new_idx]

    X = scaler.transform(features).clip(-10, 10).astype(np.float32)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.tensor(X[i:i + 512]).to(device)
            embeddings.append(model(batch).cpu().numpy())

    embeddings = np.vstack(embeddings)
    np.savez_compressed(
        emb_dir / "all_embeddings_triplet.npz",
        embeddings=embeddings,
        grid_ids=np.array(grid_ids),
    )
    print(f"Saved: {emb_dir / 'all_embeddings_triplet.npz'}  shape={embeddings.shape}")


if __name__ == "__main__":
    extract()
