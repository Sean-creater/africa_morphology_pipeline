"""
M1: Raw morphological features baseline.

No learning. Pipeline:
  1. Load 165-dim features, fill NaNs
  2. StandardScaler normalization
  3. PCA → 64-dim (for visualization and reduced-dim analysis)
  4. Save full 165-dim and PCA-reduced embeddings

The "embedding" for M1 is simply the StandardScaler-normalized 165-dim feature
vector (or its PCA projection), with no neural network involved.

Output: outputs/m1_raw_features/embeddings/
  all_embeddings.npz      — normalized 165-dim features [N, 165]
  all_embeddings_pca.npz  — PCA-reduced [N, 64]
"""
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dataset import load_morphology_features, fit_scaler
from shared.paths import FEATURES_CSV, FEATURES_VAL_CSV, OUTPUTS_DIR

OUTPUT_DIR = OUTPUTS_DIR / "m1_raw_features"
EMB_DIR    = OUTPUT_DIR / "embeddings"
PCA_COMPONENTS = 64


def run():
    print("=" * 70)
    print("M1: Raw morphological features")
    print("=" * 70)

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    features, grid_ids, feature_names, _ = load_morphology_features(FEATURES_CSV)
    print(f"Loaded analysis_set: {len(grid_ids)} grid cells, {features.shape[1]} features")

    # Scaler and PCA fit on analysis_set only (no leakage)
    scaler = fit_scaler(features)

    # Append val_set grid cells (deduplicated)
    feats_v, gids_v, _, _ = load_morphology_features(FEATURES_VAL_CSV)
    existing = set(grid_ids)
    new_idx  = [i for i, g in enumerate(gids_v) if g not in existing]
    features = np.vstack([features, feats_v[new_idx]])
    grid_ids = grid_ids + [gids_v[i] for i in new_idx]
    print(f"After adding val_set: {len(grid_ids)} grid cells total")

    X = scaler.transform(features).clip(-10, 10).astype(np.float32)
    print(f"Normalized: mean≈{X.mean():.4f}  std≈{X.std():.4f}")

    np.savez_compressed(
        EMB_DIR / "all_embeddings.npz",
        embeddings=X,
        grid_ids=np.array(grid_ids),
    )
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    # PCA fit on analysis_set only, then transform all
    n_components = min(PCA_COMPONENTS, features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X[:len(grid_ids) - len(new_idx)]).astype(np.float32)
    X_pca_val = pca.transform(X[len(grid_ids) - len(new_idx):]).astype(np.float32)
    X_pca = np.vstack([X_pca, X_pca_val]).astype(np.float32)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA {n_components}-dim: {explained*100:.1f}% variance explained")

    np.savez_compressed(
        EMB_DIR / "all_embeddings_pca.npz",
        embeddings=X_pca,
        grid_ids=np.array(grid_ids),
    )
    joblib.dump(pca, OUTPUT_DIR / "pca.pkl")

    print(f"\nSaved to: {EMB_DIR}")
    print(f"  all_embeddings.npz      — {X.shape}")
    print(f"  all_embeddings_pca.npz  — {X_pca.shape}")


if __name__ == "__main__":
    run()
