"""
M5: Self-supervised distillation from frozen DINOv3 (M3 teacher).

Student MLP distills 1024-dim DINOv3 embeddings via 5-fold cross-validation.
Output: outputs/m5_selfsup_distillation/models/student_fold{k}.pth
"""
import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dataset import (
    MorphologyDataset, load_morphology_features, fit_scaler,
)
from shared.paths import FEATURES_CSV

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    M3_EMBEDDINGS, OUTPUT_DIR, MODEL_DIR, LOG_DIR,
    TEACHER_DIM, HIDDEN_DIMS, OUTPUT_DIM, DROPOUT, BATCH_NORM,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    WARMUP_EPOCHS, MIN_LR, PATIENCE, MIN_DELTA,
    LOSS_MSE_WEIGHT, LOSS_COS_WEIGHT,
    N_FOLDS, RANDOM_SEED,
)
from models import StudentMLP


class DistillationLoss(nn.Module):
    """L = w_mse * MSE + w_cos * (1 - cosine)"""

    def __init__(self, w_mse=1.0, w_cos=0.5):
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos

    def forward(self, student, teacher):
        loss_mse = nn.functional.mse_loss(student, teacher)
        cos_sim = nn.functional.cosine_similarity(student, teacher, dim=-1).mean()
        loss_cos = 1.0 - cos_sim
        return {
            "total": self.w_mse * loss_mse + self.w_cos * loss_cos,
            "mse": loss_mse.item(),
            "cos": loss_cos.item(),
            "cos_sim": cos_sim.item(),
        }


def lr_lambda(epoch, warmup_epochs, total_epochs, min_lr):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total, comps = 0.0, {"mse": 0.0, "cos": 0.0, "cos_sim": 0.0}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            feats = batch["features"].to(device)
            teacher_emb = batch["embedding"].to(device)
            loss_dict = criterion(model(feats), teacher_emb)
            if train:
                optimizer.zero_grad()
                loss_dict["total"].backward()
                optimizer.step()
            total += loss_dict["total"].item()
            for k in comps:
                comps[k] += loss_dict[k]
            n += 1
    return total / n, {k: v / n for k, v in comps.items()}


def main():
    print("=" * 70)
    print("M5: Self-supervised distillation from frozen DINOv3")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading teacher embeddings: {M3_EMBEDDINGS}")
    teacher_data = np.load(M3_EMBEDDINGS)
    teacher_embs = teacher_data["embeddings"]
    teacher_gids = list(teacher_data["grid_ids"])
    teacher_map  = {g: e for g, e in zip(teacher_gids, teacher_embs)}
    print(f"  Teacher: {len(teacher_gids)} grid cells, dim={teacher_embs.shape[1]}")

    features, grid_ids, feature_names, _ = load_morphology_features(FEATURES_CSV)

    idx = [i for i, g in enumerate(grid_ids) if g in teacher_map]
    grid_ids  = [grid_ids[i] for i in idx]
    features  = features[idx]
    teacher_aligned = np.array([teacher_map[g] for g in grid_ids])
    print(f"  Aligned: {len(grid_ids)} samples, features={features.shape[1]}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"\n{'='*60}\nFold {fold_idx+1}/{N_FOLDS}\n{'='*60}")

        scaler = fit_scaler(features[train_idx])
        X_train = scaler.transform(features[train_idx]).clip(-10, 10)
        X_val   = scaler.transform(features[val_idx]).clip(-10, 10)

        train_ds = MorphologyDataset(X_train, [grid_ids[i] for i in train_idx], teacher_aligned[train_idx])
        val_ds   = MorphologyDataset(X_val,   [grid_ids[i] for i in val_idx],   teacher_aligned[val_idx])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        input_dim = features.shape[1]
        model = StudentMLP(input_dim, HIDDEN_DIMS, DROPOUT, BATCH_NORM, OUTPUT_DIM).to(device)
        criterion = DistillationLoss(LOSS_MSE_WEIGHT, LOSS_COS_WEIGHT)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: lr_lambda(e, WARMUP_EPOCHS, EPOCHS, MIN_LR)
        )

        best_val_loss, best_state, no_improve = float("inf"), None, 0
        log = []

        for epoch in range(EPOCHS):
            train_loss, train_c = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_c     = run_epoch(model, val_loader,   criterion, None,      device, train=False)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            log.append({"epoch": epoch+1, "lr": lr,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "val_cos_sim": val_c["cos_sim"]})

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | LR {lr:.2e} | "
                      f"train {train_loss:.4f} | val {val_loss:.4f} | "
                      f"cos_sim {val_c['cos_sim']:.4f}")

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss, best_state, no_improve = val_loss, model.state_dict(), 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        fold_results.append({"fold": fold_idx + 1, "best_val_loss": best_val_loss,
                              "n_train": len(train_idx), "n_val": len(val_idx)})
        torch.save({
            "model_state_dict": best_state,
            "fold": fold_idx + 1,
            "input_dim": input_dim,
            "output_dim": OUTPUT_DIM,
            "scaler": scaler,
            "feature_names": feature_names,
            "best_val_loss": best_val_loss,
        }, MODEL_DIR / f"student_fold{fold_idx+1}.pth")
        pd.DataFrame(log).to_csv(LOG_DIR / f"training_log_fold{fold_idx+1}.csv", index=False)

    print("\n" + "=" * 60 + "\nCROSS-VALIDATION RESULTS\n" + "=" * 60)
    fold_losses = [r["best_val_loss"] for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_loss']:.4f}")
    print(f"  Mean ± Std: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    with open(LOG_DIR / "summary.json", "w") as f:
        json.dump({"fold_results": fold_results,
                   "mean_val_loss": float(np.mean(fold_losses)),
                   "std_val_loss": float(np.std(fold_losses))}, f, indent=2)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
