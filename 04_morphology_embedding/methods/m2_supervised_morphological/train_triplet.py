"""
M2 (Triplet): Supervised morphological metric learning with triplet loss.

Hard triplets only (margin >= 0.25, max 15 per anchor).
Validation: Triplet Accuracy on all val-set triplets.
Output: outputs/m2_supervised_morphological/models/best_model_triplet.pth
"""
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.dataset import load_morphology_features, fit_scaler
from shared.paths import FEATURES_CSV, TRAIN_CSV, VAL_CSV

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, MODEL_DIR, LOG_DIR,
    HIDDEN_DIMS, OUTPUT_DIM,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    WARMUP_EPOCHS, MIN_LR,
    RANDOM_SEED, DEVICE,
)
from model import MorphologyMetricModel

DROPOUT        = 0.2
WEIGHT_DECAY   = 1e-4
MIN_MARGIN     = 0.25   # only triplets with rating gap >= 1 step
MAX_PER_ANCHOR = 15     # max triplets per anchor (prefer largest margin)
PATIENCE       = 10
MIN_DELTA      = 0.001


class TripletMorphologyDataset(Dataset):
    """
    Hard triplets from the annotation CSV (margin >= MIN_MARGIN,
    capped at MAX_PER_ANCHOR per anchor by descending margin).
    """

    def __init__(self, annotation_csv, feature_matrix: np.ndarray,
                 grid_id_to_idx: dict,
                 min_margin=MIN_MARGIN, max_per_anchor=MAX_PER_ANCHOR):
        df = pd.read_csv(annotation_csv)

        anchor_groups = defaultdict(list)
        for _, row in df.iterrows():
            ga, gb, r = row["Image_A_gridID"], row["Image_B_gridID"], float(row["Rating"])
            anchor_groups[ga].append((gb, r))
            anchor_groups[gb].append((ga, r))

        self.triplets = []   # (idx_a, idx_p, idx_n, margin)
        skipped = n_soft = 0
        for anchor_gid, peers in anchor_groups.items():
            if len(peers) < 2 or anchor_gid not in grid_id_to_idx:
                skipped += 1
                continue
            idx_a = grid_id_to_idx[anchor_gid]
            candidates = []
            for (g1, r1), (g2, r2) in combinations(peers, 2):
                if r1 == r2:
                    continue
                margin = abs(r1 - r2) / 4.0
                if margin < min_margin:
                    n_soft += 1
                    continue
                gid_p = g1 if r1 > r2 else g2
                gid_n = g2 if r1 > r2 else g1
                if gid_p in grid_id_to_idx and gid_n in grid_id_to_idx:
                    candidates.append((margin, grid_id_to_idx[gid_p], grid_id_to_idx[gid_n]))
            candidates.sort(key=lambda x: -x[0])
            for margin, idx_p, idx_n in candidates[:max_per_anchor]:
                self.triplets.append((idx_a, idx_p, idx_n, margin))

        self.features = torch.tensor(feature_matrix, dtype=torch.float32)
        print(f"TripletMorphologyDataset: {len(self.triplets):,} hard triplets  "
              f"(margin>={min_margin}, max {max_per_anchor}/anchor)  "
              f"[{skipped} anchors skipped, {n_soft:,} soft skipped]")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        ia, ip, in_, margin = self.triplets[idx]
        return (self.features[ia], self.features[ip], self.features[in_],
                torch.tensor(margin, dtype=torch.float32))


class TripletValDataset(Dataset):
    """
    All val triplets (no margin filter) for Triplet Accuracy computation.
    Triplet Accuracy = fraction where cos_sim(a,p) > cos_sim(a,n).
    """

    def __init__(self, annotation_csv, feature_matrix: np.ndarray,
                 grid_id_to_idx: dict):
        df = pd.read_csv(annotation_csv)

        anchor_groups = defaultdict(list)
        for _, row in df.iterrows():
            ga, gb, r = row["Image_A_gridID"], row["Image_B_gridID"], float(row["Rating"])
            anchor_groups[ga].append((gb, r))
            anchor_groups[gb].append((ga, r))

        self.triplets = []
        for anchor_gid, peers in anchor_groups.items():
            if len(peers) < 2 or anchor_gid not in grid_id_to_idx:
                continue
            idx_a = grid_id_to_idx[anchor_gid]
            for (g1, r1), (g2, r2) in combinations(peers, 2):
                if r1 == r2:
                    continue
                gid_p = g1 if r1 > r2 else g2
                gid_n = g2 if r1 > r2 else g1
                if gid_p in grid_id_to_idx and gid_n in grid_id_to_idx:
                    self.triplets.append((idx_a,
                                          grid_id_to_idx[gid_p],
                                          grid_id_to_idx[gid_n]))

        self.features = torch.tensor(feature_matrix, dtype=torch.float32)
        print(f"TripletValDataset: {len(self.triplets):,} triplets (all margins)")

    def __len__(self): return len(self.triplets)

    def __getitem__(self, idx):
        ia, ip, in_ = self.triplets[idx]
        return self.features[ia], self.features[ip], self.features[in_]


def triplet_cosine_loss(z_a, z_p, z_n, margin):
    d_ap = 1.0 - (z_a * z_p).sum(dim=-1)
    d_an = 1.0 - (z_a * z_n).sum(dim=-1)
    return F.relu(d_ap - d_an + margin).mean()


def lr_lambda(epoch, warmup_epochs, total_epochs, min_lr):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))


def train():
    print("=" * 70)
    print("M2 (Triplet): Supervised morphological metric learning")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df   = pd.read_csv(TRAIN_CSV)
    train_gids = set(train_df["Image_A_gridID"]) | set(train_df["Image_B_gridID"])
    features, grid_ids, feature_names, _ = load_morphology_features(
        FEATURES_CSV, train_grid_ids=list(train_gids)
    )
    scaler = fit_scaler(features)
    X = scaler.transform(features).clip(-10, 10).astype(np.float32)
    grid_id_to_idx = {g: i for i, g in enumerate(grid_ids)}

    train_ds = TripletMorphologyDataset(TRAIN_CSV, X, grid_id_to_idx)
    val_ds   = TripletValDataset(VAL_CSV, X, grid_id_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train triplets: {len(train_ds):,}  |  Val triplets: {len(val_ds):,}")

    model     = MorphologyMetricModel(features.shape[1], HIDDEN_DIMS, OUTPUT_DIM, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_lambda(e, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc, best_state, no_improve = 0.0, None, 0
    log = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for feat_a, feat_p, feat_n, margin in train_loader:
            feat_a = feat_a.to(device); feat_p = feat_p.to(device)
            feat_n = feat_n.to(device); margin = margin.to(device)
            optimizer.zero_grad()
            loss = triplet_cosine_loss(model(feat_a), model(feat_p),
                                       model(feat_n), margin)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feat_a, feat_p, feat_n in val_loader:
                za = model(feat_a.to(device))
                zp = model(feat_p.to(device))
                zn = model(feat_n.to(device))
                correct += ((za * zp).sum(-1) > (za * zn).sum(-1)).sum().item()
                total   += za.size(0)
        val_acc = correct / total

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        log.append({"epoch": epoch + 1, "lr": lr,
                    "train_triplet_loss": train_loss, "val_triplet_acc": val_acc})

        print(f"  Epoch {epoch+1:3d} | LR {lr:.2e} | "
              f"train_triplet {train_loss:.6f} | val_TripletAcc {val_acc*100:.2f}%")

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc, best_state, no_improve = val_acc, model.state_dict(), 0
            torch.save({
                "model_state_dict": best_state,
                "input_dim":        features.shape[1],
                "output_dim":       OUTPUT_DIM,
                "scaler":           scaler,
                "feature_names":    feature_names,
                "best_val_acc":     best_val_acc,
            }, MODEL_DIR / "best_model_triplet.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}  "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    pd.DataFrame(log).to_csv(LOG_DIR / "training_log_triplet.csv", index=False)
    print(f"\nBest val_TripletAcc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint: {MODEL_DIR / 'best_model_triplet.pth'}")


if __name__ == "__main__":
    train()
