"""
M4 (Triplet): Supervised metric learning with triplet loss over DINOv3 projection head.

Hard triplets only (margin >= 0.25, max 15 per anchor).
Validation: Triplet Accuracy on all val-set triplets.
Output: outputs/m4_finetuned_vit/baseline/best_projection_head_triplet.pth
"""
import sys
import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.paths import TRAIN_CSV, VAL_CSV, ANNOTATION_IMAGES_DIR

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    WEIGHTS_PATH, PROJECTION_DIM, BASELINE_BATCH_SIZE,
    BASELINE_EPOCHS, DEVICE, IMG_SIZE, BASELINE_DIR,
)
from model import MetricModel

LR             = 5e-5
WEIGHT_DECAY   = 1e-4
MIN_MARGIN     = 0.25
MAX_PER_ANCHOR = 15
PATIENCE       = 10
MIN_DELTA      = 0.001

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class TripletImageDataset(Dataset):
    """
    Builds hard triplets (margin >= MIN_MARGIN) from the annotation CSV.

    A triplet (anchor, positive, negative) is formed whenever the same
    grid cell appears in two pairs with rating gap >= MIN_MARGIN * 4 steps.
    Using only hard triplets reduces noise from near-tied annotation pairs.

    Each triplet stores the three image paths + adaptive margin.
    Images are loaded and augmented at __getitem__ time.
    """

    def __init__(self, annotation_csv, images_dir, img_size=224,
                 augment=True, min_margin=MIN_MARGIN, max_per_anchor=MAX_PER_ANCHOR):
        self.images_dir = Path(images_dir)
        self.augment    = augment
        self.resize     = transforms.Resize((img_size, img_size))
        self.jitter     = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
        )

        df = pd.read_csv(annotation_csv)

        gid_to_file = {}
        for _, row in df.iterrows():
            gid_to_file[row["Image_A_gridID"]] = str(row["Image_A"]).replace(".jpg", ".png")
            gid_to_file[row["Image_B_gridID"]] = str(row["Image_B"]).replace(".jpg", ".png")

        anchor_groups = defaultdict(list)
        for _, row in df.iterrows():
            anchor_groups[row["Image_A_gridID"]].append(
                (row["Image_B_gridID"], float(row["Rating"]))
            )
            anchor_groups[row["Image_B_gridID"]].append(
                (row["Image_A_gridID"], float(row["Rating"]))
            )

        self.triplets = []
        missing = 0
        n_soft_skipped = 0
        for anchor_gid, peers in anchor_groups.items():
            if len(peers) < 2:
                continue
            path_a = self.images_dir / gid_to_file.get(anchor_gid, "")
            if not path_a.exists():
                missing += 1
                continue
            candidates = []
            for (g1, r1), (g2, r2) in combinations(peers, 2):
                if r1 == r2:
                    continue
                margin = abs(r1 - r2) / 4.0
                if margin < min_margin:
                    n_soft_skipped += 1
                    continue
                gid_p = g1 if r1 > r2 else g2
                gid_n = g2 if r1 > r2 else g1
                path_p = self.images_dir / gid_to_file.get(gid_p, "")
                path_n = self.images_dir / gid_to_file.get(gid_n, "")
                if path_p.exists() and path_n.exists():
                    candidates.append((margin, path_a, path_p, path_n))
            candidates.sort(key=lambda x: -x[0])
            for margin, pa, pp, pn in candidates[:max_per_anchor]:
                self.triplets.append((pa, pp, pn, margin))

        print(f"TripletImageDataset: {len(self.triplets):,} hard triplets  "
              f"(margin>={min_margin}, max {max_per_anchor}/anchor)  "
              f"[{missing} anchors missing image, {n_soft_skipped:,} soft skipped]")

    def __len__(self):
        return len(self.triplets)

    def _aug(self, img, seed):
        random.seed(seed)
        if random.random() > 0.25:
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img = TF.rotate(img, angle)
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        if random.random() > 0.3:
            img = self.jitter(img)
        if random.random() > 0.5:
            img = TF.gaussian_blur(img, kernel_size=3, sigma=(0.1, 1.0))
        return img

    def __getitem__(self, idx):
        path_a, path_p, path_n, margin = self.triplets[idx]
        imgs = [self.resize(Image.open(p).convert("RGB"))
                for p in (path_a, path_p, path_n)]
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            imgs = [self._aug(img, seed) for img in imgs]
        tensors = [NORMALIZE(TF.to_tensor(img)) for img in imgs]
        return (*tensors, torch.tensor(margin, dtype=torch.float32))


class TripletValDataset(Dataset):
    """
    Val triplets built from the val annotation CSV — ALL triplets (no margin
    filter) so that the metric captures fine-grained ordering ability, not
    just easy large-gap cases that the frozen backbone already handles.
    Triplet Accuracy = fraction of triplets where cos_sim(a,p) > cos_sim(a,n).
    No augmentation at val time.
    """

    def __init__(self, annotation_csv, images_dir, img_size=224,
                 min_margin=0.0):
        self.images_dir = Path(images_dir)
        self.transform  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            NORMALIZE,
        ])

        df = pd.read_csv(annotation_csv)

        gid_to_file = {}
        for _, row in df.iterrows():
            gid_to_file[row["Image_A_gridID"]] = str(row["Image_A"]).replace(".jpg", ".png")
            gid_to_file[row["Image_B_gridID"]] = str(row["Image_B"]).replace(".jpg", ".png")

        anchor_groups = defaultdict(list)
        for _, row in df.iterrows():
            anchor_groups[row["Image_A_gridID"]].append(
                (row["Image_B_gridID"], float(row["Rating"]))
            )
            anchor_groups[row["Image_B_gridID"]].append(
                (row["Image_A_gridID"], float(row["Rating"]))
            )

        self.triplets = []
        for anchor_gid, peers in anchor_groups.items():
            if len(peers) < 2:
                continue
            path_a = self.images_dir / gid_to_file.get(anchor_gid, "")
            if not path_a.exists():
                continue
            for (g1, r1), (g2, r2) in combinations(peers, 2):
                if r1 == r2:
                    continue
                margin = abs(r1 - r2) / 4.0
                if margin < min_margin:
                    continue
                gid_p = g1 if r1 > r2 else g2
                gid_n = g2 if r1 > r2 else g1
                path_p = self.images_dir / gid_to_file.get(gid_p, "")
                path_n = self.images_dir / gid_to_file.get(gid_n, "")
                if path_p.exists() and path_n.exists():
                    self.triplets.append((path_a, path_p, path_n))

        print(f"TripletValDataset: {len(self.triplets):,} hard triplets (margin>={min_margin})")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        path_a, path_p, path_n = self.triplets[idx]
        return tuple(self.transform(Image.open(p).convert("RGB"))
                     for p in (path_a, path_p, path_n))


def triplet_cosine_loss(z_a, z_p, z_n, margin):
    d_ap = 1.0 - (z_a * z_p).sum(dim=-1)
    d_an = 1.0 - (z_a * z_n).sum(dim=-1)
    return F.relu(d_ap - d_an + margin).mean()


def train():
    print("=" * 70)
    print("M4 Stage 1 (Triplet): Supervised Metric Learning  [hard triplets]")
    print("=" * 70)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TripletImageDataset(TRAIN_CSV, ANNOTATION_IMAGES_DIR, IMG_SIZE,
                                   augment=True, min_margin=MIN_MARGIN)
    val_ds   = TripletValDataset(VAL_CSV, ANNOTATION_IMAGES_DIR, IMG_SIZE,
                                 min_margin=0.0)

    train_loader = DataLoader(train_ds, batch_size=BASELINE_BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BASELINE_BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train triplets: {len(train_ds):,}  |  Val triplets: {len(val_ds):,}")

    model     = MetricModel(str(WEIGHTS_PATH), projection_dim=PROJECTION_DIM).to(device)
    optimizer = torch.optim.AdamW(model.get_trainable_params(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=BASELINE_EPOCHS, eta_min=LR * 0.05
    )

    best_val_acc = 0.0
    no_improve   = 0
    log          = []

    for epoch in range(BASELINE_EPOCHS):
        model.head.train()
        train_loss = 0.0
        for img_a, img_p, img_n, margin in train_loader:
            img_a  = img_a.to(device)
            img_p  = img_p.to(device)
            img_n  = img_n.to(device)
            margin = margin.to(device)
            optimizer.zero_grad()
            loss = triplet_cosine_loss(model(img_a), model(img_p),
                                       model(img_n), margin)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for img_a, img_p, img_n in val_loader:
                za = model(img_a.to(device))
                zp = model(img_p.to(device))
                zn = model(img_n.to(device))
                sim_ap = (za * zp).sum(dim=-1)
                sim_an = (za * zn).sum(dim=-1)
                correct += (sim_ap > sim_an).sum().item()
                total   += za.size(0)
        val_acc = correct / total

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d} | LR {lr:.2e} | "
              f"train_triplet {train_loss:.6f} | val_TripletAcc {val_acc*100:.2f}%")
        log.append({"epoch": epoch + 1, "lr": lr,
                    "train_triplet_loss": train_loss, "val_triplet_acc": val_acc})

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.head.state_dict(),
                       BASELINE_DIR / "best_projection_head_triplet.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}  "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    pd.DataFrame(log).to_csv(BASELINE_DIR / "training_log_triplet.csv", index=False)
    print(f"\nBest val_TripletAcc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint: {BASELINE_DIR / 'best_projection_head_triplet.pth'}")


if __name__ == "__main__":
    train()
