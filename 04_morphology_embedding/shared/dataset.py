"""Shared dataset classes used across methods."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.preprocessing import StandardScaler


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_ZERO_FILL_PREFIXES = ("LC_", "H_", "V_", "Mean_")


class PairwiseDataset(Dataset):
    """Annotated image pairs with joint augmentation. Rating normalized [1,5] -> [0,1]."""

    def __init__(self, annotation_csv, images_dir, img_size=224, augment=True):
        self.images_dir = Path(images_dir)
        self.augment = augment

        df = pd.read_csv(annotation_csv)
        self.pairs = []
        for _, row in df.iterrows():
            fname_a = str(row["Image_A"]).replace(".jpg", ".png")
            fname_b = str(row["Image_B"]).replace(".jpg", ".png")
            path_a = self.images_dir / fname_a
            path_b = self.images_dir / fname_b
            if path_a.exists() and path_b.exists():
                self.pairs.append({
                    "path_a": path_a, "path_b": path_b,
                    "rating": float(row["Rating"])
                })
        print(f"PairwiseDataset: {len(self.pairs)}/{len(df)} pairs found")

        self.resize = transforms.Resize((img_size, img_size))
        self.jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1
        )

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img):
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
        return img

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img_a = self.resize(Image.open(p["path_a"]).convert("RGB"))
        img_b = self.resize(Image.open(p["path_b"]).convert("RGB"))

        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            img_a = self._augment(img_a)
            random.seed(seed)
            img_b = self._augment(img_b)

        img_a = NORMALIZE(TF.to_tensor(img_a))
        img_b = NORMALIZE(TF.to_tensor(img_b))
        target = torch.tensor((p["rating"] - 1) / 4.0, dtype=torch.float32)
        return img_a, img_b, target


class AllImagesDataset(Dataset):
    """All analysis-set images: {root}/{city}/*.png — returns (tensor, idx)."""

    def __init__(self, images_root, img_size=224):
        self.paths = []
        self.grid_ids = []
        for city_dir in sorted(Path(images_root).iterdir()):
            if not city_dir.is_dir():
                continue
            for p in sorted(city_dir.glob("*.png")):
                self.paths.append(p)
                self.grid_ids.append(p.stem)
        print(f"AllImagesDataset: {len(self.paths)} images")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            NORMALIZE,
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


class MorphologyDataset(Dataset):
    """Morphological features optionally paired with teacher embeddings."""

    def __init__(self, features: np.ndarray, grid_ids: list,
                 embeddings: np.ndarray = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.grid_ids = grid_ids
        self.embeddings = (
            torch.tensor(embeddings, dtype=torch.float32)
            if embeddings is not None else None
        )

    def __len__(self):
        return len(self.grid_ids)

    def __getitem__(self, idx):
        item = {"features": self.features[idx], "grid_id": self.grid_ids[idx]}
        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]
        return item


def load_morphology_features(features_csv, train_grid_ids=None):
    """
    Load and NaN-fill morphological features. Returns (features, grid_ids, feature_names, medians).
    If train_grid_ids is given, medians are computed on that subset only to avoid leakage.
    """
    _META_COLS    = {"GridID", "center_x", "center_y", "City", "Country"}
    _EXCLUDE_COLS = {"Area_m2"}  # constant across all 250m hex cells

    df = pd.read_csv(features_csv)
    feat_cols = [c for c in df.columns
                 if c not in _META_COLS and c not in _EXCLUDE_COLS]

    grid_ids = df["GridID"].tolist()

    zero_cols = [c for c in feat_cols
                 if any(c.startswith(p) for p in _ZERO_FILL_PREFIXES)]
    med_cols  = [c for c in feat_cols if c not in zero_cols]

    if train_grid_ids is not None:
        mask = df["GridID"].isin(set(train_grid_ids))
        medians = df.loc[mask, med_cols].median()
    else:
        medians = df[med_cols].median()

    df[zero_cols] = df[zero_cols].fillna(0.0)
    df[med_cols]  = df[med_cols].fillna(medians)

    return df[feat_cols].values.astype(np.float32), grid_ids, feat_cols, medians


def fit_scaler(features: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler

