"""
Pairwise morphology dataset for M2.

Loads annotation pairs and looks up feature vectors from the pre-loaded
feature matrix by GridID. Rating normalized [1,5] → [0,1].
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MorphologyPairDataset(Dataset):
    """
    Each item: (feat_a, feat_b, target_similarity)
    feat_*: [165] normalized feature vector (looked up from feature_matrix)
    """

    def __init__(self, annotation_csv, feature_matrix: np.ndarray,
                 grid_id_to_idx: dict):
        df = pd.read_csv(annotation_csv)
        self.pairs = []
        skipped = 0
        for _, row in df.iterrows():
            gid_a = row["Image_A_gridID"]
            gid_b = row["Image_B_gridID"]
            if gid_a in grid_id_to_idx and gid_b in grid_id_to_idx:
                self.pairs.append((
                    grid_id_to_idx[gid_a],
                    grid_id_to_idx[gid_b],
                    float(row["Rating"]),
                ))
            else:
                skipped += 1
        print(f"MorphologyPairDataset: {len(self.pairs)} pairs ({skipped} skipped — no features)")

        self.features = torch.tensor(feature_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ia, ib, rating = self.pairs[idx]
        target = torch.tensor((rating - 1) / 4.0, dtype=torch.float32)
        return self.features[ia], self.features[ib], target
