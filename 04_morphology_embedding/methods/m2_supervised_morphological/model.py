"""M2 model: MLP metric model for morphological features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyMetricModel(nn.Module):
    """
    MLP: 165-dim features → 256-dim L2-normalized embedding.

    Architecture: input → 512 → 256 → output_dim
    Each hidden layer: Linear + BatchNorm + ReLU + Dropout
    Output: L2-normalized (unit sphere, matches teacher embedding convention)
    """

    def __init__(self, input_dim=165, hidden_dims=(512, 256), output_dim=256, dropout=0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return F.normalize(self.network(x), dim=-1)


def soft_contrastive_loss(emb_a, emb_b, target_sim):
    """MSE between cosine similarity and normalized target. Embeddings must be L2-normalized."""
    cos_sim = (emb_a * emb_b).sum(dim=-1)
    return F.mse_loss(cos_sim, target_sim)
