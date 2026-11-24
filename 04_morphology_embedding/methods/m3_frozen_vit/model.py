"""
Model definitions shared with M4 (finetuned_vit).

M3 itself performs no training and extract.py does not import from this
module — it calls the frozen DINOv3 backbone directly and L2-normalizes the
raw 1024-dim CLS token (see extract.py). The ProjectionHead / ClusterLayer /
DeepClusteringModel classes below are the same model definitions used by M4;
they are kept in M3's directory for reference/parity with M4's architecture
rather than being exercised by any M3 script.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backbone" / "dinov3"))
from dinov3.hub.backbones import dinov3_vitl16


# =============================================================================
# Building blocks
# =============================================================================

class ProjectionHead(nn.Module):
    """MLP: backbone_dim → 512 → projection_dim, L2-normalized output."""

    def __init__(self, input_dim=1024, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


class ClusterLayer(nn.Module):
    """
    Learnable cluster centers on the unit hypersphere.

    Soft assignment via cosine similarity + softmax with temperature:
      q_ik = exp(cos(z_i, mu_k) / tau) / sum_j exp(cos(z_i, mu_j) / tau)
    """

    def __init__(self, n_clusters, embed_dim, temperature=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.centers = nn.Parameter(torch.randn(n_clusters, embed_dim))

    def initialize_from_kmeans(self, centers):
        """Set centers from K-means result (L2-normalized)."""
        self.centers.data.copy_(F.normalize(centers, dim=-1))

    def forward(self, z):
        """z: [B, D] → q: [B, K] soft assignments."""
        centers_norm = F.normalize(self.centers, dim=-1)
        cos_sim = z @ centers_norm.T
        return F.softmax(cos_sim / self.temperature, dim=1)


# =============================================================================
# Full models
# =============================================================================

class MetricModel(nn.Module):
    """Stage 1 model: frozen DINOv3 backbone + trainable projection head."""

    def __init__(self, weights_path, projection_dim=256):
        super().__init__()
        self.backbone = dinov3_vitl16(pretrained=True, weights=weights_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = ProjectionHead(input_dim=1024, output_dim=projection_dim)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

    def get_trainable_params(self):
        return self.head.parameters()


class DeepClusteringModel(nn.Module):
    """Stage 2 model: MetricModel + multi-scale ClusterLayers.

    Maintains one ClusterLayer per K in `cluster_scales` (e.g. [10, 20, 30]).
    All scales share the same ProjectionHead, so gradients from every scale
    shape the embedding space jointly.
    """

    def __init__(self, weights_path, projection_dim=256, cluster_scales=(10, 20, 30)):
        super().__init__()
        self.metric_model = MetricModel(weights_path, projection_dim)
        self.cluster_scales = list(cluster_scales)
        self.cluster_layers = nn.ModuleDict({
            str(k): ClusterLayer(k, projection_dim) for k in self.cluster_scales
        })

    def forward(self, x):
        """Returns z and dict {str(k): q_k} for all scales."""
        z = self.metric_model(x)
        qs = {k: layer(z) for k, layer in self.cluster_layers.items()}
        return z, qs

    def get_embeddings(self, x):
        return self.metric_model(x)

    def get_trainable_params(self):
        params = list(self.metric_model.head.parameters())
        for layer in self.cluster_layers.values():
            params.extend(layer.parameters())
        return params


# =============================================================================
# Loss functions
# =============================================================================

def soft_contrastive_loss(emb_a, emb_b, target_sim):
    """MSE between cosine similarity and target. Embeddings must be L2-normalized."""
    cos_sim = (emb_a * emb_b).sum(dim=-1)
    return F.mse_loss(cos_sim, target_sim)


def compute_target_distribution(q):
    """DEC target distribution P: sharpen Q by squaring and normalizing by cluster freq."""
    f = q.sum(dim=0, keepdim=True)
    p = (q ** 2) / f
    return p / p.sum(dim=1, keepdim=True)


def clustering_loss(q, p):
    """KL(P || Q) — DEC clustering loss."""
    return F.kl_div(q.log(), p, reduction='batchmean')
