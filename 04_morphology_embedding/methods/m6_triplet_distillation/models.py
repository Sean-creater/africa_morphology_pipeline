"""M6 StudentMLP encoder."""
import torch.nn as nn
import torch.nn.functional as F


class StudentMLP(nn.Module):
    """MLP: morphology features → L2-normalised embedding."""

    def __init__(self, input_dim, hidden_dims=(512,),
                 dropout=0.2, batch_norm=True, output_dim=256):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

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
        return F.normalize(self.network(x), p=2, dim=-1)

