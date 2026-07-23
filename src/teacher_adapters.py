"""Trainable modality adapters for the frozen DFN5B teacher."""

import torch.nn as nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return F.normalize(features + residual, dim=-1)


class ModalityAdapters(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super().__init__()
        self.sketch = ResidualAdapter(feature_dim, bottleneck_dim)
        self.photo = ResidualAdapter(feature_dim, bottleneck_dim)
