"""Score model used by the thesis-oriented FOOGD baseline."""

from __future__ import annotations

import torch
from torch import nn


class FOOGDScoreModel(nn.Module):
    """A simple MLP score network that maps features to feature-shaped scores."""

    def __init__(self, feature_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_dim = int(max(hidden_dim, feature_dim // 2))
        self.main = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.main(features)
