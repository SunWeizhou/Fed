"""Class-conditional feature generator used by the thesis FOSTER adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassConditionalFeatureGenerator(nn.Module):
    """Generate synthetic feature vectors conditioned on class labels."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        noise_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)

        self.label_embedding = nn.Embedding(self.num_classes, self.noise_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.noise_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

    def forward(self, labels: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        labels = labels.long()
        if noise is None:
            noise = torch.randn(labels.size(0), self.noise_dim, device=labels.device)
        label_vec = self.label_embedding(labels)
        fused = torch.cat([label_vec, noise], dim=1)
        return self.mlp(fused)
