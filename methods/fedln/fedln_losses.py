"""Losses used by the independent FedLN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitNormLoss(nn.Module):
    """LogitNorm objective from the original ICML 2022 paper."""

    def __init__(self, temperature: float = 0.01) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        normalized_logits = torch.div(logits, norms) / self.temperature
        return F.cross_entropy(normalized_logits, targets)
