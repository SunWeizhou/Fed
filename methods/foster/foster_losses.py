"""Loss helpers for the thesis-oriented FOSTER baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def oe_loss(logits: torch.Tensor) -> torch.Tensor:
    """Outlier Exposure loss used for synthetic external features."""
    return -(logits.mean(dim=1) - torch.logsumexp(logits, dim=1)).mean()


def classifier_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy helper kept separate for clarity in client training."""
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
