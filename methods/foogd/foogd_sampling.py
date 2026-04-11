"""Score-model sampling utilities adapted from the official FOOGD code."""

from __future__ import annotations

import numpy as np
import torch


def langevin_dynamics(score_fn, x: torch.Tensor, eps: float = 0.1, n_steps: int = 1000) -> torch.Tensor:
    for _ in range(n_steps):
        x = x + eps / 2.0 * score_fn(x)
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x


def anneal_langevin_dynamics(
    score_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor | None = None,
    eps: float = 0.1,
    n_steps: int = 100,
) -> torch.Tensor:
    if sigmas is None:
        sigmas = torch.tensor(np.exp(np.linspace(np.log(20.0), 0.0, 10)), device=x.device, dtype=x.dtype)
    for sigma in sigmas:
        sigma_value = float(sigma.item())
        for _ in range(n_steps):
            cur_eps = eps * (sigma_value / float(sigmas[-1].item())) ** 2
            x = x + cur_eps / 2.0 * score_fn(x)
            x = x + torch.randn_like(x) * np.sqrt(eps)
    return x
