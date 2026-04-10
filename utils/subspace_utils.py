#!/usr/bin/env python3
"""Helpers for selecting ViM/Fed-ViM subspace dimensions."""

from __future__ import annotations

import numpy as np
import torch


def select_vim_paper_k(feature_dim: int, num_classes: int) -> int:
    """
    Reproduce the heuristic principal-subspace dimension rule described in ViM.

    Rules used here:
    - if N > 1500: D = 1000
    - elif N <= C: D = N/2 (middle of [N/3, 2N/3])
    - else: D = 512
    """
    feature_dim = int(feature_dim)
    num_classes = int(num_classes)

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    if feature_dim > 1500:
        k = 1000
    elif feature_dim <= num_classes:
        k = feature_dim // 2
    else:
        k = 512

    return max(1, min(k, feature_dim))


def covariance_to_correlation(cov_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a covariance matrix into a numerically stable correlation matrix."""
    device = cov_matrix.device
    diag = torch.clamp(torch.diag(cov_matrix), min=1e-6)
    std = torch.sqrt(diag)
    outer_std = torch.outer(std, std)

    corr_matrix = cov_matrix / (outer_std + 1e-8)
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    corr_matrix = corr_matrix + torch.eye(corr_matrix.shape[0], device=device) * (1.0 - torch.diag(corr_matrix))
    return corr_matrix


def compute_act_k_from_covariance(cov_global: torch.Tensor, n_samples: int) -> tuple[int, torch.Tensor, torch.Tensor, float, float]:
    """Estimate ACT rank and return corrected/raw eigen spectra."""
    if n_samples <= 1:
        raise ValueError("ACT requires at least two samples.")

    device = cov_global.device
    p = cov_global.shape[0]
    corr = covariance_to_correlation(cov_global)

    eig_vals = None
    for jitter in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        try:
            corr_jittered = corr + torch.eye(p, device=device) * jitter
            eig_vals, _ = torch.linalg.eigh(corr_jittered)
            if torch.any(torch.isnan(eig_vals)) or torch.any(torch.isinf(eig_vals)):
                raise ValueError("NaN/Inf in eigenvalues")
            break
        except (torch._C._LinAlgError, ValueError):
            continue

    if eig_vals is None:
        raise RuntimeError("Failed to compute ACT correlation eigenvalues.")

    eig_vals = torch.flip(eig_vals, dims=[0])
    rho = p / (n_samples - 1)
    threshold_s = 1.0 + np.sqrt(rho)

    corrected_eigs = torch.zeros_like(eig_vals)
    corrected_eigs[-1] = eig_vals[-1]

    for j in range(p - 1):
        lambda_j = eig_vals[j]
        if lambda_j < 0.5:
            corrected_eigs[j] = lambda_j
            continue

        lambdas_rest = eig_vals[j + 1 :]
        diffs = lambdas_rest - lambda_j
        mask = torch.abs(diffs) > 1e-6
        if mask.sum() == 0:
            sum_term = 0.0
        else:
            sum_term = torch.sum(1.0 / diffs[mask])

        correction_term = 1.0 / (((3 * lambda_j + eig_vals[j + 1]) / 4) - lambda_j + 1e-8)
        m_nj = (1.0 / (p - j)) * (sum_term + correction_term)
        rho_j = (p - j) / (n_samples - 1)
        underline_m = -(1 - rho_j) * (1.0 / lambda_j) + rho_j * m_nj
        corrected_eigs[j] = -1.0 / (underline_m + 1e-8)

    valid_indices = torch.where(corrected_eigs > threshold_s)[0]
    optimal_k = int(valid_indices[-1].item() + 1) if len(valid_indices) else 1
    optimal_k = max(1, min(optimal_k, p))
    return optimal_k, corrected_eigs, eig_vals, float(rho), float(threshold_s)
