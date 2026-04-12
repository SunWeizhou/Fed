#!/usr/bin/env python3
"""Shared OOD scoring, metric, and ViM calibration utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def compute_msp_ood_scores(logits):
    """Compute MSP-based OOD scores where larger means more likely OOD."""
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return 1.0 - probs.max(axis=1)


def compute_energy_ood_scores(logits):
    """Compute energy-based OOD scores where larger means more likely OOD."""
    logits = np.asarray(logits)
    max_logits = logits.max(axis=1, keepdims=True)
    stabilized = logits - max_logits
    logsumexp = max_logits[:, 0] + np.log(np.exp(stabilized).sum(axis=1))
    return -logsumexp


def compute_vim_scores(energy, residual, alpha):
    """Compute ViM scores from energy, residual, and alpha."""
    return np.asarray(energy) - float(alpha) * np.asarray(residual)


def _iter_loaders(loaders):
    if isinstance(loaders, (list, tuple)):
        for loader in loaders:
            yield loader
    else:
        yield loaders


def compute_empirical_alpha_local_stats(model, loader, P, mu, device=None):
    """
    Compute per-loader empirical alpha statistics.

    Returns:
        dict with sum_energy, sum_residual, count
    """
    if device is None:
        device = P.device

    P = P.to(device=device, dtype=torch.float64)
    mu = mu.to(device=device, dtype=torch.float64)

    was_training = model.training
    model.eval()

    total_energy = 0.0
    total_residual = 0.0
    total_count = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits, features = model(images)

            logits = logits.to(dtype=torch.float64)
            features = features.to(dtype=torch.float64)

            energy = torch.logsumexp(logits, dim=1)
            centered = features - mu
            proj = centered @ P
            recon = proj @ P.T
            residual = torch.norm(centered - recon, p=2, dim=1)

            total_energy += energy.sum().item()
            total_residual += residual.sum().item()
            total_count += images.size(0)

    if was_training:
        model.train()

    return {
        "sum_energy": total_energy,
        "sum_residual": total_residual,
        "count": total_count,
    }


def aggregate_empirical_alpha_statistics(local_stats_list):
    """Aggregate client-local energy/residual statistics into a global empirical alpha."""
    total_energy = 0.0
    total_residual = 0.0
    total_count = 0

    for stats in local_stats_list:
        if not stats:
            continue
        total_energy += float(stats.get("sum_energy", 0.0))
        total_residual += float(stats.get("sum_residual", 0.0))
        total_count += int(stats.get("count", 0))

    if total_count == 0:
        raise ValueError("No samples were available for empirical alpha estimation.")

    mean_energy = total_energy / total_count
    mean_residual = total_residual / total_count
    alpha = abs(mean_energy) / (mean_residual + 1e-8)

    print(f"  [Empirical Alpha] Mean Energy: {mean_energy:.4f}")
    print(f"  [Empirical Alpha] Mean Residual: {mean_residual:.4f}")
    print(f"  [Empirical Alpha] Alpha = {alpha:.4f}")
    return alpha, mean_energy, mean_residual


def estimate_vim_alpha_empirical(model, loaders, P, mu, device=None):
    """
    Estimate ViM alpha empirically from ID training features.

    This computes:
        alpha = |mean(energy)| / mean(residual)
    over one or more ID train loaders.
    """
    if device is None:
        device = P.device

    local_stats_list = [
        compute_empirical_alpha_local_stats(model, loader, P, mu, device=device)
        for loader in _iter_loaders(loaders)
    ]
    return aggregate_empirical_alpha_statistics(local_stats_list)


def estimate_vim_alpha_from_statistics(P, cov_global, global_model=None, mu_global=None, device=None, num_classes=54):
    """
    Estimate ViM alpha analytically from aggregated statistics.

    This is the paper-aligned calibration path:
    1. Estimate mean residual from global covariance and subspace projector.
    2. Estimate mean energy by evaluating the classifier at mu_global.
    """
    if device is None:
        device = P.device

    P = P.to(device=device, dtype=torch.float64)
    cov_global = cov_global.to(device=device, dtype=torch.float64)
    D = P.shape[0]

    identity = torch.eye(D, device=device, dtype=torch.float64)
    projector_residual = identity - P @ P.T
    expected_residual_sq = torch.trace(projector_residual @ cov_global @ projector_residual.T)
    mean_residual = torch.sqrt(torch.clamp(expected_residual_sq, min=0)).item()

    if global_model is not None and mu_global is not None:
        with torch.no_grad():
            global_model.eval()
            model_obj = global_model.module if hasattr(global_model, 'module') else global_model

            if hasattr(model_obj, 'classifier'):
                classifier = model_obj.classifier
            elif hasattr(model_obj, 'heads'):
                classifier = model_obj.heads
            else:
                raise ValueError("Cannot find classifier head in global_model")

            mu_global = mu_global.to(device=device, dtype=torch.float64)

            if isinstance(classifier, nn.Linear):
                logits_mu = F.linear(
                    mu_global,
                    classifier.weight.to(device=device, dtype=torch.float64),
                    classifier.bias.to(device=device, dtype=torch.float64) if classifier.bias is not None else None,
                )
            elif isinstance(classifier, nn.Sequential):
                hidden = mu_global
                for module in classifier:
                    hidden = module(hidden)
                logits_mu = hidden
            else:
                raise ValueError(f"Unsupported classifier type: {type(classifier)}")

            estimated_mean_energy = torch.logsumexp(logits_mu, dim=0).item()

        print(f"  [Analytic Alpha] Using Energy(mu_global): {estimated_mean_energy:.4f}")
    else:
        estimated_mean_energy = np.log(num_classes)
        print(f"  [Analytic Alpha Fallback] Using log(C): {estimated_mean_energy:.4f}")

    alpha = abs(estimated_mean_energy) / (mean_residual + 1e-8)
    print(f"  [Analytic Alpha] Mean Residual: {mean_residual:.4f}")
    print(f"  [Analytic Alpha] Alpha = {alpha:.4f}")
    return alpha


def compute_ood_metrics(id_scores, ood_scores, invert_scores=True):
    """
    Compute AUROC, AUPR, and FPR95 for ID vs OOD scores.

    Args:
        id_scores: Scores for in-distribution samples.
        ood_scores: Scores for out-of-distribution samples.
        invert_scores: If True, flip score sign before metric computation.
            This matches the original ViM convention where lower scores imply OOD.
    """
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    if invert_scores:
        y_scores = -y_scores

    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
    }
