#!/usr/bin/env python3
"""
WDiscOOD: Whitened Linear Discriminant Analysis for OOD Detection
Based on: "WDiscOOD: Out-of-Distribution Detection via Whitened Linear Discriminant Analysis" (ICCV 2023)
Paper: https://arxiv.org/abs/2303.07543
Original Code: https://github.com/ivalab/WDiscOOD

This script implements a federated-compatible version of WDiscOOD.
Instead of PCA (which maximizes variance), WDiscOOD uses LDA (which maximizes class separability).

Key Innovation:
- PCA: max variance → may miss discriminative directions
- LDA: max between-class / within-class scatter → directly optimizes for discrimination

Mathematical Formulation:
1. Between-class scatter: S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
2. Within-class scatter: S_w = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)^T
3. WLDA: Find W that maximizes tr(W^T S_b W) / tr(W^T S_w W)
4. Discriminative subspace: top (C-1) eigenvectors of S_w^{-1} S_b
5. Residual subspace: remaining directions orthogonal to discriminative subspace

Scoring Modes:
- Pure WDiscOOD (--scoring_mode pure):
    Score = -(disc_dist + λ * resid_dist)
    Uses only geometric distances, no Energy term.
    
- WDiscOOD-Energy Hybrid (--scoring_mode hybrid, default):
    Score = Energy - α * (disc_dist + λ * resid_dist)
    Combines WLDA geometry with classifier confidence.

Author: Claude Code
Date: 2025-03-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json
import sys
from datetime import datetime
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import FedAvg_Model, Backbone
from data_utils import create_id_train_client_loaders_only, create_test_loaders_only
from server import FLServer
from utils.ood_utils import compute_ood_metrics


class WDiscOOD:
    """
    Whitened Linear Discriminant Analysis for OOD Detection.
    
    Unlike PCA which finds directions of maximum variance,
    WLDA finds directions that maximize class separation (between-class scatter)
    while minimizing within-class scatter.
    
    This is more aligned with OOD detection goals:
    - Discriminative subspace: captures class-specific patterns
    - Residual subspace: captures class-agnostic patterns (anomalies project here)
    """
    
    def __init__(self, device='cpu', regularization=1e-5):
        self.device = device
        self.regularization = regularization
        
        # Subspace matrices
        self.W_disc = None      # Discriminative subspace projection [D, C-1]
        self.W_resid = None     # Residual subspace projection [D, D-C+1]
        
        # Class statistics
        self.class_means = None  # [C, D]
        self.global_mean = None  # [D]
        self.n_classes = None
        
        # Eigenvalues for diagnostics
        self.disc_eigenvalues = None
        
    def fit_from_features(self, features, labels):
        """
        Fit WLDA from raw features and labels.
        This is for standalone testing, not federated learning.
        
        Args:
            features: [N, D] tensor of feature vectors
            labels: [N] tensor of class labels (0 to C-1)
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        n_samples, feature_dim = features.shape
        unique_labels = torch.unique(labels)
        self.n_classes = len(unique_labels)
        
        print(f"[WDiscOOD] Fitting from {n_samples} samples, {self.n_classes} classes, dim={feature_dim}")
        
        # Compute class means and global mean
        self.class_means = torch.zeros(self.n_classes, feature_dim, device=self.device)
        class_counts = torch.zeros(self.n_classes, device=self.device)
        
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.class_means[c] = features[mask].mean(dim=0)
                class_counts[c] = mask.sum()
        
        self.global_mean = features.mean(dim=0)
        
        # Compute within-class scatter S_w
        S_w = torch.zeros(feature_dim, feature_dim, device=self.device)
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                centered = features[mask] - self.class_means[c]
                S_w += centered.T @ centered
        
        # Compute between-class scatter S_b
        centered_means = self.class_means - self.global_mean
        S_b = (centered_means.T * class_counts) @ centered_means
        
        # Fit WLDA subspaces
        self._fit_wlda(S_w, S_b, class_counts.sum().item())
    
    def fit_from_statistics(self, class_means, class_counts, within_class_cov):
        """
        Fit WLDA from aggregated statistics (federated learning compatible).
        
        This is the key method for privacy-preserving federated OOD detection.
        Each client only shares:
        1. Per-class feature sums: Σ_{x∈c} z_x
        2. Per-class sample counts: n_c
        3. Per-class second moment: Σ_{x∈c} z_x z_x^T
        
        Server can then reconstruct:
        - Class means: μ_c = (Σ z_x) / n_c
        - Within-class covariance: S_w = Σ_c [E[zz^T|c] - μ_c μ_c^T]
        - Between-class covariance: S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
        
        Args:
            class_means: [C, D] tensor of per-class mean vectors
            class_counts: [C] tensor of per-class sample counts
            within_class_cov: [D, D] tensor of pooled within-class covariance
        """
        self.class_means = class_means.to(self.device)
        class_counts = class_counts.to(self.device)
        within_class_cov = within_class_cov.to(self.device)
        
        self.n_classes = class_means.shape[0]
        feature_dim = class_means.shape[1]
        total_samples = class_counts.sum().item()
        
        print(f"[WDiscOOD] Fitting from statistics: {self.n_classes} classes, dim={feature_dim}")
        print(f"[WDiscOOD] Total samples: {total_samples}")
        
        # Global mean (weighted average of class means)
        self.global_mean = (class_means.T @ class_counts) / total_samples
        
        # Within-class scatter (already provided or computed from statistics)
        # S_w = n * within_class_cov (if normalized) or directly use the sum
        S_w = within_class_cov * total_samples
        
        # Between-class scatter
        centered_means = self.class_means - self.global_mean
        S_b = (centered_means.T * class_counts) @ centered_means
        
        # Fit WLDA subspaces
        self._fit_wlda(S_w, S_b, total_samples)
    
    def _fit_wlda(self, S_w, S_b, n_samples):
        """
        Core WLDA fitting algorithm.
        
        Steps:
        1. Regularize S_w for numerical stability
        2. Whiten using S_w^{-1/2}
        3. Compute whitened between-class scatter
        4. Extract discriminative directions via eigendecomposition
        5. Transform back to original space
        6. Compute residual subspace as orthogonal complement
        """
        feature_dim = S_w.shape[0]
        
        # Step 1: Regularize within-class scatter
        S_w_reg = S_w + self.regularization * torch.eye(feature_dim, device=self.device) * torch.trace(S_w)
        
        # Step 2: Compute S_w^{-1/2} via eigendecomposition
        try:
            eig_vals_w, eig_vecs_w = torch.linalg.eigh(S_w_reg)
        except RuntimeError as e:
            print(f"[WDiscOOD] Warning: eigh failed, using SVD fallback: {e}")
            U, S, Vh = torch.linalg.svd(S_w_reg)
            eig_vals_w = S
            eig_vecs_w = U
        
        # Ensure positive eigenvalues (numerical stability)
        eig_vals_w = torch.clamp(eig_vals_w, min=1e-10)
        
        # S_w^{-1/2} = V * diag(1/sqrt(λ)) * V^T
        S_w_inv_sqrt = eig_vecs_w @ torch.diag(1.0 / torch.sqrt(eig_vals_w)) @ eig_vecs_w.T
        
        # Step 3: Whitened between-class scatter
        S_b_whitened = S_w_inv_sqrt @ S_b @ S_w_inv_sqrt
        
        # Step 4: Eigendecomposition of whitened S_b
        try:
            eig_vals_b, eig_vecs_b = torch.linalg.eigh(S_b_whitened)
        except RuntimeError as e:
            print(f"[WDiscOOD] Warning: eigh failed for S_b_whitened, using SVD: {e}")
            U, S, Vh = torch.linalg.svd(S_b_whitened)
            eig_vals_b = S
            eig_vecs_b = U
        
        # Sort by eigenvalue (descending)
        sorted_indices = torch.argsort(eig_vals_b, descending=True)
        eig_vals_b = eig_vals_b[sorted_indices]
        eig_vecs_b = eig_vecs_b[:, sorted_indices]
        
        # Store eigenvalues for diagnostics
        self.disc_eigenvalues = eig_vals_b.cpu().numpy()
        
        # Step 5: Discriminative subspace (C-1 dimensions maximum for LDA)
        # In practice, we use all directions with significant eigenvalues
        disc_dim = min(self.n_classes - 1, feature_dim)
        
        # Find effective dimension (eigenvalues above threshold)
        eig_threshold = 1e-6 * eig_vals_b[0] if eig_vals_b[0] > 0 else 1e-6
        effective_dim = (eig_vals_b > eig_threshold).sum().item()
        disc_dim = min(disc_dim, effective_dim)
        
        print(f"[WDiscOOD] Discriminative dimension: {disc_dim} (max possible: {self.n_classes - 1})")
        print(f"[WDiscOOD] Top 5 discriminative eigenvalues: {eig_vals_b[:5].cpu().numpy()}")
        
        # Transform back to original space: W_disc = S_w^{-1/2} @ V_b[:, :disc_dim]
        self.W_disc = S_w_inv_sqrt @ eig_vecs_b[:, :disc_dim]  # [D, disc_dim]
        
        # Step 6: Residual subspace (orthogonal complement)
        # Use remaining eigenvectors
        resid_dim = feature_dim - disc_dim
        if resid_dim > 0:
            self.W_resid = S_w_inv_sqrt @ eig_vecs_b[:, disc_dim:]  # [D, resid_dim]
        else:
            # Edge case: no residual subspace
            self.W_resid = torch.zeros(feature_dim, 1, device=self.device)
        
        print(f"[WDiscOOD] Residual dimension: {self.W_resid.shape[1]}")
        
        # Compute class means in discriminative space for scoring
        # IMPORTANT: Center class means before projection for correct distance calculation
        class_means_centered = self.class_means - self.global_mean
        self.class_means_disc = class_means_centered @ self.W_disc  # [C, disc_dim]
        
    def compute_scores(self, features, lambda_weight=1.0):
        """
        Compute WDiscOOD scores for a batch of features.
        
        OOD Score = Discriminative Distance + λ * Residual Distance
        
        Where:
        - Discriminative Distance: min distance to class means in discriminative subspace
        - Residual Distance: distance from origin in residual subspace
        
        Higher score → more likely OOD
        
        Args:
            features: [N, D] tensor of feature vectors
            lambda_weight: weight for residual component (default 1.0)
            
        Returns:
            scores: [N] tensor of OOD scores (higher = more OOD)
        """
        features = features.to(self.device)
        
        # Center features
        centered = features - self.global_mean  # [N, D]
        
        # Project to discriminative subspace
        z_disc = centered @ self.W_disc  # [N, disc_dim]
        
        # Project to residual subspace  
        z_resid = centered @ self.W_resid  # [N, resid_dim]
        
        # Discriminative distance: min Euclidean distance to class means
        # For each sample, find the closest class mean in discriminative space
        # [N, C] distances
        class_means_disc = self.class_means_disc  # [C, disc_dim]
        
        # Efficient pairwise distance: ||z - μ_c||^2 = ||z||^2 - 2<z,μ_c> + ||μ_c||^2
        z_disc_sq = (z_disc ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        mu_disc_sq = (class_means_disc ** 2).sum(dim=1, keepdim=True).T  # [1, C]
        cross_term = z_disc @ class_means_disc.T  # [N, C]
        
        dists_sq = z_disc_sq - 2 * cross_term + mu_disc_sq  # [N, C]
        dists_sq = torch.clamp(dists_sq, min=0)  # Numerical stability
        
        # Minimum distance to any class mean
        min_disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])  # [N]
        
        # Residual distance: L2 norm in residual subspace
        resid_dist = torch.norm(z_resid, p=2, dim=1)  # [N]
        
        # Combined score (higher = more OOD)
        # Note: We negate because for OOD detection, we want ID samples to have HIGH scores
        # and OOD samples to have LOW scores (to match ViM convention)
        # WDiscOOD original: higher disc_dist and resid_dist → more OOD
        # Our convention: higher score → more ID (to match ViM)
        # So we return: score = -disc_dist - λ * resid_dist
        scores = -(min_disc_dist + lambda_weight * resid_dist)
        
        return scores, min_disc_dist, resid_dist
    
    def compute_scores_with_energy(self, features, logits, alpha=1.0, lambda_weight=1.0):
        """
        Hybrid scoring: combine WDiscOOD with energy-based scoring.
        
        This combines the benefits of:
        1. WDiscOOD: discriminative subspace analysis
        2. Energy: classifier confidence
        
        Score = Energy - α * (Disc_Distance + λ * Resid_Distance)
        
        Args:
            features: [N, D] feature vectors
            logits: [N, C] classifier logits
            alpha: weight for geometric component
            lambda_weight: weight for residual vs discriminative
            
        Returns:
            scores: [N] OOD scores (higher = more ID)
        """
        # Energy component (higher = more confident = more ID)
        energy = torch.logsumexp(logits, dim=1)  # [N]
        
        # WDiscOOD component
        _, disc_dist, resid_dist = self.compute_scores(features, lambda_weight)
        
        # Combined score
        # Higher energy → more ID
        # Lower distances → more ID
        scores = energy - alpha * (disc_dist + lambda_weight * resid_dist)
        
        return scores


def extract_class_statistics(model, data_loaders, device, num_classes):
    """
    Extract per-class statistics from data loaders.
    
    This function extracts:
    1. Per-class feature sums
    2. Per-class feature second moments
    3. Per-class sample counts
    
    These statistics can be shared in federated learning without revealing raw data.
    
    Args:
        model: The feature extractor model
        data_loaders: List of data loaders (one per client) or single loader
        device: torch device
        num_classes: Number of classes
        
    Returns:
        class_sums: [C, D] sum of features per class
        class_sum_sq: [C, D, D] sum of outer products per class
        class_counts: [C] sample count per class
    """
    model.eval()
    
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = [data_loaders]
    
    feature_dim = None
    class_sums = None
    class_sum_sq = None
    class_counts = None
    
    with torch.no_grad():
        for loader in data_loaders:
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                output = model(images)
                if len(output) == 2:
                    logits, features = output
                else:
                    logits, _, features = output
                
                # Initialize accumulators on first batch
                if feature_dim is None:
                    feature_dim = features.shape[1]
                    class_sums = torch.zeros(num_classes, feature_dim, device=device)
                    class_sum_sq = torch.zeros(num_classes, feature_dim, feature_dim, device=device)
                    class_counts = torch.zeros(num_classes, device=device)
                
                # Accumulate per-class statistics
                for c in range(num_classes):
                    mask = labels == c
                    if mask.sum() > 0:
                        class_features = features[mask]  # [n_c, D]
                        class_sums[c] += class_features.sum(dim=0)
                        class_sum_sq[c] += class_features.T @ class_features
                        class_counts[c] += mask.sum()
    
    return class_sums, class_sum_sq, class_counts


def compute_wlda_statistics(class_sums, class_sum_sq, class_counts):
    """
    Compute WLDA statistics from aggregated per-class statistics.
    
    Args:
        class_sums: [C, D] sum of features per class
        class_sum_sq: [C, D, D] sum of outer products per class
        class_counts: [C] sample count per class
        
    Returns:
        class_means: [C, D] per-class mean vectors
        within_class_cov: [D, D] pooled within-class covariance
    """
    n_classes, feature_dim = class_sums.shape
    device = class_sums.device
    
    # Compute class means
    class_means = torch.zeros(n_classes, feature_dim, device=device)
    for c in range(n_classes):
        if class_counts[c] > 0:
            class_means[c] = class_sums[c] / class_counts[c]
    
    # Compute within-class covariance
    # S_w = Σ_c [Σ_{x∈c} (x - μ_c)(x - μ_c)^T]
    #     = Σ_c [Σ_{x∈c} xx^T - n_c * μ_c * μ_c^T]
    within_class_cov = torch.zeros(feature_dim, feature_dim, device=device)
    total_samples = class_counts.sum().item()
    
    for c in range(n_classes):
        if class_counts[c] > 0:
            # E[xx^T | c] - μ_c * μ_c^T
            class_cov = class_sum_sq[c] / class_counts[c] - torch.outer(class_means[c], class_means[c])
            within_class_cov += class_cov * class_counts[c]
    
    # Normalize to get pooled covariance
    within_class_cov = within_class_cov / total_samples
    
    return class_means, within_class_cov


def calibrate_alpha(model, data_loaders, wdiscood, device, lambda_weight=1.0):
    """
    Calibrate alpha parameter using ID training data.
    
    Alpha balances energy and geometric distance:
    α = |mean(energy)| / mean(disc_dist + λ * resid_dist)
    
    Args:
        model: Feature extractor
        data_loaders: ID training loaders
        wdiscood: Fitted WDiscOOD object
        device: torch device
        lambda_weight: Weight for residual component
        
    Returns:
        alpha: Calibrated alpha value
    """
    model.eval()
    
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = [data_loaders]
    
    total_energy = 0.0
    total_disc_dist = 0.0
    total_resid_dist = 0.0
    total_count = 0
    
    with torch.no_grad():
        for loader in data_loaders:
            for images, _ in loader:
                images = images.to(device)
                
                output = model(images)
                if len(output) == 2:
                    logits, features = output
                else:
                    logits, _, features = output
                
                # Energy
                energy = torch.logsumexp(logits, dim=1)
                total_energy += energy.sum().item()
                
                # WDiscOOD distances
                _, disc_dist, resid_dist = wdiscood.compute_scores(features, lambda_weight)
                total_disc_dist += disc_dist.sum().item()
                total_resid_dist += resid_dist.sum().item()
                total_count += images.size(0)
    
    mean_energy = total_energy / total_count
    mean_combined_dist = (total_disc_dist + lambda_weight * total_resid_dist) / total_count
    
    alpha = abs(mean_energy) / (mean_combined_dist + 1e-8)
    
    print(f"[WDiscOOD Alpha Calibration]")
    print(f"  Mean Energy: {mean_energy:.4f}")
    print(f"  Mean Disc Distance: {total_disc_dist / total_count:.4f}")
    print(f"  Mean Resid Distance: {total_resid_dist / total_count:.4f}")
    print(f"  Alpha = {alpha:.4f}")
    
    return alpha


def evaluate_ood(model, wdiscood, test_loader, near_ood_loader, far_ood_loader, 
                 device, alpha=1.0, lambda_weight=1.0, scoring_mode='hybrid'):
    """
    Evaluate OOD detection performance.
    
    Args:
        model: Feature extractor model
        wdiscood: Fitted WDiscOOD object
        test_loader: ID test data
        near_ood_loader: Near-OOD test data
        far_ood_loader: Far-OOD test data
        device: torch device
        alpha: Balance parameter (used only in hybrid mode)
        lambda_weight: Weight for residual component
        scoring_mode: 'pure' (no Energy) or 'hybrid' (with Energy)
        
    Returns:
        dict: Metrics including accuracy, AUROC, AUPR, FPR95
    """
    model.eval()
    
    def get_scores_and_acc(loader, compute_acc=False):
        all_scores = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                output = model(images)
                if len(output) == 2:
                    logits, features = output
                else:
                    logits, _, features = output
                
                # Compute scores based on mode
                if scoring_mode == 'pure':
                    # Pure WDiscOOD: no Energy term
                    scores, _, _ = wdiscood.compute_scores(features, lambda_weight=lambda_weight)
                else:
                    # Hybrid: WDiscOOD + Energy
                    scores = wdiscood.compute_scores_with_energy(
                        features, logits, alpha=alpha, lambda_weight=lambda_weight
                    )
                all_scores.extend(scores.cpu().numpy())
                
                # Compute accuracy for ID data
                if compute_acc:
                    valid_mask = labels >= 0
                    if valid_mask.any():
                        preds = logits[valid_mask].argmax(dim=1)
                        correct += (preds == labels[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return np.array(all_scores), accuracy
    
    # Get scores
    id_scores, id_acc = get_scores_and_acc(test_loader, compute_acc=True)
    near_ood_scores, _ = get_scores_and_acc(near_ood_loader)
    far_ood_scores, _ = get_scores_and_acc(far_ood_loader)
    
    # Compute metrics
    near_metrics = compute_ood_metrics(id_scores, near_ood_scores)
    far_metrics = compute_ood_metrics(id_scores, far_ood_scores)
    
    return {
        'id_accuracy': id_acc,
        'near_auroc': near_metrics['auroc'],
        'near_aupr': near_metrics['aupr'],
        'near_fpr95': near_metrics['fpr95'],
        'far_auroc': far_metrics['auroc'],
        'far_aupr': far_metrics['aupr'],
        'far_fpr95': far_metrics['fpr95'],
        'id_score_mean': float(np.mean(id_scores)),
        'near_ood_score_mean': float(np.mean(near_ood_scores)),
        'far_ood_score_mean': float(np.mean(far_ood_scores)),
    }


def run_wdiscood_evaluation(args):
    """Main evaluation function."""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("WDiscOOD: Whitened Linear Discriminant Analysis for OOD Detection")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # =====================================================
    # 1. Load Checkpoint
    # =====================================================
    print("\n>>> Step 1: Loading Checkpoint...")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Auto-detect model configuration
    model_type = checkpoint.get('config', {}).get('model_type', 'densenet121')
    checkpoint_image_size = checkpoint.get('config', {}).get('image_size', args.image_size)
    
    if checkpoint_image_size != args.image_size:
        print(f"  [Auto-detect] Overriding image_size: {args.image_size} -> {checkpoint_image_size}")
        args.image_size = checkpoint_image_size
    
    state_dict = checkpoint['global_model_state_dict']
    
    # Detect model architecture from state dict
    if 'heads.head.weight' in state_dict:
        feature_dim = state_dict['heads.head.weight'].shape[1]
        num_classes = state_dict['heads.head.weight'].shape[0]
        hidden_dim = feature_dim
    elif 'classifier.weight' in state_dict:
        feature_dim = state_dict['classifier.weight'].shape[1]
        num_classes = state_dict['classifier.weight'].shape[0]
        hidden_dim = feature_dim
    elif 'classifier.0.weight' in state_dict:
        hidden_dim = state_dict['classifier.0.weight'].shape[0]
        num_classes = state_dict['classifier.3.weight'].shape[0]
        feature_dim = state_dict['classifier.0.weight'].shape[1]
    else:
        raise ValueError("Cannot detect classifier structure from checkpoint")
    
    print(f"  [Auto-detect] model_type={model_type}")
    print(f"  [Auto-detect] feature_dim={feature_dim}, num_classes={num_classes}")
    
    # Initialize model
    backbone = Backbone(model_type=model_type, pretrained=False)
    global_model = FedAvg_Model(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
    global_model.load_state_dict(checkpoint['global_model_state_dict'])
    global_model.to(device)
    global_model.eval()
    
    # =====================================================
    # 2. Load Data
    # =====================================================
    print("\n>>> Step 2: Loading Data...")
    
    # ID train loaders for statistics extraction and alpha calibration
    train_loaders = create_id_train_client_loaders_only(
        args.data_root,
        n_clients=checkpoint.get('config', {}).get('n_clients', 5),
        alpha=checkpoint.get('config', {}).get('alpha', 0.1),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        partition_seed=checkpoint.get('config', {}).get('seed', 42),
    )
    
    # Test loaders
    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        args.data_root, 
        batch_size=args.batch_size, 
        image_size=args.image_size, 
        num_workers=args.num_workers
    )
    
    print("  -> Data loaders ready")
    
    # =====================================================
    # 3. Extract Class Statistics (Federated-Compatible)
    # =====================================================
    print("\n>>> Step 3: Extracting Class Statistics...")
    
    class_sums, class_sum_sq, class_counts = extract_class_statistics(
        global_model, train_loaders, device, num_classes
    )
    
    print(f"  Class sample distribution: min={class_counts.min().item():.0f}, max={class_counts.max().item():.0f}")
    print(f"  Total samples: {class_counts.sum().item():.0f}")
    
    # =====================================================
    # 4. Compute WLDA Statistics
    # =====================================================
    print("\n>>> Step 4: Computing WLDA Statistics...")
    
    class_means, within_class_cov = compute_wlda_statistics(class_sums, class_sum_sq, class_counts)
    
    # =====================================================
    # 5. Fit WDiscOOD
    # =====================================================
    print("\n>>> Step 5: Fitting WDiscOOD...")
    
    wdiscood = WDiscOOD(device=device, regularization=args.regularization)
    wdiscood.fit_from_statistics(class_means, class_counts, within_class_cov)
    
    # =====================================================
    # 6. Calibrate Alpha (only needed for hybrid mode)
    # =====================================================
    if args.scoring_mode == 'hybrid':
        print("\n>>> Step 6: Calibrating Alpha (hybrid mode)...")
        alpha = calibrate_alpha(
            global_model, train_loaders, wdiscood, device, 
            lambda_weight=args.lambda_weight
        )
    else:
        print("\n>>> Step 6: Skipping Alpha calibration (pure mode, no Energy)")
        alpha = 0.0  # Not used in pure mode
    
    # =====================================================
    # 7. Evaluate OOD Detection
    # =====================================================
    print(f"\n>>> Step 7: Evaluating OOD Detection ({args.scoring_mode} mode)...")
    
    metrics = evaluate_ood(
        global_model, wdiscood, 
        test_loader, near_ood_loader, far_ood_loader,
        device, alpha=alpha, lambda_weight=args.lambda_weight,
        scoring_mode=args.scoring_mode
    )
    
    # Determine method name based on scoring mode
    if args.scoring_mode == 'pure':
        method_name = "WDiscOOD"
        method_description = "Pure WLDA-based OOD detection (no Energy)"
        score_formula = "-(disc_dist + λ * resid_dist)"
    else:
        method_name = "WDiscOOD-Energy"
        method_description = "WLDA + Energy hybrid"
        score_formula = "Energy - α * (disc_dist + λ * resid_dist)"
    
    # =====================================================
    # 8. Print Results
    # =====================================================
    print("\n" + "="*60)
    print(f"{method_name} Results")
    print("="*60)
    print(f"Scoring Mode:    {args.scoring_mode}")
    print(f"Score Formula:   {score_formula}")
    print(f"ID Accuracy:     {metrics['id_accuracy']:.4f}")
    print(f"\n--- Near-OOD ---")
    print(f"AUROC:           {metrics['near_auroc']:.4f}")
    print(f"AUPR:            {metrics['near_aupr']:.4f}")
    print(f"FPR95:           {metrics['near_fpr95']:.4f}")
    print(f"\n--- Far-OOD ---")
    print(f"AUROC:           {metrics['far_auroc']:.4f}")
    print(f"AUPR:            {metrics['far_aupr']:.4f}")
    print(f"FPR95:           {metrics['far_fpr95']:.4f}")
    print(f"\n--- Score Statistics ---")
    print(f"ID Score Mean:       {metrics['id_score_mean']:.4f}")
    print(f"Near-OOD Score Mean: {metrics['near_ood_score_mean']:.4f}")
    print(f"Far-OOD Score Mean:  {metrics['far_ood_score_mean']:.4f}")
    
    # =====================================================
    # 9. Save Results
    # =====================================================
    print("\n>>> Step 8: Saving Results...")
    
    save_dir = os.path.dirname(args.checkpoint) or '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method_name,
        "description": method_description,
        "scoring_mode": args.scoring_mode,
        "score_formula": score_formula,
        "checkpoint": args.checkpoint,
        "config": {
            "model_type": model_type,
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "discriminative_dim": int(wdiscood.W_disc.shape[1]),
            "residual_dim": int(wdiscood.W_resid.shape[1]),
            "lambda_weight": float(args.lambda_weight),
            "alpha": float(alpha) if args.scoring_mode == 'hybrid' else None,
            "regularization": float(args.regularization),
        },
        "performance": {
            "id_accuracy": float(metrics['id_accuracy']),
            "near_auroc": float(metrics['near_auroc']),
            "near_aupr": float(metrics['near_aupr']),
            "near_fpr95": float(metrics['near_fpr95']),
            "far_auroc": float(metrics['far_auroc']),
            "far_aupr": float(metrics['far_aupr']),
            "far_fpr95": float(metrics['far_fpr95']),
        },
        "score_statistics": {
            "id_score_mean": float(metrics['id_score_mean']),
            "near_ood_score_mean": float(metrics['near_ood_score_mean']),
            "far_ood_score_mean": float(metrics['far_ood_score_mean']),
        },
        "discriminative_eigenvalues_top10": wdiscood.disc_eigenvalues[:10].tolist() if wdiscood.disc_eigenvalues is not None else []
    }
    
    # Use different filename based on mode
    mode_suffix = "pure" if args.scoring_mode == 'pure' else "energy"
    json_path = os.path.join(save_dir, f"wdiscood_{mode_suffix}_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result_record, f, indent=4)
    print(f"  -> Results saved to: {json_path}")
    
    # Save WDiscOOD state
    state_path = os.path.join(save_dir, f"wdiscood_state_{timestamp}.pth")
    torch.save({
        'W_disc': wdiscood.W_disc.cpu(),
        'W_resid': wdiscood.W_resid.cpu(),
        'class_means': wdiscood.class_means.cpu(),
        'global_mean': wdiscood.global_mean.cpu(),
        'class_means_disc': wdiscood.class_means_disc.cpu(),
        'alpha': alpha,
        'lambda_weight': args.lambda_weight,
        'n_classes': wdiscood.n_classes,
    }, state_path)
    print(f"  -> WDiscOOD state saved to: {state_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WDiscOOD: Whitened LDA for OOD Detection (ICCV 2023)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scoring Modes:
  pure   - Pure WDiscOOD from the original paper
           Score = -(disc_dist + λ * resid_dist)
           Uses only WLDA geometric distances, no Energy term.
           
  hybrid - WDiscOOD-Energy hybrid (default)
           Score = Energy - α * (disc_dist + λ * resid_dist)
           Combines WLDA geometry with classifier confidence.

Examples:
  # Pure WDiscOOD (original paper method)
  python evaluate_wdiscood.py --checkpoint ./path/to/model.pth --scoring_mode pure
  
  # WDiscOOD-Energy hybrid (default)
  python evaluate_wdiscood.py --checkpoint ./path/to/model.pth --scoring_mode hybrid
  
  # With custom parameters
  python evaluate_wdiscood.py --checkpoint ./path/to/model.pth --lambda_weight 0.5
"""
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                        help='Path to dataset root')
    parser.add_argument('--scoring_mode', type=str, default='hybrid', choices=['pure', 'hybrid'],
                        help='Scoring mode: pure (no Energy) or hybrid (with Energy, default)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--lambda_weight', type=float, default=1.0,
                        help='Weight for residual component (λ in score = disc + λ*resid)')
    parser.add_argument('--regularization', type=float, default=1e-5,
                        help='Regularization for within-class covariance')
    
    args = parser.parse_args()
    run_wdiscood_evaluation(args)
