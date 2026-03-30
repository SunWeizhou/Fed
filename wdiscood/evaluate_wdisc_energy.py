#!/usr/bin/env python3
"""
WDisc-Energy: Combining WLDA Residual Space with Logit Energy for OOD Detection

This method addresses the redundancy issue when combining WDiscOOD with ViM:
- WDiscOOD's discriminative subspace (WD) overlaps functionally with the classifier's Logit layer
- Both capture "class-relevant" information, leading to redundancy if used together

Solution: WDisc-Energy
- Class-relevant information → Logit Energy (classifier already learned this)
- Class-agnostic information → WLDA Residual Space (purer than PCA residual)
- Discard WD space to avoid double-counting discriminative information

Mathematical Formulation:
1. WLDA decomposes features into:
   - Discriminative subspace (WD): maximizes between-class / within-class scatter
   - Residual subspace (WDR): orthogonal complement, captures class-agnostic patterns
   
2. Score = Energy - γ * WDR_Residual
   Where:
   - Energy = log(Σ exp(logits))  [from classifier]
   - WDR_Residual = ||W_resid^T (z - μ)||  [from WLDA]
   - γ = calibration factor to align scales

Key Insight:
- PCA residual: orthogonal to max-variance directions (may include discriminative info)
- WLDA residual: orthogonal to discriminative directions (purer class-agnostic signal)

Author: Claude Code
Date: 2025-03-18
Reference: Combines ideas from ViM (NeurIPS 2022) and WDiscOOD (ICCV 2023)
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


class WDiscEnergy:
    """
    WDisc-Energy: WLDA Residual + Logit Energy for OOD Detection.
    
    This method combines:
    1. WLDA residual subspace (class-agnostic geometric information)
    2. Logit Energy (class-relevant classifier confidence)
    
    Unlike WDiscOOD which uses both WD and WDR spaces, we only use WDR
    because the classifier's logits already capture discriminative information.
    """
    
    def __init__(self, device='cpu', regularization=1e-5):
        self.device = device
        self.regularization = regularization
        
        # Only need residual subspace (discard discriminative subspace)
        self.W_resid = None     # Residual subspace projection [D, D-C+1]
        
        # Statistics
        self.global_mean = None  # [D]
        self.resid_mean = None   # Mean in residual space for centering [D-C+1]
        self.n_classes = None
        
        # Eigenvalues for diagnostics
        self.disc_eigenvalues = None
        
    def fit_from_statistics(self, class_means, class_counts, within_class_cov):
        """
        Fit WLDA from aggregated statistics (federated learning compatible).
        Only extracts the residual subspace (discards discriminative subspace).
        
        Args:
            class_means: [C, D] tensor of per-class mean vectors
            class_counts: [C] tensor of per-class sample counts
            within_class_cov: [D, D] tensor of pooled within-class covariance
        """
        class_means = class_means.to(self.device)
        class_counts = class_counts.to(self.device)
        within_class_cov = within_class_cov.to(self.device)
        
        self.n_classes = class_means.shape[0]
        feature_dim = class_means.shape[1]
        total_samples = class_counts.sum().item()
        
        print(f"[WDisc-Energy] Fitting from statistics: {self.n_classes} classes, dim={feature_dim}")
        
        # Global mean (weighted average of class means)
        self.global_mean = (class_means.T @ class_counts) / total_samples
        
        # Within-class scatter
        S_w = within_class_cov * total_samples
        
        # Between-class scatter
        centered_means = class_means - self.global_mean
        S_b = (centered_means.T * class_counts) @ centered_means
        
        # Fit WLDA and extract ONLY the residual subspace
        self._fit_wlda_residual_only(S_w, S_b, total_samples, feature_dim)
    
    def _fit_wlda_residual_only(self, S_w, S_b, n_samples, feature_dim):
        """
        Core WLDA fitting - extract only the residual subspace.
        """
        # Step 1: Regularize within-class scatter
        S_w_reg = S_w + self.regularization * torch.eye(feature_dim, device=self.device) * torch.trace(S_w)
        
        # Step 2: Compute S_w^{-1/2} via eigendecomposition
        try:
            eig_vals_w, eig_vecs_w = torch.linalg.eigh(S_w_reg)
        except RuntimeError as e:
            print(f"[WDisc-Energy] Warning: eigh failed, using SVD fallback: {e}")
            U, S, Vh = torch.linalg.svd(S_w_reg)
            eig_vals_w = S
            eig_vecs_w = U
        
        eig_vals_w = torch.clamp(eig_vals_w, min=1e-10)
        S_w_inv_sqrt = eig_vecs_w @ torch.diag(1.0 / torch.sqrt(eig_vals_w)) @ eig_vecs_w.T
        
        # Step 3: Whitened between-class scatter
        S_b_whitened = S_w_inv_sqrt @ S_b @ S_w_inv_sqrt
        
        # Step 4: Eigendecomposition
        try:
            eig_vals_b, eig_vecs_b = torch.linalg.eigh(S_b_whitened)
        except RuntimeError as e:
            print(f"[WDisc-Energy] Warning: eigh failed, using SVD: {e}")
            U, S, Vh = torch.linalg.svd(S_b_whitened)
            eig_vals_b = S
            eig_vecs_b = U
        
        # Sort by eigenvalue (descending)
        sorted_indices = torch.argsort(eig_vals_b, descending=True)
        eig_vals_b = eig_vals_b[sorted_indices]
        eig_vecs_b = eig_vecs_b[:, sorted_indices]
        
        self.disc_eigenvalues = eig_vals_b.cpu().numpy()
        
        # Step 5: Determine discriminative dimension
        disc_dim = min(self.n_classes - 1, feature_dim)
        eig_threshold = 1e-6 * eig_vals_b[0] if eig_vals_b[0] > 0 else 1e-6
        effective_dim = (eig_vals_b > eig_threshold).sum().item()
        disc_dim = min(disc_dim, effective_dim)
        
        print(f"[WDisc-Energy] Discriminative dimension: {disc_dim} (discarded)")
        
        # Step 6: Extract ONLY residual subspace (the key difference from WDiscOOD)
        # Residual = directions with smallest eigenvalues (class-agnostic)
        resid_dim = feature_dim - disc_dim
        if resid_dim > 0:
            self.W_resid = S_w_inv_sqrt @ eig_vecs_b[:, disc_dim:]  # [D, resid_dim]
        else:
            self.W_resid = torch.zeros(feature_dim, 1, device=self.device)
        
        print(f"[WDisc-Energy] Residual dimension: {self.W_resid.shape[1]} (used for OOD)")
        print(f"[WDisc-Energy] Note: Discriminative info will come from Logit Energy")
        
    def compute_residual(self, features):
        """
        Compute WLDA residual norm for features.
        
        Args:
            features: [N, D] tensor of feature vectors
            
        Returns:
            residual: [N] tensor of residual norms
        """
        features = features.to(self.device)
        
        # Center features
        centered = features - self.global_mean  # [N, D]
        
        # Project to residual subspace
        z_resid = centered @ self.W_resid  # [N, resid_dim]
        
        # L2 norm in residual subspace
        residual = torch.norm(z_resid, p=2, dim=1)  # [N]
        
        return residual
    
    def compute_scores(self, features, logits, gamma=1.0):
        """
        Compute WDisc-Energy scores.
        
        Score = Energy - γ * WDR_Residual
        
        Where:
        - Energy = logsumexp(logits) → higher = more confident = more ID
        - WDR_Residual = ||W_resid^T (z - μ)|| → higher = more anomalous
        - γ calibrates the scale
        
        Higher score → more likely ID
        
        Args:
            features: [N, D] feature vectors
            logits: [N, C] classifier logits
            gamma: scale factor for residual term
            
        Returns:
            scores: [N] OOD scores (higher = more ID)
        """
        # Energy from classifier (class-relevant)
        energy = torch.logsumexp(logits, dim=1)  # [N]
        
        # WLDA residual (class-agnostic)
        residual = self.compute_residual(features)  # [N]
        
        # Combined score: Energy - γ * Residual
        # Higher energy → more ID
        # Higher residual → more OOD
        scores = energy - gamma * residual
        
        return scores, energy, residual


def extract_class_statistics(model, data_loaders, device, num_classes):
    """
    Extract per-class statistics from data loaders.
    (Same as WDiscOOD implementation)
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
                
                output = model(images)
                if len(output) == 2:
                    logits, features = output
                else:
                    logits, _, features = output
                
                if feature_dim is None:
                    feature_dim = features.shape[1]
                    class_sums = torch.zeros(num_classes, feature_dim, device=device)
                    class_sum_sq = torch.zeros(num_classes, feature_dim, feature_dim, device=device)
                    class_counts = torch.zeros(num_classes, device=device)
                
                for c in range(num_classes):
                    mask = labels == c
                    if mask.sum() > 0:
                        class_features = features[mask]
                        class_sums[c] += class_features.sum(dim=0)
                        class_sum_sq[c] += class_features.T @ class_features
                        class_counts[c] += mask.sum()
    
    return class_sums, class_sum_sq, class_counts


def compute_wlda_statistics(class_sums, class_sum_sq, class_counts):
    """
    Compute WLDA statistics from aggregated per-class statistics.
    (Same as WDiscOOD implementation)
    """
    n_classes, feature_dim = class_sums.shape
    device = class_sums.device
    
    class_means = torch.zeros(n_classes, feature_dim, device=device)
    for c in range(n_classes):
        if class_counts[c] > 0:
            class_means[c] = class_sums[c] / class_counts[c]
    
    within_class_cov = torch.zeros(feature_dim, feature_dim, device=device)
    total_samples = class_counts.sum().item()
    
    for c in range(n_classes):
        if class_counts[c] > 0:
            class_cov = class_sum_sq[c] / class_counts[c] - torch.outer(class_means[c], class_means[c])
            within_class_cov += class_cov * class_counts[c]
    
    within_class_cov = within_class_cov / total_samples
    
    return class_means, within_class_cov


def calibrate_gamma(model, data_loaders, wdisc_energy, device):
    """
    Calibrate gamma parameter using ID training data.
    
    γ = |mean(energy)| / mean(residual)
    
    This aligns the scales of Energy and Residual components.
    Same calibration logic as ViM's alpha.
    """
    model.eval()
    
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = [data_loaders]
    
    total_energy = 0.0
    total_residual = 0.0
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
                
                # WLDA Residual
                residual = wdisc_energy.compute_residual(features)
                total_residual += residual.sum().item()
                total_count += images.size(0)
    
    mean_energy = total_energy / total_count
    mean_residual = total_residual / total_count
    
    gamma = abs(mean_energy) / (mean_residual + 1e-8)
    
    print(f"[WDisc-Energy Gamma Calibration]")
    print(f"  Mean Energy (from Logits): {mean_energy:.4f}")
    print(f"  Mean WLDA Residual: {mean_residual:.4f}")
    print(f"  Gamma = {gamma:.4f}")
    
    return gamma, mean_energy, mean_residual


def evaluate_ood(model, wdisc_energy, test_loader, near_ood_loader, far_ood_loader, 
                 device, gamma=1.0):
    """
    Evaluate OOD detection performance.
    """
    model.eval()
    
    def get_scores_and_acc(loader, compute_acc=False):
        all_scores = []
        all_energies = []
        all_residuals = []
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
                
                # Compute WDisc-Energy scores
                scores, energy, residual = wdisc_energy.compute_scores(features, logits, gamma=gamma)
                all_scores.extend(scores.cpu().numpy())
                all_energies.extend(energy.cpu().numpy())
                all_residuals.extend(residual.cpu().numpy())
                
                # Compute accuracy for ID data
                if compute_acc:
                    valid_mask = labels >= 0
                    if valid_mask.any():
                        preds = logits[valid_mask].argmax(dim=1)
                        correct += (preds == labels[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return np.array(all_scores), np.array(all_energies), np.array(all_residuals), accuracy
    
    # Get scores
    id_scores, id_energy, id_resid, id_acc = get_scores_and_acc(test_loader, compute_acc=True)
    near_scores, near_energy, near_resid, _ = get_scores_and_acc(near_ood_loader)
    far_scores, far_energy, far_resid, _ = get_scores_and_acc(far_ood_loader)
    
    # Compute metrics
    near_metrics = compute_ood_metrics(id_scores, near_scores)
    far_metrics = compute_ood_metrics(id_scores, far_scores)
    
    return {
        'id_accuracy': id_acc,
        'near_auroc': near_metrics['auroc'],
        'near_aupr': near_metrics['aupr'],
        'near_fpr95': near_metrics['fpr95'],
        'far_auroc': far_metrics['auroc'],
        'far_aupr': far_metrics['aupr'],
        'far_fpr95': far_metrics['fpr95'],
        # Score statistics
        'id_score_mean': float(np.mean(id_scores)),
        'near_ood_score_mean': float(np.mean(near_scores)),
        'far_ood_score_mean': float(np.mean(far_scores)),
        # Component statistics for analysis
        'id_energy_mean': float(np.mean(id_energy)),
        'id_resid_mean': float(np.mean(id_resid)),
        'near_energy_mean': float(np.mean(near_energy)),
        'near_resid_mean': float(np.mean(near_resid)),
        'far_energy_mean': float(np.mean(far_energy)),
        'far_resid_mean': float(np.mean(far_resid)),
    }


def run_wdisc_energy_evaluation(args):
    """Main evaluation function."""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("WDisc-Energy: WLDA Residual + Logit Energy for OOD Detection")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"\nKey Insight:")
    print(f"  - Class-relevant info → Logit Energy (classifier already learned)")
    print(f"  - Class-agnostic info → WLDA Residual (purer than PCA residual)")
    print(f"  - WD space discarded to avoid redundancy with Logits")
    
    # =====================================================
    # 1. Load Checkpoint
    # =====================================================
    print("\n>>> Step 1: Loading Checkpoint...")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    model_type = checkpoint.get('config', {}).get('model_type', 'densenet121')
    checkpoint_image_size = checkpoint.get('config', {}).get('image_size', args.image_size)
    
    if checkpoint_image_size != args.image_size:
        print(f"  [Auto-detect] Overriding image_size: {args.image_size} -> {checkpoint_image_size}")
        args.image_size = checkpoint_image_size
    
    state_dict = checkpoint['global_model_state_dict']
    
    # Detect model architecture
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
    
    train_loaders = create_id_train_client_loaders_only(
        args.data_root,
        n_clients=checkpoint.get('config', {}).get('n_clients', 5),
        alpha=checkpoint.get('config', {}).get('alpha', 0.1),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        partition_seed=checkpoint.get('config', {}).get('seed', 42),
    )
    
    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        args.data_root, 
        batch_size=args.batch_size, 
        image_size=args.image_size, 
        num_workers=args.num_workers
    )
    
    print("  -> Data loaders ready")
    
    # =====================================================
    # 3. Extract Class Statistics
    # =====================================================
    print("\n>>> Step 3: Extracting Class Statistics...")
    
    class_sums, class_sum_sq, class_counts = extract_class_statistics(
        global_model, train_loaders, device, num_classes
    )
    
    print(f"  Total samples: {class_counts.sum().item():.0f}")
    
    # =====================================================
    # 4. Compute WLDA Statistics
    # =====================================================
    print("\n>>> Step 4: Computing WLDA Statistics...")
    
    class_means, within_class_cov = compute_wlda_statistics(class_sums, class_sum_sq, class_counts)
    
    # =====================================================
    # 5. Fit WDisc-Energy (WLDA Residual Only)
    # =====================================================
    print("\n>>> Step 5: Fitting WDisc-Energy...")
    
    wdisc_energy = WDiscEnergy(device=device, regularization=args.regularization)
    wdisc_energy.fit_from_statistics(class_means, class_counts, within_class_cov)
    
    # =====================================================
    # 6. Calibrate Gamma
    # =====================================================
    print("\n>>> Step 6: Calibrating Gamma...")
    
    gamma, mean_energy, mean_residual = calibrate_gamma(
        global_model, train_loaders, wdisc_energy, device
    )
    
    # =====================================================
    # 7. Evaluate OOD Detection
    # =====================================================
    print("\n>>> Step 7: Evaluating OOD Detection...")
    
    metrics = evaluate_ood(
        global_model, wdisc_energy, 
        test_loader, near_ood_loader, far_ood_loader,
        device, gamma=gamma
    )
    
    # =====================================================
    # 8. Print Results
    # =====================================================
    print("\n" + "="*70)
    print("WDisc-Energy Results")
    print("="*70)
    print(f"ID Accuracy:     {metrics['id_accuracy']:.4f}")
    print(f"\n--- Near-OOD ---")
    print(f"AUROC:           {metrics['near_auroc']:.4f}")
    print(f"AUPR:            {metrics['near_aupr']:.4f}")
    print(f"FPR95:           {metrics['near_fpr95']:.4f}")
    print(f"\n--- Far-OOD ---")
    print(f"AUROC:           {metrics['far_auroc']:.4f}")
    print(f"AUPR:            {metrics['far_aupr']:.4f}")
    print(f"FPR95:           {metrics['far_fpr95']:.4f}")
    
    print(f"\n--- Component Analysis ---")
    print(f"{'':20} {'Energy':>12} {'WLDA Resid':>12} {'Score':>12}")
    print(f"{'ID (test)':20} {metrics['id_energy_mean']:>12.4f} {metrics['id_resid_mean']:>12.4f} {metrics['id_score_mean']:>12.4f}")
    print(f"{'Near-OOD':20} {metrics['near_energy_mean']:>12.4f} {metrics['near_resid_mean']:>12.4f} {metrics['near_ood_score_mean']:>12.4f}")
    print(f"{'Far-OOD':20} {metrics['far_energy_mean']:>12.4f} {metrics['far_resid_mean']:>12.4f} {metrics['far_ood_score_mean']:>12.4f}")
    
    # =====================================================
    # 9. Save Results
    # =====================================================
    print("\n>>> Step 8: Saving Results...")
    
    save_dir = os.path.dirname(args.checkpoint) or '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "WDisc-Energy",
        "description": "WLDA Residual + Logit Energy (no WD space to avoid redundancy)",
        "checkpoint": args.checkpoint,
        "config": {
            "model_type": model_type,
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "residual_dim": int(wdisc_energy.W_resid.shape[1]),
            "gamma": float(gamma),
            "regularization": float(args.regularization),
        },
        "calibration": {
            "mean_energy": float(mean_energy),
            "mean_residual": float(mean_residual),
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
        "component_analysis": {
            "id_energy_mean": float(metrics['id_energy_mean']),
            "id_resid_mean": float(metrics['id_resid_mean']),
            "near_energy_mean": float(metrics['near_energy_mean']),
            "near_resid_mean": float(metrics['near_resid_mean']),
            "far_energy_mean": float(metrics['far_energy_mean']),
            "far_resid_mean": float(metrics['far_resid_mean']),
        },
    }
    
    json_path = os.path.join(save_dir, f"wdisc_energy_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result_record, f, indent=4)
    print(f"  -> Results saved to: {json_path}")
    
    # Save state
    state_path = os.path.join(save_dir, f"wdisc_energy_state_{timestamp}.pth")
    torch.save({
        'W_resid': wdisc_energy.W_resid.cpu(),
        'global_mean': wdisc_energy.global_mean.cpu(),
        'gamma': gamma,
        'n_classes': wdisc_energy.n_classes,
    }, state_path)
    print(f"  -> State saved to: {state_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WDisc-Energy: WLDA Residual + Logit Energy for OOD Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This method combines the best of ViM and WDiscOOD:
- Uses Logit Energy for class-relevant information (like ViM)
- Uses WLDA Residual for class-agnostic information (purer than PCA)
- Avoids redundancy by discarding WDiscOOD's discriminative subspace

Examples:
  python evaluate_wdisc_energy.py --checkpoint ./experiments_v6/densenet121/best_model.pth
  
  CUDA_VISIBLE_DEVICES=1 python evaluate_wdisc_energy.py --checkpoint ./path/to/model.pth
"""
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                        help='Path to dataset root')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--regularization', type=float, default=1e-5,
                        help='Regularization for within-class covariance')
    
    args = parser.parse_args()
    run_wdisc_energy_evaluation(args)
