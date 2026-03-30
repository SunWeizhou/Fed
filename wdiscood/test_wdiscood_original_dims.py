#!/usr/bin/env python3
"""
Test original WDiscOOD dimension selection strategy.

Compare two strategies:
1. Our implementation: disc_dim = min(C-1, effective_dim) with threshold 1e-6 * λ_max
2. Original paper: Use all discriminants with fisher_ratio > 0

This tests whether the original paper's more flexible approach yields better results.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models
from data_utils import create_id_train_loader_only, create_test_loaders_only
from models import Backbone, FedAvg_Model
from utils.ood_utils import compute_ood_metrics


class WDiscOOD_Original:
    """WDiscOOD with original paper's dimension selection."""
    
    def __init__(self, device='cuda', dim_strategy='original'):
        """
        Args:
            dim_strategy: 'original' (fisher_ratio > 0) or 'fixed' (min(C-1, D))
        """
        self.device = device
        self.dim_strategy = dim_strategy
        
        # To be computed
        self.W_disc = None
        self.W_resid = None
        self.class_means_disc = None
        self.global_mean = None
        self.n_classes = None
        self.disc_dim = None
        self.resid_dim = None
        self.fisher_ratios = None
        self.whitener = None
        self.global_mean_resid = None
    
    def fit(self, features, labels):
        """Fit WLDA with original paper's dimension selection."""
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        n_samples, feature_dim = features.shape
        self.n_classes = int(labels.max().item()) + 1
        
        print(f"Fitting WLDA with {self.n_classes} classes, {feature_dim} features...")
        
        # Global mean
        self.global_mean = features.mean(dim=0)
        
        # Compute class means and within-class scatter from raw features.
        class_means = torch.zeros(self.n_classes, feature_dim, device=self.device)
        class_counts = torch.zeros(self.n_classes, device=self.device)
        S_w = torch.zeros(feature_dim, feature_dim, device=self.device)
        
        for c in range(self.n_classes):
            mask = labels == c
            class_features = features[mask]
            class_mean = class_features.mean(dim=0)
            class_means[c] = class_mean
            class_counts[c] = mask.sum()
            
            # Within-class scatter
            centered = class_features - class_mean
            S_w += centered.T @ centered
        
        # Match evaluate_wdiscood.py pure mode so only dimension strategy changes.
        S_w_reg = S_w + 1e-5 * torch.eye(feature_dim, device=self.device) * torch.trace(S_w)

        eigvals_w, eigvecs_w = torch.linalg.eigh(S_w_reg)
        eigvals_w = torch.clamp(eigvals_w, min=1e-10)
        self.whitener = eigvecs_w @ torch.diag(1.0 / torch.sqrt(eigvals_w)) @ eigvecs_w.T

        centered_means = class_means - self.global_mean
        S_b = (centered_means.T * class_counts) @ centered_means
        S_b_whitened = self.whitener @ S_b @ self.whitener
        eigvals_b, eigvecs_b = torch.linalg.eigh(S_b_whitened)
        
        # Sort in descending order
        idx = torch.argsort(eigvals_b, descending=True)
        eigvals_b = eigvals_b[idx]
        eigvecs_b = eigvecs_b[:, idx]
        
        self.fisher_ratios = eigvals_b.cpu().numpy()
        
        # Dimension selection based on strategy
        if self.dim_strategy == 'original':
            # Original paper: fisher_ratio > 0
            disc_dim = (eigvals_b > 0).sum().item()
            print(f"[Original] Using all discriminants with fisher_ratio > 0: {disc_dim} dims")
        elif self.dim_strategy == 'fixed':
            # Our implementation: min(C-1, effective_dim)
            max_disc_dim = min(self.n_classes - 1, feature_dim)
            eig_threshold = 1e-6 * eigvals_b[0]
            effective_dim = (eigvals_b > eig_threshold).sum().item()
            disc_dim = min(max_disc_dim, effective_dim)
            print(f"[Fixed] Using min(C-1, effective_dim): {disc_dim} dims (max={max_disc_dim}, effective={effective_dim})")
        else:
            raise ValueError(f"Unknown strategy: {self.dim_strategy}")
        
        self.disc_dim = disc_dim
        self.resid_dim = feature_dim - disc_dim

        self.W_disc = self.whitener @ eigvecs_b[:, :disc_dim]

        if self.resid_dim > 0:
            self.W_resid = self.whitener @ eigvecs_b[:, disc_dim:]
        else:
            self.W_resid = torch.zeros(feature_dim, 1, device=self.device)

        class_means_centered = class_means - self.global_mean
        self.class_means_disc = class_means_centered @ self.W_disc
        self.global_mean_resid = None
        
        print(f"Discriminative space: {self.disc_dim} dims")
        print(f"Residual space: {self.resid_dim} dims")
        print(f"Top 10 Fisher ratios: {self.fisher_ratios[:10]}")
        
        # Check if there are positive fisher ratios beyond C-1
        max_theoretical = self.n_classes - 1
        if disc_dim > max_theoretical:
            print(f"⚠️  Found {disc_dim - max_theoretical} discriminants beyond C-1 with positive fisher ratio")
            print(f"   Fisher ratios [{max_theoretical}:{disc_dim}]: {self.fisher_ratios[max_theoretical:disc_dim]}")
    
    def compute_scores(self, features, lambda_weight=1.0):
        """Compute WDiscOOD scores."""
        features = features.to(self.device)

        centered = features - self.global_mean
        z_disc = centered @ self.W_disc
        z_resid = centered @ self.W_resid

        dists_sq = torch.cdist(z_disc, self.class_means_disc, p=2) ** 2
        disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])
        resid_dist = torch.norm(z_resid, p=2, dim=1)

        scores = -(disc_dist + lambda_weight * resid_dist)

        return scores.cpu().numpy(), disc_dist.cpu().numpy(), resid_dist.cpu().numpy()


def extract_features(model, dataloader, device):
    """Extract features from model."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = torch.zeros(images.size(0), dtype=torch.long)
            
            images = images.to(device)
            _, features = model(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def evaluate_wdiscood_strategy(checkpoint_path, data_root, strategy='original', device='cuda'):
    """Evaluate WDiscOOD with different dimension selection strategies."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'densenet169')
    num_classes = config.get('num_classes', 54)
    image_size = config.get('image_size', 320)
    
    print(f"\n{'='*80}")
    print(f"Model: {model_type}, Strategy: {strategy}")
    print(f"{'='*80}\n")
    
    # Match evaluate_wdiscood.py checkpoint loading.
    if 'global_model_state_dict' in checkpoint:
        state_dict = checkpoint['global_model_state_dict']

        if 'heads.head.weight' in state_dict:
            hidden_dim = state_dict['heads.head.weight'].shape[1]
        elif 'classifier.weight' in state_dict:
            hidden_dim = state_dict['classifier.weight'].shape[1]
        elif 'classifier.0.weight' in state_dict:
            hidden_dim = state_dict['classifier.0.weight'].shape[0]
        else:
            raise ValueError("Cannot detect classifier structure from checkpoint")

        backbone = Backbone(model_type=model_type, pretrained=False)
        model = FedAvg_Model(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
        model.load_state_dict(state_dict)
    else:
        model = models.create_model(model_type, num_classes=num_classes)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('server_model_state', {}))
        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace('_orig_mod.', '').replace('model.', '')
            cleaned[key] = v
        model.load_state_dict(cleaned, strict=False)

    model = model.to(device)
    model.eval()
    
    # Get data loaders
    train_loader = create_id_train_loader_only(data_root, batch_size=64, image_size=image_size)
    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(data_root, batch_size=64, image_size=image_size)
    
    # Extract features
    print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    near_features, _ = extract_features(model, near_ood_loader, device)
    far_features, _ = extract_features(model, far_ood_loader, device)
    
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")
    print(f"Near-OOD: {near_features.shape}, Far-OOD: {far_features.shape}")
    
    # Fit WDiscOOD
    wdiscood = WDiscOOD_Original(device=device, dim_strategy=strategy)
    wdiscood.fit(train_features, train_labels)
    
    # Compute scores
    print("\nComputing scores...")
    id_scores, id_disc, id_resid = wdiscood.compute_scores(test_features, lambda_weight=1.0)
    near_scores, near_disc, near_resid = wdiscood.compute_scores(near_features, lambda_weight=1.0)
    far_scores, far_disc, far_resid = wdiscood.compute_scores(far_features, lambda_weight=1.0)
    
    # Evaluate
    print("\nEvaluating...")
    
    # Near-OOD
    near_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(near_scores))])
    near_all_scores = np.concatenate([id_scores, near_scores])
    near_auroc = roc_auc_score(near_labels, near_all_scores)
    near_aupr = average_precision_score(near_labels, near_all_scores)
    
    # Far-OOD
    far_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(far_scores))])
    far_all_scores = np.concatenate([id_scores, far_scores])
    far_auroc = roc_auc_score(far_labels, far_all_scores)
    far_aupr = average_precision_score(far_labels, far_all_scores)
    
    results = {
        'model_type': model_type,
        'strategy': strategy,
        'disc_dim': wdiscood.disc_dim,
        'resid_dim': wdiscood.resid_dim,
        'fisher_ratios_top10': wdiscood.fisher_ratios[:10].tolist(),
        'near_auroc': float(near_auroc),
        'near_aupr': float(near_aupr),
        'far_auroc': float(far_auroc),
        'far_aupr': float(far_aupr),
        'avg_auroc': float((near_auroc + far_auroc) / 2),
    }
    
    print(f"\n{'='*80}")
    print(f"Results for {strategy} strategy:")
    print(f"{'='*80}")
    print(f"Discriminative dim: {wdiscood.disc_dim}")
    print(f"Residual dim: {wdiscood.resid_dim}")
    print(f"Near-OOD AUROC: {near_auroc*100:.2f}%")
    print(f"Far-OOD AUROC: {far_auroc*100:.2f}%")
    print(f"Average AUROC: {(near_auroc + far_auroc)/2*100:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test WDiscOOD dimension selection strategies')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset')
    parser.add_argument('--output', type=str, default='paper_tools/wdiscood_dim_strategy_comparison.json')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test both strategies
    results = {}
    
    print("\n" + "="*80)
    print("TESTING DIMENSION SELECTION STRATEGIES")
    print("="*80)
    
    for strategy in ['fixed', 'original']:
        results[strategy] = evaluate_wdiscood_strategy(
            args.checkpoint, 
            args.data_root, 
            strategy=strategy,
            device=device
        )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} │ {'Fixed (Ours)':>15} │ {'Original Paper':>15} │ {'Difference':>12}")
    print("-"*80)
    
    metrics = ['disc_dim', 'near_auroc', 'far_auroc', 'avg_auroc']
    for metric in metrics:
        fixed_val = results['fixed'][metric]
        orig_val = results['original'][metric]
        diff = orig_val - fixed_val
        
        if metric == 'disc_dim':
            print(f"{metric:<25} │ {fixed_val:>15} │ {orig_val:>15} │ {diff:>+12}")
        else:
            print(f"{metric:<25} │ {fixed_val*100:>14.2f}% │ {orig_val*100:>14.2f}% │ {diff*100:>+11.2f}%")
    
    # Check if there are dims beyond C-1
    fixed_dim = results['fixed']['disc_dim']
    orig_dim = results['original']['disc_dim']
    n_classes = results['fixed']['model_type']  # Placeholder
    
    if orig_dim > fixed_dim:
        print(f"\n⚠️  Original strategy uses {orig_dim - fixed_dim} more dimensions than fixed strategy!")
        print(f"   These are discriminants beyond C-1 with small positive Fisher ratios.")
    elif orig_dim < fixed_dim:
        print(f"\n⚠️  Original strategy uses {fixed_dim - orig_dim} fewer dimensions!")
        print(f"   Some discriminants have Fisher ratio ≤ 0.")
    else:
        print(f"\n✓ Both strategies use the same {orig_dim} dimensions.")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
