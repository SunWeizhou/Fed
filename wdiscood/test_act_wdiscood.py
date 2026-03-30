#!/usr/bin/env python3
"""
Test Fisher-spectrum dimension selection for WDiscOOD.

Historical filename retained for continuity, but the previous ACT-on-correlation
branch has been replaced by Fisher-Wachter thresholding following:
  Wang & Yao (2017), "Extreme eigenvalues of large-dimensional spiked Fisher
  matrices with application".

Try:
  - Fixed: min(C-1, effective_dim)
  - Wachter: count Fisher roots above the right edge of the Wachter bulk
  - Original: fisher_ratio > 0

For WLDA, the generalized eigenvalues of S_w^{-1} S_b are Fisher roots.
We model them with MANOVA-style degrees of freedom:
  - between-class df = C - 1
  - within-class df = N - C
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models
from data_utils import create_id_train_loader_only, create_test_loaders_only
from models import Backbone, FedAvg_Model


class FisherWachterSelector:
    """Dimension selection by the Wachter bulk edge of a Fisher matrix."""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def get_optimal_k(self, fisher_eigs, n_samples, n_classes):
        """
        Estimate the number of signal dimensions by counting generalized Fisher
        eigenvalues outside the Wachter bulk.

        Wang & Yao (2017) study S = S2^{-1} S1 where:
          S1 = H / m,  m = C - 1     (between-class covariance estimate)
          S2 = E / n,  n = N - C     (within-class covariance estimate)

        Our WLDA eigenvalues are roots of E^{-1} H, so they differ from the
        paper's Fisher matrix eigenvalues by a factor m / n:
          S = (n / m) E^{-1} H
          lambda_raw = (m / n) lambda_fisher

        Therefore the threshold applied to WLDA Fisher roots is:
          beta_raw = (beta_fisher + d_n) * (m / n)

        Args:
            fisher_eigs: Descending eigenvalues of S_w^{-1} S_b
            n_samples: Total number of ID training samples
            n_classes: Number of ID classes

        Returns:
            optimal_k: Estimated number of signal dimensions
            diagnostics: Dict with Wachter parameters and threshold
        """
        p = int(fisher_eigs.numel())
        between_df = max(int(n_classes - 1), 1)
        within_df = max(int(n_samples - n_classes), 1)

        c = p / between_df
        raw_y = p / within_df
        if raw_y >= 1.0:
            print(
                f"[Wachter] Warning: y=p/n={raw_y:.4f} >= 1, "
                "clipping to 1-1e-6 to stay within Wachter support formula."
            )
        y = min(raw_y, 1.0 - 1e-6)

        sqrt_term = np.sqrt(c + y - c * y)
        beta_fisher = ((1.0 + sqrt_term) / max(1.0 - y, self.eps)) ** 2
        d_n = np.log(max(p, 2)) / (p ** (2.0 / 3.0))
        beta_fisher_corrected = beta_fisher + d_n
        scale_m_over_n = between_df / within_df
        beta_raw = beta_fisher_corrected * scale_m_over_n

        valid_indices = torch.where(fisher_eigs > beta_raw)[0]
        optimal_k = int(valid_indices[-1].item() + 1) if len(valid_indices) > 0 else 1

        diagnostics = {
            'feature_dim': p,
            'between_df': between_df,
            'within_df': within_df,
            'c': float(c),
            'y_raw': float(raw_y),
            'y_used': float(y),
            'beta_fisher_asymptotic': float(beta_fisher),
            'beta_fisher_corrected': float(beta_fisher_corrected),
            'beta_raw': float(beta_raw),
            'scale_m_over_n': float(scale_m_over_n),
            'd_n_penalty': float(d_n),
        }

        print(
            f"[Wachter] p={p}, between_df={between_df}, within_df={within_df}, "
            f"c={c:.4f}, y={y:.4f}"
        )
        print(
            f"[Wachter] beta_fisher(asymptotic)={beta_fisher:.6f}, "
            f"beta_fisher(corrected)={beta_fisher_corrected:.6f}, "
            f"beta_raw(E^-1H scale)={beta_raw:.6f}"
        )
        print(f"[Wachter] Top 10 Fisher eigs: {fisher_eigs[:10].detach().cpu().numpy()}")
        print(f"[Wachter] Found {optimal_k} eigenvalues above the right edge")

        return optimal_k, diagnostics


class WDiscOOD_ACT:
    """WDiscOOD with Fisher-Wachter-based dimension selection."""
    
    def __init__(self, device='cuda', dim_strategy='wachter'):
        self.device = device
        self.dim_strategy = dim_strategy
        self.wachter_selector = FisherWachterSelector()
        
        # To be computed
        self.W_disc = None
        self.W_resid = None
        self.class_means_disc = None
        self.global_mean = None
        self.n_classes = None
        self.disc_dim = None
        self.resid_dim = None
        self.fisher_ratios = None
        self.wachter_info = None
        self.whitener = None
        self.global_mean_resid = None
    
    def fit(self, features, labels):
        """Fit WLDA with Fisher-Wachter-based dimension selection."""
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
        
        # Match evaluate_wdiscood.py pure mode so only dimension selection changes.
        S_w_reg = S_w + 1e-5 * torch.eye(feature_dim, device=self.device) * torch.trace(S_w)

        eigvals_w, eigvecs_w = torch.linalg.eigh(S_w_reg)
        eigvals_w = torch.clamp(eigvals_w, min=1e-10)
        self.whitener = eigvecs_w @ torch.diag(1.0 / torch.sqrt(eigvals_w)) @ eigvecs_w.T

        centered_means = class_means - self.global_mean
        S_b = (centered_means.T * class_counts) @ centered_means
        S_b_whitened = self.whitener @ S_b @ self.whitener
        
        # Eigendecomposition
        eigvals_b, eigvecs_b = torch.linalg.eigh(S_b_whitened)
        
        # Sort in descending order
        idx = torch.argsort(eigvals_b, descending=True)
        eigvals_b = eigvals_b[idx]
        eigvecs_b = eigvecs_b[:, idx]
        
        self.fisher_ratios = eigvals_b.cpu().numpy()
        
        # Dimension selection
        if self.dim_strategy in {'act', 'wachter'}:
            disc_dim, wachter_info = self.wachter_selector.get_optimal_k(
                eigvals_b,
                n_samples,
                self.n_classes,
            )
            self.wachter_info = wachter_info
            print(f"[Wachter] Selected discriminative dimension: {disc_dim}")
        elif self.dim_strategy == 'fixed':
            max_disc_dim = min(self.n_classes - 1, feature_dim)
            eig_threshold = 1e-6 * eigvals_b[0]
            effective_dim = (eigvals_b > eig_threshold).sum().item()
            disc_dim = min(max_disc_dim, effective_dim)
            print(f"[Fixed] Using min(C-1, effective_dim): {disc_dim} dims")
        elif self.dim_strategy == 'original':
            disc_dim = (eigvals_b > 0).sum().item()
            print(f"[Original] Using fisher_ratio > 0: {disc_dim} dims")
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

        return scores.cpu().numpy()


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


def evaluate_strategy(checkpoint_path, data_root, strategy='wachter', device='cuda'):
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
    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        data_root, batch_size=64, image_size=image_size
    )
    
    # Extract features
    print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, _ = extract_features(model, test_loader, device)
    near_features, _ = extract_features(model, near_ood_loader, device)
    far_features, _ = extract_features(model, far_ood_loader, device)
    
    # Fit WDiscOOD
    wdiscood = WDiscOOD_ACT(device=device, dim_strategy=strategy)
    wdiscood.fit(train_features, train_labels)
    
    # Compute scores
    print("\nComputing scores...")
    id_scores = wdiscood.compute_scores(test_features, lambda_weight=1.0)
    near_scores = wdiscood.compute_scores(near_features, lambda_weight=1.0)
    far_scores = wdiscood.compute_scores(far_features, lambda_weight=1.0)
    
    # Evaluate
    near_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(near_scores))])
    near_all = np.concatenate([id_scores, near_scores])
    near_auroc = roc_auc_score(near_labels, near_all)
    
    far_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(far_scores))])
    far_all = np.concatenate([id_scores, far_scores])
    far_auroc = roc_auc_score(far_labels, far_all)
    
    results = {
        'model_type': model_type,
        'strategy': strategy,
        'disc_dim': wdiscood.disc_dim,
        'resid_dim': wdiscood.resid_dim,
        'near_auroc': float(near_auroc),
        'far_auroc': float(far_auroc),
        'avg_auroc': float((near_auroc + far_auroc) / 2),
    }

    if strategy in {'act', 'wachter'}:
        results['fisher_eigs_top10'] = wdiscood.fisher_ratios[:10].tolist()
        results['wachter'] = wdiscood.wachter_info
    
    print(f"\n{'='*80}")
    print(f"Results for {strategy} strategy:")
    print(f"{'='*80}")
    print(f"Discriminative dim: {wdiscood.disc_dim}")
    print(f"Near-OOD AUROC: {near_auroc*100:.2f}%")
    print(f"Far-OOD AUROC: {far_auroc*100:.2f}%")
    print(f"Average AUROC: {(near_auroc + far_auroc)/2*100:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test Fisher-Wachter-based WDiscOOD dimension selection'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset')
    parser.add_argument('--output', type=str, default='paper_tools/act_wdiscood_comparison.json')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test all three strategies
    results = {}
    
    print("\n" + "="*80)
    print("TESTING DIMENSION SELECTION STRATEGIES FOR WDISCOOD")
    print("="*80)
    
    for strategy in ['fixed', 'wachter', 'original']:
        results[strategy] = evaluate_strategy(
            args.checkpoint,
            args.data_root,
            strategy=strategy,
            device=device
        )
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} │ {'Fixed (C-1)':>15} │ {'Wachter':>15} │ {'Original (>0)':>15}")
    print("-"*80)
    
    metrics = ['disc_dim', 'near_auroc', 'far_auroc', 'avg_auroc']
    for metric in metrics:
        fixed = results['fixed'][metric]
        act = results['wachter'][metric]
        orig = results['original'][metric]
        
        if metric == 'disc_dim':
            print(f"{metric:<25} │ {fixed:>15} │ {act:>15} │ {orig:>15}")
        else:
            print(f"{metric:<25} │ {fixed*100:>14.2f}% │ {act*100:>14.2f}% │ {orig*100:>14.2f}%")
    
    # Determine best
    best_strategy = max(results.items(), key=lambda x: x[1]['avg_auroc'])
    print(f"\n🏆 Best strategy: {best_strategy[0].upper()} (Avg AUROC: {best_strategy[1]['avg_auroc']*100:.2f}%)")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
