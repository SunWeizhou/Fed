#!/usr/bin/env python3
"""
Grid search for optimal λ (lambda) in Pure WDiscOOD.

Pure WDiscOOD Score = -(disc_dist + λ·resid_dist)
"""

import argparse
import json
import os
import glob
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models
from data_utils import create_test_loaders_only, create_id_train_loader_only


class WDiscOODGridSearch:
    """WDiscOOD for grid search."""
    
    def __init__(self):
        self.V_D = None
        self.V_R = None
        self.class_means_wd = None
        self.global_mean_wdr = None
        
    def fit(self, features, labels):
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        n_samples, n_features = features_np.shape
        classes = np.unique(labels_np)
        n_classes = len(classes)
        
        global_mean = np.mean(features_np, axis=0)
        
        # Within-class scatter
        Sw = np.zeros((n_features, n_features))
        class_means = {}
        class_counts = {}
        
        for c in classes:
            mask = labels_np == c
            class_features = features_np[mask]
            class_mean = np.mean(class_features, axis=0)
            class_means[c] = class_mean
            class_counts[c] = len(class_features)
            centered = class_features - class_mean
            Sw += centered.T @ centered
        
        Sw /= n_samples
        Sw += 1e-6 * np.eye(n_features)
        
        # Whitening
        eigvals_w, eigvecs_w = np.linalg.eigh(Sw)
        eigvals_w = np.maximum(eigvals_w, 1e-10)
        Sw_inv_sqrt = eigvecs_w @ np.diag(1.0 / np.sqrt(eigvals_w)) @ eigvecs_w.T
        
        # Between-class scatter in whitened space
        Sb_tilde = np.zeros((n_features, n_features))
        for c in classes:
            diff = Sw_inv_sqrt @ (class_means[c] - global_mean)
            Sb_tilde += class_counts[c] * np.outer(diff, diff)
        Sb_tilde /= n_samples
        
        # Eigendecomposition
        eigvals_b, eigvecs_b = np.linalg.eigh(Sb_tilde)
        idx = np.argsort(eigvals_b)[::-1]
        eigvecs_b = eigvecs_b[:, idx]
        
        k = n_classes - 1
        V_D_tilde = eigvecs_b[:, :k]
        V_R_tilde = eigvecs_b[:, k:]
        
        self.V_D = Sw_inv_sqrt @ V_D_tilde
        self.V_R = Sw_inv_sqrt @ V_R_tilde
        
        self.class_means_wd = np.array([
            class_means[c] @ self.V_D for c in sorted(class_means.keys())
        ])
        self.global_mean_wdr = global_mean @ self.V_R
        
    def get_distances(self, features):
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        
        # Center features (match evaluate_wdiscood.py implementation)
        global_mean = np.mean(features_np, axis=0)
        centered = features_np - global_mean
        
        # Project centered features to discriminative space
        features_wd = centered @ self.V_D
        # Project centered CLASS means to discriminative space
        global_mean_wd = global_mean @ self.V_D
        class_means_wd_centered = self.class_means_wd - global_mean_wd
        
        disc_dists = []
        for f in features_wd:
            dists = np.linalg.norm(class_means_wd_centered - f, axis=1)
            disc_dists.append(np.min(dists))
        disc_dist = np.array(disc_dists)
        
        # Project centered features to residual space
        features_wdr = centered @ self.V_R
        # Distance to origin (because already centered)
        resid_dist = np.linalg.norm(features_wdr, axis=1)
        
        return disc_dist, resid_dist


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting", leave=False):
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = torch.zeros(images.size(0), dtype=torch.long)
            
            images = images.to(device)
            # FedAvg_Model returns (logits, features)
            _, features = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def find_checkpoint(experiments_root: str, model_type: str) -> str:
    pattern = f"{experiments_root}/{model_type}/experiment_*/best_model.pth"
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]
    
    pattern = f"{experiments_root}/{model_type}/experiment_*/checkpoint_epoch_*.pth"
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]
    
    raise FileNotFoundError(f"No checkpoint found for {model_type}")


def grid_search_lambda(checkpoint_path: str, data_root: str, lambda_values: list, device: str = 'cuda'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'densenet169')
    num_classes = config.get('num_classes', 54)
    image_size = config.get('image_size', 320)
    
    print(f"Model: {model_type}, Classes: {num_classes}, Image size: {image_size}")
    
    # Create model
    model = models.create_model(model_type, num_classes=num_classes)
    
    # Load weights
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
    
    print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    near_features, _ = extract_features(model, near_ood_loader, device)
    far_features, _ = extract_features(model, far_ood_loader, device)
    
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")
    
    # Fit WDiscOOD
    print("Fitting WLDA...")
    wdiscood = WDiscOODGridSearch()
    wdiscood.fit(train_features, train_labels)
    
    # Get distances
    print("Computing distances...")
    id_disc, id_resid = wdiscood.get_distances(test_features)
    near_disc, near_resid = wdiscood.get_distances(near_features)
    far_disc, far_resid = wdiscood.get_distances(far_features)
    
    # Grid search
    results = []
    print(f"\nSearching λ ∈ [{lambda_values[0]:.2f}, {lambda_values[-1]:.2f}]...")
    print("-" * 70)
    print(f"{'λ':>8} │ {'Near AUROC':>12} │ {'Far AUROC':>12} │ {'Avg AUROC':>12}")
    print("-" * 70)
    
    best_lambda = None
    best_near_auroc = 0
    
    for lam in lambda_values:
        # Score = -(disc + λ * resid)  [Lower score = more OOD]
        # For sklearn ROC: labels=1 for ID, scores should be HIGHER for ID
        # So we use negative of distance: score = -(disc + λ*resid)
        id_scores = -(id_disc + lam * id_resid)
        near_scores = -(near_disc + lam * near_resid)
        far_scores = -(far_disc + lam * far_resid)
        
        near_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(near_scores))])
        near_all = np.concatenate([id_scores, near_scores])
        near_auroc = roc_auc_score(near_labels, near_all)
        
        far_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(far_scores))])
        far_all = np.concatenate([id_scores, far_scores])
        far_auroc = roc_auc_score(far_labels, far_all)
        
        avg_auroc = (near_auroc + far_auroc) / 2
        
        results.append({
            'lambda': float(lam),
            'near_auroc': float(near_auroc),
            'far_auroc': float(far_auroc),
            'avg_auroc': float(avg_auroc)
        })
        
        marker = " *" if near_auroc > best_near_auroc else ""
        print(f"{lam:>8.2f} │ {near_auroc*100:>11.2f}% │ {far_auroc*100:>11.2f}% │ {avg_auroc*100:>11.2f}%{marker}")
        
        if near_auroc > best_near_auroc:
            best_near_auroc = near_auroc
            best_lambda = lam
    
    print("-" * 70)
    print(f"Best λ = {best_lambda:.2f} (Near AUROC: {best_near_auroc*100:.2f}%)")
    
    return {
        'model_type': model_type,
        'best_lambda': float(best_lambda),
        'best_near_auroc': float(best_near_auroc),
        'grid_results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Grid search λ for Pure WDiscOOD')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--experiments_root', type=str, default='./experiments/experiments_rerun_v1')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset')
    parser.add_argument('--lambda_min', type=float, default=0.1)
    parser.add_argument('--lambda_max', type=float, default=3.0)
    parser.add_argument('--lambda_steps', type=int, default=15)
    parser.add_argument('--all_models', action='store_true')
    parser.add_argument('--output', type=str, default='paper_tools/wdiscood_lambda_search.json')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps).tolist()
    
    if args.all_models:
        all_models = ['convnext_base', 'deit_base', 'densenet169', 'efficientnet_v2_s',
                      'mobilenetv3_large', 'resnet101', 'resnet50', 'vit_b_16', 'vit_b_32']
        
        all_results = {}
        for model_type in all_models:
            print(f"\n{'='*70}")
            print(f"Model: {model_type}")
            print(f"{'='*70}")
            
            try:
                checkpoint = find_checkpoint(args.experiments_root, model_type)
                result = grid_search_lambda(checkpoint, args.data_root, lambda_values, device)
                all_results[model_type] = result
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: Optimal λ for each model")
        print("=" * 70)
        print(f"{'Model':<20} │ {'Best λ':>8} │ {'Near AUROC':>12} │ {'Far AUROC':>12}")
        print("-" * 70)
        
        for model_type, result in all_results.items():
            best_lam = result['best_lambda']
            for r in result['grid_results']:
                if abs(r['lambda'] - best_lam) < 0.01:
                    print(f"{model_type:<20} │ {best_lam:>8.2f} │ {r['near_auroc']*100:>11.2f}% │ {r['far_auroc']*100:>11.2f}%")
                    break
        
        avg_best_lambda = np.mean([r['best_lambda'] for r in all_results.values()])
        print("-" * 70)
        print(f"Average optimal λ: {avg_best_lambda:.2f}")
        
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to: {args.output}")
        
    else:
        if args.checkpoint:
            checkpoint = args.checkpoint
        elif args.model_type:
            checkpoint = find_checkpoint(args.experiments_root, args.model_type)
        else:
            parser.error("Provide --checkpoint, --model_type, or --all_models")
        
        result = grid_search_lambda(checkpoint, args.data_root, lambda_values, device)


if __name__ == '__main__':
    main()
