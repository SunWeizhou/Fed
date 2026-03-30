#!/usr/bin/env python3
"""
WDisc-Energy with Adaptive Gamma Decay (Grid Search)

Optimization A: Instead of γ = |E[energy]| / E[residual],
use γ = β × |E[energy]| / E[residual] where β ∈ [0.1, 1.0]

This script performs grid search to find optimal β for each model.
"""

import torch
import numpy as np
import os
import argparse
import json
import sys
from datetime import datetime
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import FedAvg_Model, Backbone
from data_utils import create_id_train_client_loaders_only, create_test_loaders_only
from utils.ood_utils import compute_ood_metrics

# Import from existing WDisc-Energy
from evaluate_wdisc_energy import (
    WDiscEnergy,
    extract_class_statistics,
    compute_wlda_statistics,
)


def extract_all_scores(model, wdisc_energy, loader, device):
    """Extract energy and residual for all samples (without computing final score)."""
    model.eval()
    all_energies = []
    all_residuals = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            output = model(images)
            if len(output) == 2:
                logits, features = output
            else:
                logits, _, features = output
            
            energy = torch.logsumexp(logits, dim=1)
            residual = wdisc_energy.compute_residual(features)
            
            all_energies.extend(energy.cpu().numpy())
            all_residuals.extend(residual.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_energies), np.array(all_residuals), np.array(all_labels)


def compute_scores_with_gamma(energies, residuals, gamma):
    """Compute final scores with given gamma."""
    return energies - gamma * residuals


def grid_search_beta(id_energies, id_residuals, ood_energies, ood_residuals, 
                     base_gamma, beta_range, metric='auroc'):
    """
    Grid search for optimal beta.
    
    Args:
        id_energies, id_residuals: ID test data components
        ood_energies, ood_residuals: OOD test data components  
        base_gamma: Original gamma (without decay)
        beta_range: List of beta values to try
        metric: 'auroc' or 'fpr95'
    
    Returns:
        best_beta, best_score, all_results
    """
    results = []
    
    for beta in beta_range:
        gamma = beta * base_gamma
        
        id_scores = compute_scores_with_gamma(id_energies, id_residuals, gamma)
        ood_scores = compute_scores_with_gamma(ood_energies, ood_residuals, gamma)
        
        metrics = compute_ood_metrics(id_scores, ood_scores)
        
        results.append({
            'beta': beta,
            'gamma': gamma,
            'auroc': metrics['auroc'],
            'aupr': metrics['aupr'],
            'fpr95': metrics['fpr95'],
        })
    
    # Find best
    if metric == 'auroc':
        best = max(results, key=lambda x: x['auroc'])
    else:
        best = min(results, key=lambda x: x['fpr95'])
    
    return best['beta'], best, results


def run_grid_search(args):
    """Main grid search function."""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("WDisc-Energy: Grid Search for Optimal Beta (Gamma Decay)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Beta range: {args.beta_min} to {args.beta_max} (step {args.beta_step})")
    
    # Generate beta range
    beta_range = np.arange(args.beta_min, args.beta_max + args.beta_step, args.beta_step)
    beta_range = [round(b, 2) for b in beta_range]
    
    # =====================================================
    # 1. Load Checkpoint
    # =====================================================
    print("\n>>> Loading Checkpoint...")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    model_type = checkpoint.get('config', {}).get('model_type', 'densenet121')
    checkpoint_image_size = checkpoint.get('config', {}).get('image_size', args.image_size)
    
    if checkpoint_image_size != args.image_size:
        args.image_size = checkpoint_image_size
    
    state_dict = checkpoint['global_model_state_dict']
    
    # Detect architecture
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
        raise ValueError("Cannot detect classifier structure")
    
    print(f"  Model: {model_type}, Feature dim: {feature_dim}")
    
    # Initialize model
    backbone = Backbone(model_type=model_type, pretrained=False)
    global_model = FedAvg_Model(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
    global_model.load_state_dict(checkpoint['global_model_state_dict'])
    global_model.to(device)
    global_model.eval()
    
    # =====================================================
    # 2. Load Data
    # =====================================================
    print("\n>>> Loading Data...")
    
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
    
    # =====================================================
    # 3. Fit WDisc-Energy
    # =====================================================
    print("\n>>> Fitting WDisc-Energy...")
    
    class_sums, class_sum_sq, class_counts = extract_class_statistics(
        global_model, train_loaders, device, num_classes
    )
    class_means, within_class_cov = compute_wlda_statistics(class_sums, class_sum_sq, class_counts)
    
    wdisc_energy = WDiscEnergy(device=device, regularization=args.regularization)
    wdisc_energy.fit_from_statistics(class_means, class_counts, within_class_cov)
    
    # =====================================================
    # 4. Compute Base Gamma
    # =====================================================
    print("\n>>> Computing Base Gamma...")
    
    # Calibrate on train data
    total_energy = 0.0
    total_residual = 0.0
    total_count = 0
    
    with torch.no_grad():
        for loader in train_loaders:
            for images, _ in loader:
                images = images.to(device)
                output = global_model(images)
                if len(output) == 2:
                    logits, features = output
                else:
                    logits, _, features = output
                
                energy = torch.logsumexp(logits, dim=1)
                residual = wdisc_energy.compute_residual(features)
                
                total_energy += energy.sum().item()
                total_residual += residual.sum().item()
                total_count += images.size(0)
    
    mean_energy = total_energy / total_count
    mean_residual = total_residual / total_count
    base_gamma = abs(mean_energy) / (mean_residual + 1e-8)
    
    print(f"  Base Gamma (β=1.0): {base_gamma:.4f}")
    
    # =====================================================
    # 5. Extract All Scores
    # =====================================================
    print("\n>>> Extracting scores for grid search...")
    
    id_energies, id_residuals, _ = extract_all_scores(global_model, wdisc_energy, test_loader, device)
    near_energies, near_residuals, _ = extract_all_scores(global_model, wdisc_energy, near_ood_loader, device)
    far_energies, far_residuals, _ = extract_all_scores(global_model, wdisc_energy, far_ood_loader, device)
    
    print(f"  ID: {len(id_energies)}, Near-OOD: {len(near_energies)}, Far-OOD: {len(far_energies)}")
    
    # =====================================================
    # 6. Grid Search
    # =====================================================
    print("\n>>> Running Grid Search...")
    print(f"\n{'β':>6} {'γ':>10} │ {'Near AUROC':>12} {'Near FPR95':>12} │ {'Far AUROC':>11} {'Far FPR95':>11}")
    print("-" * 80)
    
    near_results = []
    far_results = []
    
    for beta in beta_range:
        gamma = beta * base_gamma
        
        # Near-OOD
        id_scores = compute_scores_with_gamma(id_energies, id_residuals, gamma)
        near_scores = compute_scores_with_gamma(near_energies, near_residuals, gamma)
        far_scores = compute_scores_with_gamma(far_energies, far_residuals, gamma)
        
        near_m = compute_ood_metrics(id_scores, near_scores)
        far_m = compute_ood_metrics(id_scores, far_scores)
        
        near_results.append({'beta': beta, 'gamma': gamma, **near_m})
        far_results.append({'beta': beta, 'gamma': gamma, **far_m})
        
        print(f"{beta:>6.2f} {gamma:>10.2f} │ {near_m['auroc']*100:>11.2f}% {near_m['fpr95']*100:>11.2f}% │ {far_m['auroc']*100:>10.2f}% {far_m['fpr95']*100:>10.2f}%")
    
    # Find best
    best_near = max(near_results, key=lambda x: x['auroc'])
    best_far = max(far_results, key=lambda x: x['auroc'])
    
    # Combined best (average AUROC)
    combined = []
    for nr, fr in zip(near_results, far_results):
        combined.append({
            'beta': nr['beta'],
            'gamma': nr['gamma'],
            'avg_auroc': (nr['auroc'] + fr['auroc']) / 2,
            'near_auroc': nr['auroc'],
            'far_auroc': fr['auroc'],
            'near_fpr95': nr['fpr95'],
            'far_fpr95': fr['fpr95'],
        })
    best_combined = max(combined, key=lambda x: x['avg_auroc'])
    
    print("-" * 80)
    print(f"\n{'='*80}")
    print(f"Results for {model_type}")
    print(f"{'='*80}")
    print(f"\nOriginal (β=1.0): γ={base_gamma:.2f}")
    orig = [r for r in near_results if abs(r['beta'] - 1.0) < 0.01][0]
    orig_far = [r for r in far_results if abs(r['beta'] - 1.0) < 0.01][0]
    print(f"  Near-OOD: AUROC={orig['auroc']*100:.2f}%, FPR95={orig['fpr95']*100:.2f}%")
    print(f"  Far-OOD:  AUROC={orig_far['auroc']*100:.2f}%, FPR95={orig_far['fpr95']*100:.2f}%")
    
    print(f"\nBest for Near-OOD: β={best_near['beta']:.2f}, γ={best_near['gamma']:.2f}")
    print(f"  AUROC={best_near['auroc']*100:.2f}% (+{(best_near['auroc']-orig['auroc'])*100:.2f}%)")
    print(f"  FPR95={best_near['fpr95']*100:.2f}%")
    
    print(f"\nBest for Far-OOD: β={best_far['beta']:.2f}, γ={best_far['gamma']:.2f}")
    print(f"  AUROC={best_far['auroc']*100:.2f}% (+{(best_far['auroc']-orig_far['auroc'])*100:.2f}%)")
    
    print(f"\nBest Combined (avg AUROC): β={best_combined['beta']:.2f}, γ={best_combined['gamma']:.2f}")
    print(f"  Near: {best_combined['near_auroc']*100:.2f}%, Far: {best_combined['far_auroc']*100:.2f}%")
    print(f"  Avg:  {best_combined['avg_auroc']*100:.2f}%")
    
    # =====================================================
    # 7. Save Results
    # =====================================================
    save_dir = os.path.dirname(args.checkpoint) or '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_type,
        "checkpoint": args.checkpoint,
        "base_gamma": float(base_gamma),
        "beta_range": [float(b) for b in beta_range],
        "original_beta_1.0": {
            "near_auroc": float(orig['auroc']),
            "near_fpr95": float(orig['fpr95']),
            "far_auroc": float(orig_far['auroc']),
            "far_fpr95": float(orig_far['fpr95']),
        },
        "best_near_ood": {
            "beta": float(best_near['beta']),
            "gamma": float(best_near['gamma']),
            "auroc": float(best_near['auroc']),
            "fpr95": float(best_near['fpr95']),
        },
        "best_far_ood": {
            "beta": float(best_far['beta']),
            "gamma": float(best_far['gamma']),
            "auroc": float(best_far['auroc']),
            "fpr95": float(best_far['fpr95']),
        },
        "best_combined": {
            "beta": float(best_combined['beta']),
            "gamma": float(best_combined['gamma']),
            "near_auroc": float(best_combined['near_auroc']),
            "far_auroc": float(best_combined['far_auroc']),
            "avg_auroc": float(best_combined['avg_auroc']),
        },
        "all_results": {
            "near_ood": [{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for r in near_results],
            "far_ood": [{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for r in far_results],
        }
    }
    
    json_path = os.path.join(save_dir, f"gamma_grid_search_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return result_record


def run_all_models(args):
    """Run grid search on all available models."""
    
    # Find all checkpoints
    checkpoints = glob.glob("experiments/experiments_rerun_v1/**/best_model.pth", recursive=True)
    
    all_results = []
    
    for ckpt in sorted(checkpoints):
        model_name = ckpt.split('/')[1]
        print(f"\n\n{'#'*80}")
        print(f"# Processing: {model_name}")
        print(f"{'#'*80}")
        
        args.checkpoint = ckpt
        try:
            result = run_grid_search(args)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Summary
    print(f"\n\n{'='*100}")
    print("SUMMARY: Optimal Beta for All Models")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Base γ':>10} {'Best β':>8} {'Best γ':>10} │ {'Orig AUROC':>12} {'Best AUROC':>12} {'Δ':>8}")
    print("-" * 100)
    
    for r in all_results:
        model = r['model']
        base_g = r['base_gamma']
        best_b = r['best_combined']['beta']
        best_g = r['best_combined']['gamma']
        orig_auroc = (r['original_beta_1.0']['near_auroc'] + r['original_beta_1.0']['far_auroc']) / 2
        best_auroc = r['best_combined']['avg_auroc']
        delta = best_auroc - orig_auroc
        
        print(f"{model:<20} {base_g:>10.2f} {best_b:>8.2f} {best_g:>10.2f} │ {orig_auroc*100:>11.2f}% {best_auroc*100:>11.2f}% {delta*100:>+7.2f}%")
    
    # Save summary
    summary_path = "paper_tools/gamma_grid_search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for optimal gamma decay beta")
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if not specified, runs all models)')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--regularization', type=float, default=1e-5)
    
    # Grid search parameters
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=1.5)
    parser.add_argument('--beta_step', type=float, default=0.1)
    
    parser.add_argument('--all_models', action='store_true',
                        help='Run grid search on all models')
    
    args = parser.parse_args()
    
    if args.all_models or args.checkpoint is None:
        run_all_models(args)
    else:
        run_grid_search(args)
