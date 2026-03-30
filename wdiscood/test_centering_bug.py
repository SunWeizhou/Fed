"""Compare paper-consistent WD centers vs a centered-means variant."""
import torch
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import models
import data_utils

def test_both_approaches(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt['config']['model_type']
    num_classes = 54  # Fixed for Plankton dataset
    
    # Load model
    model = models.create_model(
        model_type=model_type,
        num_classes=num_classes
    )
    model.load_state_dict(ckpt['global_model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    test_loader, near_ood_loader, far_ood_loader = data_utils.create_test_loaders_only(
        data_root='./Plankton_OOD_Dataset',
        batch_size=32
    )
    
    # Extract features
    def extract_all_features(loader):
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(device)
                logits, features = model(images)
                all_feats.append(features.cpu())
                all_labels.append(labels)
        return torch.cat(all_feats), torch.cat(all_labels)
    
    print("Extracting features...")
    id_feats, id_labels = extract_all_features(test_loader)
    near_feats, _ = extract_all_features(near_ood_loader)
    far_feats, _ = extract_all_features(far_ood_loader)
    
    id_feats = id_feats.to(device)
    id_labels = id_labels.to(device)
    near_feats = near_feats.to(device)
    far_feats = far_feats.to(device)
    
    # Fit paper-consistent WLDA: whiten first, then solve LDA in whitened space.
    global_mean = id_feats.mean(dim=0)
    class_means = torch.stack([id_feats[id_labels == c].mean(dim=0) for c in range(num_classes)])
    
    # Compute S_w and S_b
    S_w = torch.zeros(id_feats.shape[1], id_feats.shape[1], device=device)
    for c in range(num_classes):
        mask = id_labels == c
        centered = id_feats[mask] - class_means[c]
        S_w += centered.T @ centered
    
    centered_means = class_means - global_mean
    class_counts = torch.tensor([(id_labels == c).sum().item() for c in range(num_classes)], device=device)
    S_b = (centered_means.T * class_counts) @ centered_means
    
    S_w_cov = S_w / id_feats.shape[0] + 1e-6 * torch.eye(S_w.shape[0], device=device)
    eigvals_w, eigvecs_w = torch.linalg.eigh(S_w_cov)
    eigvals_w = torch.clamp(eigvals_w, min=1e-10)
    S_w_inv_sqrt = eigvecs_w @ torch.diag(1.0 / torch.sqrt(eigvals_w)) @ eigvecs_w.T

    class_means_white = class_means @ S_w_inv_sqrt
    global_mean_white = global_mean @ S_w_inv_sqrt
    centered_means_white = class_means_white - global_mean_white
    S_b_whitened = (centered_means_white.T * class_counts) @ centered_means_white / id_feats.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(S_b_whitened)
    eigvals = torch.flip(eigvals, dims=[0])
    eigvecs = torch.flip(eigvecs, dims=[1])
    
    disc_dim = min(num_classes - 1, (eigvals > 1e-6 * eigvals[0]).sum().item())
    W_disc = eigvecs[:, :disc_dim]
    W_resid = torch.eye(id_feats.shape[1], device=device) - W_disc @ W_disc.T
    global_mean_resid = global_mean_white @ W_resid
    
    print(f"\nDiscriminative dimension: {disc_dim}")
    
    # Approach 1: paper-consistent Eq. 9.
    class_means_disc_v1 = class_means_white @ W_disc
    
    def compute_scores_v1(features):
        z_white = features @ S_w_inv_sqrt
        z_disc = z_white @ W_disc
        z_resid = z_white @ W_resid
        
        dists_sq = torch.cdist(z_disc.unsqueeze(0), class_means_disc_v1.unsqueeze(0), p=2).squeeze(0) ** 2
        disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])
        resid_dist = torch.norm(z_resid - global_mean_resid, p=2, dim=1)
        
        return -(disc_dist + resid_dist)
    
    # Approach 2: centered class means in WD. This is the variant under test.
    class_means_disc_v2 = (class_means_white - global_mean_white) @ W_disc
    
    def compute_scores_v2(features):
        z_white = features @ S_w_inv_sqrt
        z_disc = z_white @ W_disc
        z_resid = z_white @ W_resid
        
        dists_sq = torch.cdist(z_disc.unsqueeze(0), class_means_disc_v2.unsqueeze(0), p=2).squeeze(0) ** 2
        disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])
        resid_dist = torch.norm(z_resid - global_mean_resid, p=2, dim=1)
        
        return -(disc_dist + resid_dist)
    
    # Compute scores
    print("\nComputing scores with both approaches...")
    id_scores_v1 = compute_scores_v1(id_feats)
    near_scores_v1 = compute_scores_v1(near_feats)
    far_scores_v1 = compute_scores_v1(far_feats)
    
    id_scores_v2 = compute_scores_v2(id_feats)
    near_scores_v2 = compute_scores_v2(near_feats)
    far_scores_v2 = compute_scores_v2(far_feats)
    
    # Compute AUROC
    from sklearn.metrics import roc_auc_score
    
    y_true_near = [1]*len(id_feats) + [0]*len(near_feats)
    y_true_far = [1]*len(id_feats) + [0]*len(far_feats)
    
    near_auroc_v1 = roc_auc_score(y_true_near, torch.cat([id_scores_v1, near_scores_v1]).cpu().numpy())
    far_auroc_v1 = roc_auc_score(y_true_far, torch.cat([id_scores_v1, far_scores_v1]).cpu().numpy())
    
    near_auroc_v2 = roc_auc_score(y_true_near, torch.cat([id_scores_v2, near_scores_v2]).cpu().numpy())
    far_auroc_v2 = roc_auc_score(y_true_far, torch.cat([id_scores_v2, far_scores_v2]).cpu().numpy())
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nApproach 1 (paper-consistent WD centers):")
    print(f"  Near-OOD AUROC: {near_auroc_v1*100:.2f}%")
    print(f"  Far-OOD AUROC: {far_auroc_v1*100:.2f}%")
    print(f"  Average: {(near_auroc_v1 + far_auroc_v1)/2*100:.2f}%")
    
    print(f"\nApproach 2 (centered WD centers variant):")
    print(f"  Near-OOD AUROC: {near_auroc_v2*100:.2f}%")
    print(f"  Far-OOD AUROC: {far_auroc_v2*100:.2f}%")
    print(f"  Average: {(near_auroc_v2 + far_auroc_v2)/2*100:.2f}%")
    
    print(f"\nDifference:")
    print(f"  Near-OOD: {(near_auroc_v2 - near_auroc_v1)*100:+.2f}%")
    print(f"  Far-OOD: {(far_auroc_v2 - far_auroc_v1)*100:+.2f}%")

if __name__ == '__main__':
    checkpoint = 'experiments/experiments_rerun_v1/densenet169/experiment_20260316_163844/best_model.pth'
    test_both_approaches(checkpoint)
