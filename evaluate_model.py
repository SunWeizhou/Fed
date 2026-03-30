#!/usr/bin/env python3
"""模型评估脚本。"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from models import FedAvg_Model, Backbone
from data_utils import create_id_train_loader_only, create_test_loaders_only
from utils.ood_utils import compute_ood_metrics, compute_vim_scores, estimate_vim_alpha_empirical

# 设置非交互式后端
import matplotlib
matplotlib.use('Agg')

# 忽略所有 matplotlib 和 sklearn 的 UserWarning (字体和 precision 警告)
warnings.filterwarnings('ignore', category=UserWarning)


def load_checkpoint(checkpoint_path, device):
    """加载检查点"""
    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # 提取关键信息
    round_num = checkpoint.get('round', 'Unknown')
    test_acc = checkpoint.get('test_acc', 0.0)

    print(f"  - Round: {round_num}")
    print(f"  - Test Accuracy: {test_acc:.2f}%")

    return checkpoint


def create_model_from_config(config, device):
    """根据配置创建模型"""
    model_type = config.get('model_type', 'densenet121')
    num_classes = 54  # 检查点显示模型是54类

    backbone = Backbone(model_type=model_type, pretrained=False)
    model = FedAvg_Model(backbone, num_classes=num_classes, hidden_dim=512)

    model = model.to(device)
    model.eval()

    return model


def evaluate_classification(model, test_loader, device, class_names=None):
    """Evaluate classification performance"""
    print("\n" + "="*60)
    print("Classification Performance Evaluation")
    print("="*60)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / total
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Correctly Classified: {correct}/{total}")

    # 打印分类报告
    print("\nClassification Report:")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(54)]
    print(classification_report(all_labels, all_preds, target_names=class_names,
                                digits=4, zero_division=0))

    # 返回 accuracy (已经是百分比形式，例如 95.92)
    return accuracy


def compute_energy_scores(logits):
    """计算 Energy-based OOD 分数 (负的 LogSumExp)

    注意：这返回的是 OOD 检测分数，不是原始 Energy 值。
    - 分数越低（更负）→ 越可能是 ID（高置信度，LogSumExp 大）
    - 分数越高（接近 0）→ 越可能是 OOD（低置信度，LogSumExp 小）

    这仅用于 Energy-only OOD 检测的 fallback 方法。
    实际的 ViM Score 计算使用的是正的 LogSumExp（见 evaluate_ood 函数）。
    """
    # Energy-based OOD Score = -log(sum(exp(logits)))
    # 对于 ID: LogSumExp 大 (~6-8) → 分数为负 (~-6 to -8)
    # 对于 OOD: LogSumExp 小 (~2-4) → 分数较高 (~-2 to -4)
    return -torch.logsumexp(logits, dim=1).numpy()


def evaluate_ood(model, id_loader, near_ood_loader, far_ood_loader, device, use_energy=False, vim_stats=None,
                 train_loader=None, alpha_method='empirical'):
    """Evaluate OOD detection performance for both Near-OOD and Far-OOD"""
    print("\n" + "="*60)
    print("OOD Detection Performance Evaluation")
    print("="*60)

    if use_energy:
        print("\nUsing fallback method: Energy-based OOD detection...")

        # 提取 ID 数据 logit
        id_logits = []
        with torch.no_grad():
            for images, _ in id_loader:
                images = images.to(device)
                logits, _ = model(images)
                id_logits.append(logits.cpu())

        id_logits = torch.cat(id_logits, dim=0)

        # 提取 Near-OOD 和 Far-OOD 数据 logit
        near_ood_logits = []
        with torch.no_grad():
            for images, _ in near_ood_loader:
                images = images.to(device)
                logits, _ = model(images)
                near_ood_logits.append(logits.cpu())

        near_ood_logits = torch.cat(near_ood_logits, dim=0)

        far_ood_logits = []
        with torch.no_grad():
            for images, _ in far_ood_loader:
                images = images.to(device)
                logits, _ = model(images)
                far_ood_logits.append(logits.cpu())

        far_ood_logits = torch.cat(far_ood_logits, dim=0)

        # 计算 Energy 分数 (负号: 低 energy = OOD)
        id_scores = -torch.logsumexp(id_logits, dim=1).numpy()
        near_ood_scores = -torch.logsumexp(near_ood_logits, dim=1).numpy()
        far_ood_scores = -torch.logsumexp(far_ood_logits, dim=1).numpy()

        method_name = "Energy (Fallback method - not accurate)"

    else:
        # Fed-ViM 模型 - 使用完整的 ViM Score (带全局子空间投影)
        if vim_stats is None or vim_stats.get('P') is None:
            print("\nWarning: Global subspace statistics not found, using simplified ViM Score...")
            # 简化版本: 使用到特征均值的距离作为残差代理
            id_features = []
            id_logits = []
            with torch.no_grad():
                for images, labels in id_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    id_features.append(features.cpu())
                    id_logits.append(logits.cpu())

            id_features = torch.cat(id_features, dim=0)
            id_logits = torch.cat(id_logits, dim=0)

            near_ood_features = []
            near_ood_logits = []
            with torch.no_grad():
                for images, _ in near_ood_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    near_ood_features.append(features.cpu())
                    near_ood_logits.append(logits.cpu())

            near_ood_features = torch.cat(near_ood_features, dim=0)
            near_ood_logits = torch.cat(near_ood_logits, dim=0)

            far_ood_features = []
            far_ood_logits = []
            with torch.no_grad():
                for images, _ in far_ood_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    far_ood_features.append(features.cpu())
                    far_ood_logits.append(logits.cpu())

            far_ood_features = torch.cat(far_ood_features, dim=0)
            far_ood_logits = torch.cat(far_ood_logits, dim=0)

            # 计算特征均值 (作为全局子空间中心)
            mu_global = id_features.mean(dim=0)

            # 简化版残差: 直接到均值的距离
            id_residual = torch.norm(id_features - mu_global, p=2, dim=1).numpy()
            near_ood_residual = torch.norm(near_ood_features - mu_global, p=2, dim=1).numpy()
            far_ood_residual = torch.norm(far_ood_features - mu_global, p=2, dim=1).numpy()

            method_name = "ViM (Simplified - no subspace projection)"
        else:
            # 完整版 ViM Score - 使用全局子空间投影
            print("\nUsing full Fed-ViM OOD detection (with global subspace projection)...")

            # 加载全局统计量
            P_global = vim_stats['P'].to(device)  # 全局子空间投影矩阵
            mu_global = vim_stats['mu'].to(device)  # 全局特征均值

            print(f"  - Subspace dimension: {P_global.shape[1]}")
            print(f"  - Feature dimension: {P_global.shape[0]}")

            # 提取 ID 数据特征和 logit
            id_features = []
            id_logits = []
            with torch.no_grad():
                for images, labels in id_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    id_features.append(features)
                    id_logits.append(logits)

            id_features = torch.cat(id_features, dim=0)
            id_logits = torch.cat(id_logits, dim=0)

            # 提取 Near-OOD 数据特征和 logit
            near_ood_features = []
            near_ood_logits = []
            with torch.no_grad():
                for images, _ in near_ood_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    near_ood_features.append(features)
                    near_ood_logits.append(logits)

            near_ood_features = torch.cat(near_ood_features, dim=0)
            near_ood_logits = torch.cat(near_ood_logits, dim=0)

            # 提取 Far-OOD 数据特征和 logit
            far_ood_features = []
            far_ood_logits = []
            with torch.no_grad():
                for images, _ in far_ood_loader:
                    images = images.to(device)
                    logits, features = model(images)
                    far_ood_features.append(features)
                    far_ood_logits.append(logits)

            far_ood_features = torch.cat(far_ood_features, dim=0)
            far_ood_logits = torch.cat(far_ood_logits, dim=0)

            # 计算完整版残差: Residual = ||(I - PP^T)(z - mu)||
            # 1. 中心化
            z_centered_id = id_features - mu_global
            z_centered_near_ood = near_ood_features - mu_global
            z_centered_far_ood = far_ood_features - mu_global

            # 2. 投影到子空间
            z_proj_id = torch.matmul(z_centered_id, P_global)
            z_proj_near_ood = torch.matmul(z_centered_near_ood, P_global)
            z_proj_far_ood = torch.matmul(z_centered_far_ood, P_global)

            # 3. 重构
            z_recon_id = torch.matmul(z_proj_id, P_global.T)
            z_recon_near_ood = torch.matmul(z_proj_near_ood, P_global.T)
            z_recon_far_ood = torch.matmul(z_proj_far_ood, P_global.T)

            # 4. 计算残差 (原始特征与重构特征的差异)
            id_residual = torch.norm(z_centered_id - z_recon_id, p=2, dim=1).cpu().numpy()
            near_ood_residual = torch.norm(z_centered_near_ood - z_recon_near_ood, p=2, dim=1).cpu().numpy()
            far_ood_residual = torch.norm(z_centered_far_ood - z_recon_far_ood, p=2, dim=1).cpu().numpy()

            method_name = f"ViM (Full - {P_global.shape[1]}D subspace projection)"

        # 计算 Energy (LogSumExp)
        id_energy = torch.logsumexp(id_logits, dim=1).cpu().numpy()
        near_ood_energy = torch.logsumexp(near_ood_logits, dim=1).cpu().numpy()
        far_ood_energy = torch.logsumexp(far_ood_logits, dim=1).cpu().numpy()

        if alpha_method == 'empirical' and train_loader is not None:
            alpha, _, _ = estimate_vim_alpha_empirical(
                model=model,
                loaders=train_loader,
                P=P_global,
                mu=mu_global,
                device=device,
            )
            print(f"  - Using Empirical Alpha (from ID train features): {alpha:.4f}")
        elif vim_stats is not None and 'alpha' in vim_stats and vim_stats['alpha'] is not None:
            alpha = vim_stats['alpha']
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            print(f"  - Using Stored Alpha from checkpoint: {alpha:.4f}")
        else:
            # Fallback: 如果检查点里没有，再用测试集计算 (不推荐，但作为保底)
            print(f"  - Warning: Pre-calibrated Alpha not found in checkpoint!")
            print(f"    Re-calibrating on Test set (WARNING: This may cause data leakage!)")
            mean_energy = np.mean(id_energy)
            mean_residual = np.mean(id_residual)
            alpha = mean_energy / (mean_residual + 1e-8)
            print(f"  - Auto-calibrated Alpha (on Test set): {alpha:.4f}")

        print(f"  - ID Residual Mean: {np.mean(id_residual):.4f} ± {np.std(id_residual):.4f}")
        print(f"  - ID Energy Mean: {np.mean(id_energy):.4f} ± {np.std(id_energy):.4f}")
        print(f"  - Near-OOD Residual Mean: {np.mean(near_ood_residual):.4f} ± {np.std(near_ood_residual):.4f}")
        print(f"  - Near-OOD Energy Mean: {np.mean(near_ood_energy):.4f} ± {np.std(near_ood_energy):.4f}")
        print(f"  - Far-OOD Residual Mean: {np.mean(far_ood_residual):.4f} ± {np.std(far_ood_residual):.4f}")
        print(f"  - Far-OOD Energy Mean: {np.mean(far_ood_energy):.4f} ± {np.std(far_ood_energy):.4f}")

        # ViM Score = Energy - alpha * Residual (原版 ViM 公式)
        # 参考: Sun et al. "ViM: Out-of-Distribution Detection with Variance in Mahalanobis Distance", NeurIPS 2022
        # 机制说明:
        #   - ID: 高 Energy (置信度高 ~6-8) + 低 Residual → Score 高
        #   - OOD: 低 Energy (置信度低 ~2-4) + 高 Residual → Score 低
        #   - alpha 将几何距离归一化到能量空间
        # 结果: OOD 的 Score 更低 (Low score = more likely OOD)
        id_scores = compute_vim_scores(id_energy, id_residual, alpha)
        near_ood_scores = compute_vim_scores(near_ood_energy, near_ood_residual, alpha)
        far_ood_scores = compute_vim_scores(far_ood_energy, far_ood_residual, alpha)

        print(f"\n{method_name}")

    # 评估 Near-OOD 检测
    print("\n" + "="*60)
    print("Near-OOD Detection Results")
    print("="*60)
    near_metrics = compute_ood_metrics(id_scores, near_ood_scores)
    near_auroc = near_metrics['auroc']
    near_aupr = near_metrics['aupr']
    near_fpr95 = near_metrics['fpr95']

    print(f"  - AUROC: {near_auroc:.4f}")
    print(f"  - AUPR:  {near_aupr:.4f}")
    print(f"  - FPR95: {near_fpr95:.4f}")
    print(f"  - ID Score Mean: {np.mean(id_scores):.4f} ± {np.std(id_scores):.4f}")
    print(f"  - Near-OOD Score Mean: {np.mean(near_ood_scores):.4f} ± {np.std(near_ood_scores):.4f}")

    # 评估 Far-OOD 检测
    print("\n" + "="*60)
    print("Far-OOD Detection Results")
    print("="*60)
    far_metrics = compute_ood_metrics(id_scores, far_ood_scores)
    far_auroc = far_metrics['auroc']
    far_aupr = far_metrics['aupr']
    far_fpr95 = far_metrics['fpr95']

    print(f"  - AUROC: {far_auroc:.4f}")
    print(f"  - AUPR:  {far_aupr:.4f}")
    print(f"  - FPR95: {far_fpr95:.4f}")
    print(f"  - ID Score Mean: {np.mean(id_scores):.4f} ± {np.std(id_scores):.4f}")
    print(f"  - Far-OOD Score Mean: {np.mean(far_ood_scores):.4f} ± {np.std(far_ood_scores):.4f}")

    results = {
        'method': method_name,
        'near_auroc': near_auroc,
        'near_aupr': near_aupr,
        'near_fpr95': near_fpr95,
        'far_auroc': far_auroc,
        'far_aupr': far_aupr,
        'far_fpr95': far_fpr95,
        'id_scores_mean': np.mean(id_scores),
        'id_scores_std': np.std(id_scores),
        'near_ood_scores_mean': np.mean(near_ood_scores),
        'near_ood_scores_std': np.std(near_ood_scores),
        'far_ood_scores_mean': np.mean(far_ood_scores),
        'far_ood_scores_std': np.std(far_ood_scores)
    }

    return results, (id_scores, near_ood_scores, far_ood_scores, method_name)
def plot_ood_distribution(id_scores, near_ood_scores, far_ood_scores, method_name, output_path):
    """Plot OOD score distribution for both Near-OOD and Far-OOD"""
    plt.figure(figsize=(18, 5))

    # Near-OOD Distribution
    plt.subplot(1, 3, 1)
    plt.hist(id_scores, bins=50, alpha=0.6, label='ID (In-Distribution)', color='blue', density=True)
    plt.hist(near_ood_scores, bins=50, alpha=0.6, label='Near-OOD', color='orange', density=True)
    plt.xlabel(f'{method_name} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'ID vs Near-OOD Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Far-OOD Distribution
    plt.subplot(1, 3, 2)
    plt.hist(id_scores, bins=50, alpha=0.6, label='ID (In-Distribution)', color='blue', density=True)
    plt.hist(far_ood_scores, bins=50, alpha=0.6, label='Far-OOD', color='red', density=True)
    plt.xlabel(f'{method_name} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'ID vs Far-OOD Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Combined Distribution
    plt.subplot(1, 3, 3)
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID', color='blue', density=True)
    plt.hist(near_ood_scores, bins=50, alpha=0.5, label='Near-OOD', color='orange', density=True)
    plt.hist(far_ood_scores, bins=50, alpha=0.5, label='Far-OOD', color='red', density=True)
    plt.xlabel(f'{method_name} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'All Distributions Combined', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"OOD distribution plot saved: {output_path}")


def plot_tsne_feature_space(model, id_loader, near_ood_loader, far_ood_loader, device, output_path, max_samples=2000):
    """
    Plot t-SNE feature space visualization for ID, Near-OOD, and Far-OOD

    Args:
        model: Trained model
        id_loader: ID data loader
        near_ood_loader: Near-OOD data loader
        far_ood_loader: Far-OOD data loader
        device: Computing device
        output_path: Save path
        max_samples: Maximum number of samples (to prevent slow computation)
    """
    from sklearn.manifold import TSNE

    model.eval()
    features_list = []
    labels_list = []  # 0 for ID, 1 for Near-OOD, 2 for Far-OOD

    print("\nCollecting features for t-SNE visualization...")

    # 1. Collect ID features
    with torch.no_grad():
        count = 0
        for data, _ in id_loader:
            data = data.to(device)
            _, feats = model(data)
            features_list.append(feats.cpu().numpy())
            labels_list.append(np.zeros(len(feats)))
            count += len(feats)
            if count > max_samples // 3:
                break

    # 2. Collect Near-OOD features
    with torch.no_grad():
        count = 0
        for data, _ in near_ood_loader:
            data = data.to(device)
            _, feats = model(data)
            features_list.append(feats.cpu().numpy())
            labels_list.append(np.ones(len(feats)))
            count += len(feats)
            if count > max_samples // 3:
                break

    # 3. Collect Far-OOD features
    with torch.no_grad():
        count = 0
        for data, _ in far_ood_loader:
            data = data.to(device)
            _, feats = model(data)
            features_list.append(feats.cpu().numpy())
            labels_list.append(2 * np.ones(len(feats)))
            count += len(feats)
            if count > max_samples // 3:
                break

    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)

    print(f"  - ID samples: {np.sum(y==0)}")
    print(f"  - Near-OOD samples: {np.sum(y==1)}")
    print(f"  - Far-OOD samples: {np.sum(y==2)}")
    print(f"  - Feature dimension: {X.shape[1]}")

    # 4. t-SNE dimensionality reduction
    print("\nComputing t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto',
                perplexity=min(30, len(X) // 4))  # Adaptive perplexity
    X_embedded = tsne.fit_transform(X)

    print("  t-SNE computation completed!")

    # 5. Plotting
    plt.figure(figsize=(12, 10))

    # ID data (blue)
    id_mask = y == 0
    plt.scatter(X_embedded[id_mask, 0], X_embedded[id_mask, 1],
               c='blue', alpha=0.5, s=15, label='In-Distribution (ID)', edgecolors='none')

    # Near-OOD data (orange)
    near_ood_mask = y == 1
    plt.scatter(X_embedded[near_ood_mask, 0], X_embedded[near_ood_mask, 1],
               c='orange', alpha=0.5, s=15, label='Near-OOD', edgecolors='none')

    # Far-OOD data (red)
    far_ood_mask = y == 2
    plt.scatter(X_embedded[far_ood_mask, 0], X_embedded[far_ood_mask, 1],
               c='red', alpha=0.5, s=15, label='Far-OOD', edgecolors='none')

    plt.legend(fontsize=12, markerscale=2)
    plt.title('t-SNE Feature Space Visualization\n(ID vs Near-OOD vs Far-OOD Separability)',
              fontsize=15, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.2)

    # Add annotation
    plt.text(0.02, 0.98,
             f"ID (blue, n={np.sum(id_mask)}): {np.sum(y==0)} samples\n"
             f"Near-OOD (orange, n={np.sum(near_ood_mask)}): {np.sum(y==1)} samples\n"
             f"Far-OOD (red, n={np.sum(far_ood_mask)}): {np.sum(y==2)} samples\n"
             f"Feature dim: {X.shape[1]} → t-SNE (2D)",
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"t-SNE feature space visualization saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint path (e.g., experiments_fedvim_dense121_224/experiment_XXX/best_model.pth)')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                       help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Image size (default: read from config.json, or 320)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as checkpoint directory)')
    parser.add_argument('--enable_tsne', action='store_true', default=False,
                       help='Enable t-SNE feature space visualization (time-consuming)')
    parser.add_argument('--tsne_samples', type=int, default=2000,
                       help='Maximum samples for t-SNE (default: 2000)')
    parser.add_argument('--alpha_method', type=str, default='empirical', choices=['empirical', 'stored'],
                       help='ViM alpha 评估口径。正式论文结果建议使用 empirical。')

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, 'config.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\nExperiment Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"Warning: Configuration file not found {config_path}")
        config = {}

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device)

    # Create model
    model = create_model_from_config(config, device)
    model.load_state_dict(checkpoint['global_model_state_dict'])

    # Load data (returns 3 loaders: test, near_ood, far_ood)
    print("\nLoading datasets...")
    # Use provided image_size, or fall back to config, or default to 320
    image_size = args.image_size if args.image_size is not None else config.get('image_size', 320)
    print(f"  - Using image_size: {image_size}")
    train_loader = None
    if args.alpha_method == 'empirical':
        train_loader = create_id_train_loader_only(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=image_size
        )

    id_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size
    )

    print(f"  - ID classes: 54 (Plankton dataset)")
    print(f"  - ID batches: {len(id_loader)}")
    print(f"  - Near-OOD batches: {len(near_ood_loader)}")
    print(f"  - Far-OOD batches: {len(far_ood_loader)}")

    # Evaluate classification
    num_classes = 54  # Actual number of classes
    class_names = [f"Class {i}" for i in range(num_classes)]
    accuracy = evaluate_classification(model, id_loader, device, class_names)

    # Load ViM statistics (if Fed-ViM model)
    vim_stats = None
    if 'vim_stats' in checkpoint:
        vim_stats = checkpoint['vim_stats']
        print("\nLoaded ViM statistics from checkpoint:")
        print(f"  - P shape: {vim_stats['P'].shape}")
        print(f"  - mu shape: {vim_stats['mu'].shape}")
    else:
        print("\nWarning: vim_stats not found in checkpoint, using simplified ViM Score")
        print("  (Recommendation: Ensure global subspace statistics are saved during training)")

    # Evaluate OOD detection
    ood_results, (id_scores, near_ood_scores, far_ood_scores, method_name) = evaluate_ood(
        model, id_loader, near_ood_loader, far_ood_loader, device,
        use_energy=False,
        vim_stats=vim_stats,
        train_loader=train_loader,
        alpha_method=args.alpha_method,
    )

    # Save results
    output_dir = args.output_dir if args.output_dir else checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Generating Visualization Results")
    print("="*60)

    # 1. Save OOD distribution plot
    ood_plot_path = os.path.join(output_dir, f'ood_distribution.png')
    plot_ood_distribution(id_scores, near_ood_scores, far_ood_scores, method_name, ood_plot_path)

    # 2. t-SNE feature space visualization (optional, time-consuming)
    if args.enable_tsne:
        tsne_path = os.path.join(output_dir, f'tsne_feature_space.png')
        plot_tsne_feature_space(model, id_loader, near_ood_loader, far_ood_loader, device, tsne_path,
                               max_samples=args.tsne_samples)

    # Save evaluation summary
    summary_path = os.path.join(output_dir, f'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: Plankton ({num_classes} classes)\n\n")

        f.write("Classification Performance:\n")
        f.write(f"  - Accuracy: {accuracy:.2f}%\n\n")

        f.write(f"OOD Detection Performance:\n")
        f.write(f"  - Method: {method_name}\n\n")

        f.write("Near-OOD Detection:\n")
        f.write(f"  - AUROC: {ood_results['near_auroc']:.4f}\n")
        f.write(f"  - AUPR:  {ood_results['near_aupr']:.4f}\n")
        f.write(f"  - FPR95: {ood_results['near_fpr95']:.4f}\n")
        f.write(f"  - ID Score Mean: {ood_results['id_scores_mean']:.4f} ± {ood_results['id_scores_std']:.4f}\n")
        f.write(f"  - Near-OOD Score Mean: {ood_results['near_ood_scores_mean']:.4f} ± {ood_results['near_ood_scores_std']:.4f}\n\n")

        f.write("Far-OOD Detection:\n")
        f.write(f"  - AUROC: {ood_results['far_auroc']:.4f}\n")
        f.write(f"  - AUPR:  {ood_results['far_aupr']:.4f}\n")
        f.write(f"  - FPR95: {ood_results['far_fpr95']:.4f}\n")
        f.write(f"  - ID Score Mean: {ood_results['id_scores_mean']:.4f} ± {ood_results['id_scores_std']:.4f}\n")
        f.write(f"  - Far-OOD Score Mean: {ood_results['far_ood_scores_mean']:.4f} ± {ood_results['far_ood_scores_std']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  1. OOD Distribution: {os.path.basename(ood_plot_path)}")
    if args.enable_tsne:
        print(f"  2. t-SNE Feature Space: {os.path.basename(tsne_path)}")
    print(f"  3. Evaluation Summary: {os.path.basename(summary_path)}")


if __name__ == '__main__':
    main()
