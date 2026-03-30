import copy
from typing import Any

import numpy as np
import torch

from utils.ood_utils import compute_ood_metrics, compute_vim_scores


class FLServer:
    """联邦学习服务端。"""

    def __init__(self, global_model, device):
        self.global_model = global_model
        self.device = device
        self.global_model.to(device)

        self.P_global = None
        self.mu_global = None
        self.alpha = 1.0
        self.alpha_calibrated = False

    def get_global_parameters(self):
        """获取全局模型参数。"""
        params = {}
        for key, value in self.global_model.state_dict().items():
            params[f"model.{key}"] = value.clone()
        return params

    def set_global_parameters(self, params):
        """设置全局模型参数。"""
        model_params = {}
        for key, value in params.items():
            if key.startswith("model."):
                model_params[key.replace("model.", "")] = value
            else:
                model_params[key] = value

        self.global_model.load_state_dict(model_params, strict=False)
        print(f"  [Server] Updated Global Model with {len(model_params)} params")

    def aggregate(self, updates, sample_sizes):
        """FedAvg 聚合。"""
        total_samples = sum(sample_sizes)
        aggregated_params = copy.deepcopy(updates[0])

        for key in aggregated_params.keys():
            if "num_batches_tracked" in key:
                continue

            aggregated_params[key] = torch.zeros_like(aggregated_params[key], dtype=torch.float)

            for update, n_samples in zip(updates, sample_sizes):
                weight = n_samples / total_samples
                param_data = update[key]
                if param_data.dtype != torch.float:
                    param_data = param_data.float()
                aggregated_params[key] += param_data * weight

        return aggregated_params

    def update_global_subspace(self, client_stats_list, k=None, target_variance_ratio=0.95):
        """基于聚合统计量更新全局 ViM 子空间。"""
        if not client_stats_list:
            return {"P": None, "mu": None}

        print("  [Server] Updating Global Subspace (PCA)...")

        total_count = sum(stats["count"] for stats in client_stats_list)
        if total_count == 0:
            return {"P": None, "mu": None}

        feature_dim = client_stats_list[0]["sum_z"].shape[0]
        global_sum_z = torch.zeros(feature_dim, device=self.device)
        global_sum_zzT = torch.zeros(feature_dim, feature_dim, device=self.device)

        for stats in client_stats_list:
            global_sum_z += stats["sum_z"]
            global_sum_zzT += stats["sum_zzT"]

        self.global_sum_z = global_sum_z
        self.global_sum_zzT = global_sum_zzT
        self.global_count = total_count

        self.mu_global = global_sum_z / total_count
        e_zzt = global_sum_zzT / total_count
        cov_global = e_zzt - torch.outer(self.mu_global, self.mu_global)
        cov_global += torch.eye(feature_dim, device=self.device) * 1e-6

        try:
            eig_vals, eig_vecs = torch.linalg.eigh(cov_global)

            if k is None:
                total_variance = eig_vals.sum()
                cumulative_variance = 0.0
                selected_indices = []
                for idx in range(len(eig_vals) - 1, -1, -1):
                    cumulative_variance += eig_vals[idx].item()
                    selected_indices.append(idx)
                    if cumulative_variance / total_variance >= target_variance_ratio:
                        break

                selected_indices = sorted(selected_indices)
                k_dynamic = len(selected_indices)
                print(f"  [Auto-K] Selected k={k_dynamic} to cover {target_variance_ratio * 100:.1f}% variance.")
                print(f"  [Auto-K] Cumulative variance: {cumulative_variance / total_variance * 100:.2f}%")
                self.P_global = eig_vecs[:, selected_indices]
            else:
                self.P_global = eig_vecs[:, -k:]
                print(f"  [Fixed-K] Using k={k} (fixed).")

            print(f"  [Server] PCA Done. Subspace shape: {self.P_global.shape}")
        except Exception as exc:
            print(f"  [Server] PCA Failed: {exc}")
            return {"P": None, "mu": None, "eigenvalues": None}

        return {"P": self.P_global, "mu": self.mu_global, "eigenvalues": eig_vals}

    def _compute_id_statistics(self, data_loader, vim_stats):
        """使用 ID 数据估计用于 ViM 打分的 alpha。"""
        self.global_model.eval()
        all_energy = []
        all_residuals = []

        P = vim_stats["P"]
        mu = vim_stats["mu"]

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                model_output = self.global_model(data)
                if len(model_output) == 2:
                    logits, features = model_output
                else:
                    logits, _, features = model_output

                energy = torch.logsumexp(logits, dim=1)
                all_energy.append(energy.cpu())

                z_centered = features - mu
                z_proj = torch.matmul(z_centered, P)
                z_recon = torch.matmul(z_proj, P.T)
                residual = torch.norm(z_centered - z_recon, p=2, dim=1)
                all_residuals.append(residual.cpu())

        mean_energy = torch.cat(all_energy).mean().item()
        mean_residual = torch.cat(all_residuals).mean().item()
        alpha = abs(mean_energy) / (mean_residual + 1e-8)

        print(f"  [Auto-Alpha] ID Mean Energy: {mean_energy:.4f} | ID Mean Res: {mean_residual:.4f}")
        print(f"  [Auto-Alpha] Calculated Alpha = {alpha:.4f}")
        return alpha

    def _compute_scores_and_metrics(self, data_loader, vim_stats=None, alpha=1.0):
        """一次性计算分类指标和 OOD 打分。"""
        self.global_model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_scores = []

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                model_output = self.global_model(data)
                if len(model_output) == 2:
                    logits_g, features = model_output
                else:
                    logits_g, _, features = model_output

                valid_mask = targets >= 0
                if valid_mask.any():
                    valid_targets = targets[valid_mask]
                    valid_logits = logits_g[valid_mask]

                    loss = torch.nn.functional.cross_entropy(valid_logits, valid_targets)
                    total_loss += loss.item() * valid_targets.size(0)
                    total_samples += valid_targets.size(0)

                    _, preds = torch.max(valid_logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(valid_targets.cpu().numpy())

                if vim_stats is not None and vim_stats["P"] is not None:
                    P = vim_stats["P"]
                    mu = vim_stats["mu"]
                    z_centered = features - mu
                    z_proj = torch.matmul(z_centered, P)
                    z_recon = torch.matmul(z_proj, P.T)
                    residual = torch.norm(z_centered - z_recon, p=2, dim=1)
                    energy = torch.logsumexp(logits_g, dim=1)
                    scores = compute_vim_scores(
                        energy.cpu().numpy(),
                        residual.cpu().numpy(),
                        alpha,
                    )
                    scores = torch.from_numpy(scores).to(features.device)
                else:
                    scores = torch.norm(features, dim=1)

                all_scores.extend(scores.cpu().numpy())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) if all_preds and all_targets else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "scores": np.array(all_scores),
            "targets": np.array(all_targets),
        }

    def evaluate_global_model(self, test_loader, near_ood_loader, far_ood_loader, vim_stats=None):
        """评估全局模型的分类和 OOD 检测性能。"""
        metrics: dict[str, Any] = {}
        print("  正在评估 Global Model (Fed-ViM)...")

        current_alpha = 1.0
        if vim_stats is not None and "alpha" in vim_stats:
            current_alpha = vim_stats["alpha"]
            print(f"  [Server] Using pre-calibrated Alpha from aggregated statistics: {current_alpha:.4f}")
        elif vim_stats is not None and vim_stats.get("P") is not None:
            print("  [Warning] Calibrating Alpha on Test Data (Test-Set Leakage Risk)...")
            current_alpha = self._compute_id_statistics(test_loader, vim_stats)

        self.alpha = current_alpha
        self.alpha_calibrated = True

        id_results = self._compute_scores_and_metrics(test_loader, vim_stats, alpha=current_alpha)
        metrics["id_accuracy"] = id_results["accuracy"]
        metrics["id_loss"] = id_results["loss"]
        id_scores = id_results["scores"]

        print(f"    -> ID Acc: {metrics['id_accuracy']:.4f}")

        if near_ood_loader:
            near_results = self._compute_scores_and_metrics(near_ood_loader, vim_stats, alpha=current_alpha)
            near_scores = near_results["scores"]
            near_metrics = compute_ood_metrics(id_scores, near_scores)
            metrics["near_auroc"] = near_metrics["auroc"]
            metrics["near_aupr"] = near_metrics["aupr"]
            metrics["near_fpr95"] = near_metrics["fpr95"]
            print(f"    -> Near AUROC: {metrics['near_auroc']:.4f}")
            print(f"    -> Near AUPR: {metrics['near_aupr']:.4f}")
            print(f"    -> Near FPR95: {metrics['near_fpr95']:.4f}")

        if far_ood_loader:
            far_results = self._compute_scores_and_metrics(far_ood_loader, vim_stats, alpha=current_alpha)
            far_scores = far_results["scores"]
            far_metrics = compute_ood_metrics(id_scores, far_scores)
            metrics["far_auroc"] = far_metrics["auroc"]
            metrics["far_aupr"] = far_metrics["aupr"]
            metrics["far_fpr95"] = far_metrics["fpr95"]
            print(f"    -> Far AUROC: {metrics['far_auroc']:.4f}")
            print(f"    -> Far AUPR: {metrics['far_aupr']:.4f}")
            print(f"    -> Far FPR95: {metrics['far_fpr95']:.4f}")

        return metrics


if __name__ == "__main__":
    print("测试联邦学习服务端...")
    from data_utils import create_federated_loaders
    from models import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = create_model()
    server = FLServer(model, device)

    print("\n测试参数管理...")
    global_params = server.get_global_parameters()
    print(f"全局参数数量: {len(global_params)}")

    print("\n测试参数聚合...")
    client_updates = []
    for _ in range(3):
        client_update = {}
        for name, param in global_params.items():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                client_update[name] = torch.randn_like(param)
            else:
                client_update[name] = param.clone()
        client_updates.append(client_update)
    client_sample_sizes = [100, 150, 200]
    aggregated_params = server.aggregate(client_updates, client_sample_sizes)
    print(f"聚合参数数量: {len(aggregated_params)}")

    print("\n测试模型评估...")
    data_root = "./data"
    try:
        _, test_loader, near_ood_loader, far_ood_loader, _ = create_federated_loaders(
            data_root,
            n_clients=3,
            batch_size=4,
            image_size=224,
        )
        metrics = server.evaluate_global_model(
            test_loader,
            near_ood_loader,
            far_ood_loader,
            None,
        )
        print("评估指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    except Exception as exc:
        print(f"跳过数据评估测试（可能缺少数据）: {exc}")

    print("服务端测试完成!")
