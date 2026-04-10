#!/usr/bin/env python3
"""ACT-FedViM post-hoc evaluation implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils import create_test_loaders_only
from evaluation_common import (
    build_model_from_checkpoint,
    calibrate_empirical_alpha,
    load_checkpoint_bundle,
    load_empirical_alpha_loaders,
    reconstruct_covariance_from_vim_stats,
    resolve_device,
    resolve_image_size,
    result_output_dir,
    write_result_json,
)
from server import FLServer
from utils.subspace_utils import select_vim_paper_k


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"


class ACTCorrection:
    """Adjusted Correlation Thresholding for adaptive rank selection."""

    def __init__(self, device: torch.device):
        self.device = device

    def cov2corr(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """Convert covariance to a numerically stable correlation matrix."""
        diag = torch.clamp(torch.diag(cov_matrix), min=1e-6)
        std = torch.sqrt(diag)
        outer_std = torch.outer(std, std)

        corr_matrix = cov_matrix / (outer_std + 1e-8)
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        corr_matrix = corr_matrix + torch.eye(corr_matrix.shape[0], device=self.device) * (
            1.0 - torch.diag(corr_matrix)
        )
        return corr_matrix

    def get_optimal_k(self, cov_global: torch.Tensor, n_samples: int) -> tuple[int, torch.Tensor, torch.Tensor, float, float]:
        """Estimate ACT rank and return corrected/raw eigen spectra."""
        p = cov_global.shape[0]
        corr = self.cov2corr(cov_global)

        eig_vals = None
        for jitter in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
            try:
                corr_jittered = corr + torch.eye(p, device=self.device) * jitter
                eig_vals, _ = torch.linalg.eigh(corr_jittered)
                if torch.any(torch.isnan(eig_vals)) or torch.any(torch.isinf(eig_vals)):
                    raise ValueError("NaN/Inf in eigenvalues")
                break
            except (torch._C._LinAlgError, ValueError):
                continue

        if eig_vals is None:
            raise RuntimeError("Failed to compute ACT correlation eigenvalues.")

        eig_vals = torch.flip(eig_vals, dims=[0])
        rho = p / (n_samples - 1)
        threshold_s = 1.0 + np.sqrt(rho)

        corrected_eigs = torch.zeros_like(eig_vals)
        corrected_eigs[-1] = eig_vals[-1]

        for j in range(p - 1):
            lambda_j = eig_vals[j]
            if lambda_j < 0.5:
                corrected_eigs[j] = lambda_j
                continue

            lambdas_rest = eig_vals[j + 1 :]
            diffs = lambdas_rest - lambda_j
            mask = torch.abs(diffs) > 1e-6
            if mask.sum() == 0:
                sum_term = 0.0
            else:
                sum_term = torch.sum(1.0 / diffs[mask])

            correction_term = 1.0 / (((3 * lambda_j + eig_vals[j + 1]) / 4) - lambda_j + 1e-8)
            m_nj = (1.0 / (p - j)) * (sum_term + correction_term)
            rho_j = (p - j) / (n_samples - 1)
            underline_m = -(1 - rho_j) * (1.0 / lambda_j) + rho_j * m_nj
            corrected_eigs[j] = -1.0 / (underline_m + 1e-8)

        valid_indices = torch.where(corrected_eigs > threshold_s)[0]
        optimal_k = int(valid_indices[-1].item() + 1) if len(valid_indices) else 1
        optimal_k = max(1, min(optimal_k, p))
        return optimal_k, corrected_eigs, eig_vals, float(rho), float(threshold_s)


def run_act_fedvim(args: argparse.Namespace) -> Path:
    """Run ACT-FedViM evaluation from a trained FedViM checkpoint."""
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    mu_global, cov_global, total_count = reconstruct_covariance_from_vim_stats(checkpoint.get("vim_stats", {}), device)

    act_solver = ACTCorrection(device=device)
    act_k, corrected_eigs, raw_corr_eigs, rho, threshold_s = act_solver.get_optimal_k(cov_global, total_count)
    fixed_k = select_vim_paper_k(feature_dim=feature_dim, num_classes=num_classes)

    eig_vals, eig_vecs = torch.linalg.eigh(cov_global)
    top_indices = torch.argsort(eig_vals, descending=True)[:act_k]
    P_act = eig_vecs[:, top_indices]

    if args.alpha_method == "empirical":
        alpha_loaders = load_empirical_alpha_loaders(
            config=config,
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
        )
        alpha, mean_energy, mean_residual = calibrate_empirical_alpha(
            model=model,
            loaders=alpha_loaders,
            P=P_act,
            mu=mu_global,
            device=device,
        )
        alpha_source = "empirical_from_client_statistics"
    else:
        alpha = checkpoint.get("vim_stats", {}).get("alpha")
        if alpha is None:
            raise ValueError("Checkpoint does not contain a stored alpha; rerun with --alpha_method empirical.")
        mean_energy = None
        mean_residual = None
        alpha_source = "stored_from_checkpoint"

    server = FLServer(model, device=device)
    act_vim_stats = {"P": P_act, "mu": mu_global, "alpha": float(alpha)}
    metrics = server.evaluate_global_model(
        test_loader=test_loader,
        near_ood_loader=near_ood_loader,
        far_ood_loader=far_ood_loader,
        vim_stats=act_vim_stats,
    )

    output_dir = result_output_dir(checkpoint_dir, args.output_dir)
    result_payload = {
        "method": "ACT-FedViM",
        "thesis_title": THESIS_TITLE,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "model_type": model_type,
        "image_size": image_size,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "id_accuracy": float(metrics["id_accuracy"]),
        "near_auroc": float(metrics["near_auroc"]),
        "near_aupr": float(metrics["near_aupr"]),
        "near_fpr95": float(metrics["near_fpr95"]),
        "far_auroc": float(metrics["far_auroc"]),
        "far_aupr": float(metrics["far_aupr"]),
        "far_fpr95": float(metrics["far_fpr95"]),
        "alpha": float(alpha),
        "alpha_source": alpha_source,
        "fixed_k": int(fixed_k),
        "act_k": int(act_k),
        "compression_rate": float(1.0 - (act_k / fixed_k)),
        "feature_compression_rate": float(1.0 - (act_k / feature_dim)),
        "n_train_samples": int(total_count),
        "act_threshold_s": threshold_s,
        "act_rho": rho,
        "empirical_train_mean_energy": None if mean_energy is None else float(mean_energy),
        "empirical_train_mean_residual": None if mean_residual is None else float(mean_residual),
        "raw_corr_top10_eigs": [float(x) for x in raw_corr_eigs[:10].tolist()],
        "corrected_top10_eigs": [float(x) for x in corrected_eigs[:10].tolist()],
        "result_scope": "five_cnn_mainline",
    }
    result_path = write_result_json(output_dir, "act_fedvim_results.json", result_payload)

    stats_payload = {
        "method": "ACT-FedViM",
        "vim_stats": {
            "P": P_act.detach().cpu(),
            "mu": mu_global.detach().cpu(),
            "alpha": float(alpha),
        },
        "fixed_k": int(fixed_k),
        "act_k": int(act_k),
    }
    torch.save(stats_payload, output_dir / "act_fedvim_stats.pth")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT-FedViM from a trained FedViM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or final_model.pth.")
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset", help="Dataset root.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument(
        "--alpha_method",
        type=str,
        default="empirical",
        choices=["empirical", "stored"],
        help="Empirical is the paper-default ACT-FedViM evaluation protocol.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for ACT-FedViM artifacts. Default: checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = run_act_fedvim(args)
    print(f"ACT-FedViM results saved to: {result_path}")


if __name__ == "__main__":
    main()
