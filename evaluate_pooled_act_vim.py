#!/usr/bin/env python3
"""Evaluate centralized pooled-statistics ACT-ViM on a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import create_test_loaders_only
from evaluation_common import (
    build_model_from_checkpoint,
    calibrate_empirical_alpha,
    compute_loader_feature_statistics,
    load_checkpoint_bundle,
    load_pooled_train_loader,
    resolve_device,
    resolve_image_size,
    result_output_dir,
    write_result_json,
)
from server import FLServer
from utils.subspace_utils import compute_act_k_from_covariance, select_vim_paper_k


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"


def run_pooled_act_vim(args: argparse.Namespace) -> Path:
    """Run pooled-statistics ACT-ViM on the same backbone checkpoint used by FedViM."""
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    pooled_train_loader = load_pooled_train_loader(
        config=config,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )
    test_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    pooled_stats = compute_loader_feature_statistics(model, pooled_train_loader, device)
    mu_global = pooled_stats["mu"]
    cov_global = pooled_stats["cov"]
    total_count = int(pooled_stats["count"])

    act_k, corrected_eigs, raw_corr_eigs, rho, threshold_s = compute_act_k_from_covariance(cov_global, total_count)
    fixed_k = select_vim_paper_k(feature_dim=feature_dim, num_classes=num_classes)

    eig_vals, eig_vecs = torch.linalg.eigh(cov_global)
    top_indices = torch.argsort(eig_vals, descending=True)[:act_k]
    P_act = eig_vecs[:, top_indices]

    alpha, mean_energy, mean_residual = calibrate_empirical_alpha(
        model=model,
        loaders=pooled_train_loader,
        P=P_act,
        mu=mu_global,
        device=device,
    )

    server = FLServer(model, device=device)
    vim_stats = {"P": P_act, "mu": mu_global, "alpha": float(alpha)}
    metrics = server.evaluate_global_model(
        test_loader=test_loader,
        near_ood_loader=near_ood_loader,
        far_ood_loader=far_ood_loader,
        vim_stats=vim_stats,
    )

    output_dir = result_output_dir(checkpoint_dir, args.output_dir)
    result_payload = {
        "method": "Pooled-ACT-ViM",
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
        "alpha_source": "empirical_from_pooled_train",
        "fixed_k": int(fixed_k),
        "act_k": int(act_k),
        "compression_rate": float(1.0 - (act_k / fixed_k)),
        "feature_compression_rate": float(1.0 - (act_k / feature_dim)),
        "n_train_samples": total_count,
        "act_threshold_s": threshold_s,
        "act_rho": rho,
        "empirical_train_mean_energy": float(mean_energy),
        "empirical_train_mean_residual": float(mean_residual),
        "raw_corr_top10_eigs": [float(x) for x in raw_corr_eigs[:10].tolist()],
        "corrected_top10_eigs": [float(x) for x in corrected_eigs[:10].tolist()],
        "statistics_source": "pooled_id_train_same_split",
        "result_scope": "pooled_statistics_baseline",
    }
    result_path = write_result_json(output_dir, "pooled_act_vim_results.json", result_payload)

    torch.save(
        {
            "method": "Pooled-ACT-ViM",
            "vim_stats": {
                "P": P_act.detach().cpu(),
                "mu": mu_global.detach().cpu(),
                "alpha": float(alpha),
                "sum_z": pooled_stats["sum_z"].detach().cpu(),
                "sum_zzT": pooled_stats["sum_zzT"].detach().cpu(),
                "count": total_count,
            },
            "fixed_k": int(fixed_k),
            "act_k": int(act_k),
        },
        output_dir / "pooled_act_vim_stats.pth",
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pooled-statistics ACT-ViM from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or final_model.pth.")
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset", help="Dataset root.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for pooled ACT-ViM artifacts. Default: checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = run_pooled_act_vim(args)
    print(f"Pooled-ACT-ViM results saved to: {result_path}")


if __name__ == "__main__":
    main()
