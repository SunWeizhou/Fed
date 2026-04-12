#!/usr/bin/env python3
"""ACT-FedViM post-hoc evaluation implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import create_test_loaders_only
from evaluation_common import (
    aggregate_feature_statistics_from_loaders,
    build_model_from_checkpoint,
    calibrate_empirical_alpha,
    load_checkpoint_bundle,
    load_empirical_alpha_loaders,
    resolve_device,
    resolve_image_size,
    result_output_dir,
    write_result_json,
)
from server import FLServer
from utils.subspace_utils import compute_act_k_from_covariance, select_vim_paper_k


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"

def run_act_fedvim(args: argparse.Namespace) -> Path:
    """Run ACT-FedViM evaluation from federated ID statistics."""
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    alpha_loaders = load_empirical_alpha_loaders(
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

    federated_stats = aggregate_feature_statistics_from_loaders(model, alpha_loaders, device)
    mu_global = federated_stats["mu"]
    cov_global = federated_stats["cov"].to(dtype=torch.float64)
    total_count = int(federated_stats["count"])

    act_k, corrected_eigs, raw_corr_eigs, rho, threshold_s = compute_act_k_from_covariance(cov_global, total_count)
    fixed_k = select_vim_paper_k(feature_dim=feature_dim, num_classes=num_classes)

    eig_vals, eig_vecs = torch.linalg.eigh(cov_global)
    P_act = eig_vecs[:, -act_k:].contiguous()

    if args.alpha_method != "empirical":
        raise ValueError("ACT-FedViM evaluation now requires empirical alpha recomputation from client ID-train splits.")

    alpha, mean_energy, mean_residual = calibrate_empirical_alpha(
        model=model,
        loaders=alpha_loaders,
        P=P_act,
        mu=mu_global,
        device=device,
    )
    alpha_source = "empirical_from_client_splits"

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
        "statistics_source": "federated_client_aggregation",
        "result_scope": "five_cnn_mainline",
    }
    result_path = write_result_json(output_dir, "act_fedvim_results.json", result_payload)

    stats_payload = {
        "method": "ACT-FedViM",
        "vim_stats": {
            "P": P_act.detach().cpu(),
            "mu": mu_global.detach().cpu(),
            "alpha": float(alpha),
            "sum_z": federated_stats["sum_z"].detach().cpu(),
            "sum_zzT": federated_stats["sum_zzT"].detach().cpu(),
            "count": int(total_count),
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
        help="Post-hoc ACT-FedViM requires empirical alpha from client ID-train splits.",
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
