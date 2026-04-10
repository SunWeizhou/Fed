#!/usr/bin/env python3
"""Evaluate the fixed-k FedViM baseline from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def run_fedvim(args: argparse.Namespace) -> Path:
    """Run the fixed-k FedViM evaluation protocol."""
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

    vim_stats = checkpoint.get("vim_stats", {})
    mu_global, cov_global, total_count = reconstruct_covariance_from_vim_stats(vim_stats, device)
    fixed_k = select_vim_paper_k(feature_dim=feature_dim, num_classes=num_classes)

    checkpoint_P = vim_stats.get("P")
    if checkpoint_P is not None and checkpoint_P.shape[1] == fixed_k:
        P_fixed = checkpoint_P.to(device)
    else:
        eig_vals, eig_vecs = torch.linalg.eigh(cov_global)
        top_indices = torch.argsort(eig_vals, descending=True)[:fixed_k]
        P_fixed = eig_vecs[:, top_indices]

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
            P=P_fixed,
            mu=mu_global,
            device=device,
        )
        alpha_source = "empirical_from_client_statistics"
    else:
        alpha = vim_stats.get("alpha")
        if alpha is None:
            raise ValueError("Checkpoint does not contain a stored alpha; rerun with --alpha_method empirical.")
        mean_energy = None
        mean_residual = None
        alpha_source = "stored_from_checkpoint"

    server = FLServer(model, device=device)
    fedvim_stats = {
        "P": P_fixed,
        "mu": mu_global,
        "alpha": float(alpha),
    }
    metrics = server.evaluate_global_model(
        test_loader=test_loader,
        near_ood_loader=near_ood_loader,
        far_ood_loader=far_ood_loader,
        vim_stats=fedvim_stats,
    )

    output_dir = result_output_dir(checkpoint_dir, args.output_dir)
    result_payload = {
        "method": "FedViM",
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
        "act_k": None,
        "compression_rate": None,
        "feature_compression_rate": float(1.0 - (fixed_k / feature_dim)),
        "n_train_samples": int(total_count),
        "empirical_train_mean_energy": None if mean_energy is None else float(mean_energy),
        "empirical_train_mean_residual": None if mean_residual is None else float(mean_residual),
        "result_scope": "five_cnn_mainline",
    }
    result_path = write_result_json(output_dir, "fedvim_results.json", result_payload)

    torch.save(
        {
            "method": "FedViM",
            "vim_stats": {
                "P": P_fixed.detach().cpu(),
                "mu": mu_global.detach().cpu(),
                "alpha": float(alpha),
            },
            "fixed_k": int(fixed_k),
        },
        output_dir / "fedvim_stats.pth",
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fixed-k FedViM baseline.")
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
        help="Empirical is the paper-default FedViM evaluation protocol.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for FedViM artifacts. Default: checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = run_fedvim(args)
    print(f"FedViM results saved to: {result_path}")


if __name__ == "__main__":
    main()
