#!/usr/bin/env python3
"""Evaluate the federated-compatible MSP and Energy baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils import create_test_loaders_only
from evaluation_common import (
    build_model_from_checkpoint,
    load_checkpoint_bundle,
    resolve_device,
    resolve_image_size,
    result_output_dir,
    write_result_json,
)
from utils.ood_utils import compute_energy_ood_scores, compute_msp_ood_scores, compute_ood_metrics


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"


def collect_logits(model, loader, device: torch.device, collect_targets: bool = False):
    """Collect logits and optional labels for a whole loader."""
    logits_list = []
    target_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            logits_list.append(logits.cpu())
            if collect_targets:
                target_list.append(labels.cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if collect_targets:
        targets = torch.cat(target_list, dim=0).numpy()
        return logits, targets
    return logits


def evaluate_method(method_key: str, id_logits, near_logits, far_logits):
    """Evaluate one output-space baseline."""
    if method_key == "msp":
        label = "MSP"
        scorer = compute_msp_ood_scores
    elif method_key == "energy":
        label = "Energy"
        scorer = compute_energy_ood_scores
    else:
        raise ValueError(f"Unsupported method: {method_key}")

    id_scores = scorer(id_logits)
    near_scores = scorer(near_logits)
    far_scores = scorer(far_logits)

    near_metrics = compute_ood_metrics(id_scores, near_scores, invert_scores=False)
    far_metrics = compute_ood_metrics(id_scores, far_scores, invert_scores=False)
    return label, id_scores, near_scores, far_scores, near_metrics, far_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MSP and Energy baselines for a FedViM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or final_model.pth.")
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset", help="Dataset root.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=None, help="Override checkpoint image size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["msp", "energy"],
        choices=["msp", "energy"],
        help="Baselines to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <checkpoint_dir>/baselines_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    id_loader, near_ood_loader, far_ood_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    print("\nCollecting logits for ID / Near-OOD / Far-OOD...")
    id_logits, id_targets = collect_logits(model, id_loader, device, collect_targets=True)
    near_logits = collect_logits(model, near_ood_loader, device)
    far_logits = collect_logits(model, far_ood_loader, device)
    id_predictions = np.argmax(id_logits, axis=1)
    id_accuracy = float(np.mean(id_predictions == id_targets))

    output_dir = result_output_dir(checkpoint_dir / "baselines_eval", args.output_dir)
    checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())

    for method_key in args.methods:
        label, id_scores, near_scores, far_scores, near_metrics, far_metrics = evaluate_method(
            method_key=method_key,
            id_logits=id_logits,
            near_logits=near_logits,
            far_logits=far_logits,
        )

        result_payload = {
            "method": label,
            "thesis_title": THESIS_TITLE,
            "checkpoint": checkpoint_path,
            "model_type": model_type,
            "image_size": image_size,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "id_accuracy": id_accuracy,
            "near_auroc": float(near_metrics["auroc"]),
            "near_aupr": float(near_metrics["aupr"]),
            "near_fpr95": float(near_metrics["fpr95"]),
            "far_auroc": float(far_metrics["auroc"]),
            "far_aupr": float(far_metrics["aupr"]),
            "far_fpr95": float(far_metrics["fpr95"]),
            "alpha": None,
            "alpha_source": "not_applicable",
            "fixed_k": None,
            "act_k": None,
            "compression_rate": None,
            "feature_compression_rate": None,
            "id_score_mean": float(id_scores.mean()),
            "near_score_mean": float(near_scores.mean()),
            "far_score_mean": float(far_scores.mean()),
            "result_scope": "five_cnn_mainline",
        }

        filename = f"{method_key}_results.json"
        result_path = write_result_json(output_dir, filename, result_payload)
        print(f"\n=== {label} ===")
        print(
            f"Near-OOD: AUROC={near_metrics['auroc']:.4f} "
            f"AUPR={near_metrics['aupr']:.4f} FPR95={near_metrics['fpr95']:.4f}"
        )
        print(
            f"Far-OOD:  AUROC={far_metrics['auroc']:.4f} "
            f"AUPR={far_metrics['aupr']:.4f} FPR95={far_metrics['fpr95']:.4f}"
        )
        print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
