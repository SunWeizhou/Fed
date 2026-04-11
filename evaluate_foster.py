#!/usr/bin/env python3
"""Evaluate the thesis-oriented independent FOSTER baseline."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the independent FOSTER baseline.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--evaluation_score", type=str, default="msp", choices=["msp", "energy"])
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def collect_logits(model, loader, device: torch.device, collect_targets: bool = False):
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    id_loader, near_loader, far_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    print("\nCollecting logits for FOSTER evaluation...")
    id_logits, id_targets = collect_logits(model, id_loader, device, collect_targets=True)
    near_logits = collect_logits(model, near_loader, device)
    far_logits = collect_logits(model, far_loader, device)

    if args.evaluation_score == "msp":
        scorer = compute_msp_ood_scores
    else:
        scorer = compute_energy_ood_scores

    id_scores = scorer(id_logits)
    near_scores = scorer(near_logits)
    far_scores = scorer(far_logits)
    near_metrics = compute_ood_metrics(id_scores, near_scores, invert_scores=False)
    far_metrics = compute_ood_metrics(id_scores, far_scores, invert_scores=False)
    id_accuracy = float(np.mean(np.argmax(id_logits, axis=1) == id_targets))

    output_dir = result_output_dir(checkpoint_dir, args.output_dir)
    payload = {
        "method": "FOSTER",
        "thesis_title": THESIS_TITLE,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
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
        "evaluation_score": args.evaluation_score,
        "n_clients": int(config.get("n_clients", 5)),
        "alpha": float(config.get("alpha", 0.1)),
        "communication_rounds": int(config.get("communication_rounds", config.get("iters", 0))),
        "local_epochs": int(config.get("local_epochs", config.get("wk_iters", 0))),
        "loss_weight": float(config.get("loss_weight", 0.1)),
        "generator_type": config.get("generator_type", "class_conditional_feature_generator"),
        "dirichlet_alpha": float(config.get("alpha", 0.1)),
        "result_scope": config.get("result_scope", "supplemental_foster_baseline"),
        "notes": config.get(
            "notes",
            "Thesis-oriented FOSTER adaptation evaluated with a post-hoc output score.",
        ),
    }

    filename = "foster_results.json" if args.evaluation_score == "msp" else f"foster_results_{args.evaluation_score}.json"
    result_path = write_result_json(output_dir, filename, payload)
    print(f"Near-OOD AUROC={payload['near_auroc']:.4f} AUPR={payload['near_aupr']:.4f} FPR95={payload['near_fpr95']:.4f}")
    print(f"Far-OOD  AUROC={payload['far_auroc']:.4f} AUPR={payload['far_aupr']:.4f} FPR95={payload['far_fpr95']:.4f}")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
