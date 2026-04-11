#!/usr/bin/env python3
"""Evaluate the thesis-oriented FOOGD baseline."""

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
from methods.foogd import FOOGDScoreModel
from utils.ood_utils import compute_energy_ood_scores, compute_msp_ood_scores, compute_ood_metrics


THESIS_TITLE = "FedViM：面向海洋浮游生物多中心监测的联邦分布外检测方法研究"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the independent FOOGD baseline.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./Plankton_OOD_Dataset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--evaluation_score", type=str, default="sm", choices=["sm", "msp", "energy"])
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def collect_outputs(model, score_model, loader, device: torch.device, collect_targets: bool = False):
    logits_list = []
    score_list = []
    target_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits, features = model(images)
            scores = score_model(features).norm(dim=-1)
            logits_list.append(logits.cpu())
            score_list.append(scores.cpu())
            if collect_targets:
                target_list.append(labels.cpu())
    logits = torch.cat(logits_list, dim=0).numpy()
    scores = torch.cat(score_list, dim=0).numpy()
    if collect_targets:
        targets = torch.cat(target_list, dim=0).numpy()
        return logits, scores, targets
    return logits, scores


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint, config, checkpoint_dir = load_checkpoint_bundle(args.checkpoint, device)
    model, model_type, num_classes, feature_dim = build_model_from_checkpoint(checkpoint, config, device)
    image_size = resolve_image_size(config, args.image_size)

    score_hidden_dim = int(config.get("score_hidden_dim", 1024))
    score_model = FOOGDScoreModel(feature_dim=feature_dim, hidden_dim=score_hidden_dim).to(device)
    score_model.load_state_dict(checkpoint["score_model_state_dict"], strict=True)
    score_model.eval()

    id_loader, near_loader, far_loader = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    print("\nCollecting logits and score-model outputs for FOOGD evaluation...")
    id_logits, id_sm_scores, id_targets = collect_outputs(model, score_model, id_loader, device, collect_targets=True)
    near_logits, near_sm_scores = collect_outputs(model, score_model, near_loader, device)
    far_logits, far_sm_scores = collect_outputs(model, score_model, far_loader, device)

    if args.evaluation_score == "sm":
        id_scores = id_sm_scores
        near_scores = near_sm_scores
        far_scores = far_sm_scores
        invert_scores = False
    elif args.evaluation_score == "msp":
        id_scores = compute_msp_ood_scores(id_logits)
        near_scores = compute_msp_ood_scores(near_logits)
        far_scores = compute_msp_ood_scores(far_logits)
        invert_scores = False
    else:
        id_scores = compute_energy_ood_scores(id_logits)
        near_scores = compute_energy_ood_scores(near_logits)
        far_scores = compute_energy_ood_scores(far_logits)
        invert_scores = False

    near_metrics = compute_ood_metrics(id_scores, near_scores, invert_scores=invert_scores)
    far_metrics = compute_ood_metrics(id_scores, far_scores, invert_scores=invert_scores)
    id_accuracy = float(np.mean(np.argmax(id_logits, axis=1) == id_targets))

    output_dir = result_output_dir(checkpoint_dir, args.output_dir)
    payload = {
        "method": "FOOGD",
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
        "score_model_hidden_dim": score_hidden_dim,
        "n_clients": int(config.get("n_clients", 5)),
        "alpha": float(config.get("alpha", 0.1)),
        "communication_rounds": int(config.get("communication_rounds", 0)),
        "local_epochs": int(config.get("local_epochs", 0)),
        "lambda_sag": float(config.get("lambda_sag", 0.05)),
        "lambda_sm3d": float(config.get("lambda_sm3d", 0.5)),
        "fourier_mix_alpha": float(config.get("fourier_mix_alpha", 1.0)),
        "result_scope": config.get("result_scope", "supplemental_foogd_baseline"),
        "notes": config.get(
            "notes",
            "Thesis-oriented FOOGD adaptation evaluated with the official score-model norm.",
        ),
    }

    filename = "foogd_results.json" if args.evaluation_score == "sm" else f"foogd_results_{args.evaluation_score}.json"
    result_path = write_result_json(output_dir, filename, payload)
    print(f"Near-OOD AUROC={payload['near_auroc']:.4f} AUPR={payload['near_aupr']:.4f} FPR95={payload['near_fpr95']:.4f}")
    print(f"Far-OOD  AUROC={payload['far_auroc']:.4f} AUPR={payload['far_aupr']:.4f} FPR95={payload['far_fpr95']:.4f}")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
