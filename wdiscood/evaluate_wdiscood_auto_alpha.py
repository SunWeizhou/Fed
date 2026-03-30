#!/usr/bin/env python3
"""
Evaluate WDiscOOD with:
1. Wachter-based discriminative dimension selection
2. Data-driven alpha estimated from ID train features

Alpha options:
- median: median(disc_dist) / median(resid_norm)
- mean: mean(disc_dist) / mean(resid_norm)
- q90: quantile_0.9(disc_dist) / quantile_0.9(resid_norm)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models
from data_utils import create_id_train_loader_only, create_test_loaders_only
from models import Backbone, FedAvg_Model
from test_act_wdiscood import WDiscOOD_ACT, extract_features


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_type = config.get("model_type", "densenet169")
    num_classes = config.get("num_classes", 54)
    image_size = config.get("image_size", 320)

    if "global_model_state_dict" in checkpoint:
        state_dict = checkpoint["global_model_state_dict"]
        if "heads.head.weight" in state_dict:
            hidden_dim = state_dict["heads.head.weight"].shape[1]
        elif "classifier.weight" in state_dict:
            hidden_dim = state_dict["classifier.weight"].shape[1]
        elif "classifier.0.weight" in state_dict:
            hidden_dim = state_dict["classifier.0.weight"].shape[0]
        else:
            raise ValueError("Cannot detect classifier structure from checkpoint")

        backbone = Backbone(model_type=model_type, pretrained=False)
        model = FedAvg_Model(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
        model.load_state_dict(state_dict)
    else:
        model = models.create_model(model_type, num_classes=num_classes)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("server_model_state", {}))
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.replace("_orig_mod.", "").replace("model.", "")] = value
        model.load_state_dict(cleaned, strict=False)

    return model.to(device).eval(), model_type, image_size


def estimate_alpha(train_features, wdiscood, alpha_mode):
    centered = train_features - wdiscood.global_mean
    z_disc = centered @ wdiscood.W_disc
    z_resid = centered @ wdiscood.W_resid

    dists_sq = torch.cdist(z_disc, wdiscood.class_means_disc, p=2) ** 2
    disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])
    resid_norm = torch.norm(z_resid, p=2, dim=1)

    if alpha_mode == "median":
        numerator = disc_dist.median()
        denominator = resid_norm.median()
    elif alpha_mode == "mean":
        numerator = disc_dist.mean()
        denominator = resid_norm.mean()
    elif alpha_mode == "q90":
        numerator = torch.quantile(disc_dist, 0.9)
        denominator = torch.quantile(resid_norm, 0.9)
    else:
        raise ValueError(f"Unknown alpha mode: {alpha_mode}")

    alpha = float(numerator.item() / (denominator.item() + 1e-8))
    stats = {
        "alpha_mode": alpha_mode,
        "disc_median": float(disc_dist.median().item()),
        "resid_median": float(resid_norm.median().item()),
        "disc_mean": float(disc_dist.mean().item()),
        "resid_mean": float(resid_norm.mean().item()),
        "disc_q90": float(torch.quantile(disc_dist, 0.9).item()),
        "resid_q90": float(torch.quantile(resid_norm, 0.9).item()),
    }
    return alpha, stats


def compute_scores(features, wdiscood, alpha):
    features = features.to(wdiscood.device)
    centered = features - wdiscood.global_mean
    z_disc = centered @ wdiscood.W_disc
    z_resid = centered @ wdiscood.W_resid
    dists_sq = torch.cdist(z_disc, wdiscood.class_means_disc, p=2) ** 2
    disc_dist = torch.sqrt(dists_sq.min(dim=1)[0])
    resid_norm = torch.norm(z_resid, p=2, dim=1)
    return (-(disc_dist + alpha * resid_norm)).cpu().numpy()


def evaluate(checkpoint_path, data_root, alpha_mode, output_path, device):
    model, model_type, image_size = load_model(checkpoint_path, device)

    train_loader = create_id_train_loader_only(data_root, batch_size=64, image_size=image_size)
    test_loader, near_loader, far_loader = create_test_loaders_only(
        data_root, batch_size=64, image_size=image_size
    )

    print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, _ = extract_features(model, test_loader, device)
    near_features, _ = extract_features(model, near_loader, device)
    far_features, _ = extract_features(model, far_loader, device)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    wdiscood = WDiscOOD_ACT(device=device, dim_strategy="wachter")
    wdiscood.fit(train_features, train_labels)

    alpha, alpha_stats = estimate_alpha(train_features, wdiscood, alpha_mode)

    print(f"[Auto Alpha] mode={alpha_mode}, alpha={alpha:.6f}")
    id_scores = compute_scores(test_features, wdiscood, alpha)
    near_scores = compute_scores(near_features, wdiscood, alpha)
    far_scores = compute_scores(far_features, wdiscood, alpha)

    near_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(near_scores))])
    far_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(far_scores))])
    near_all = np.concatenate([id_scores, near_scores])
    far_all = np.concatenate([id_scores, far_scores])

    near_auroc = float(roc_auc_score(near_labels, near_all))
    far_auroc = float(roc_auc_score(far_labels, far_all))

    result = {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "disc_dim": int(wdiscood.disc_dim),
        "resid_dim": int(wdiscood.resid_dim),
        "alpha": float(alpha),
        "alpha_stats": alpha_stats,
        "wachter": wdiscood.wachter_info,
        "near_auroc": near_auroc,
        "far_auroc": far_auroc,
        "avg_auroc": float((near_auroc + far_auroc) / 2.0),
    }

    print(json.dumps(result, indent=2))
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDiscOOD with auto alpha")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_root", default="./Plankton_OOD_Dataset", type=str)
    parser.add_argument("--alpha_mode", default="median", choices=["median", "mean", "q90"])
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args.checkpoint, args.data_root, args.alpha_mode, args.output, device)


if __name__ == "__main__":
    main()
