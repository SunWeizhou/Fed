#!/usr/bin/env python3
"""
Verify structural claims about the WDiscOOD Fisher spectrum.

Checks:
1. The discriminative spectrum comes from S_w^{-1/2} S_b S_w^{-1/2}.
2. rank(S_b) is at most C - 1.
3. Eigenvalues beyond C - 1 are numerical tail terms rather than meaningful
   residual-background energy.
4. The sum of Fisher-tail eigenvalues is not the same quantity as actual WDR
   projected feature energy.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models
from data_utils import create_id_train_loader_only
from models import Backbone, FedAvg_Model
from test_act_wdiscood import extract_features


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


def build_wlda_statistics(features, labels, reg_scale=1e-5):
    device = features.device
    n_samples, feature_dim = features.shape
    n_classes = int(labels.max().item()) + 1

    global_mean = features.mean(dim=0)
    class_means = torch.zeros(n_classes, feature_dim, device=device)
    class_counts = torch.zeros(n_classes, device=device)
    s_w = torch.zeros(feature_dim, feature_dim, device=device)

    for class_idx in range(n_classes):
        mask = labels == class_idx
        class_features = features[mask]
        class_mean = class_features.mean(dim=0)
        class_means[class_idx] = class_mean
        class_counts[class_idx] = mask.sum()
        centered = class_features - class_mean
        s_w += centered.T @ centered

    centered_means = class_means - global_mean
    s_b = (centered_means.T * class_counts) @ centered_means

    s_w_reg = s_w + reg_scale * torch.eye(feature_dim, device=device) * torch.trace(s_w)
    eigvals_w, eigvecs_w = torch.linalg.eigh(s_w_reg)
    eigvals_w = torch.clamp(eigvals_w, min=1e-10)
    whitener = eigvecs_w @ torch.diag(1.0 / torch.sqrt(eigvals_w)) @ eigvecs_w.T

    fisher_matrix = whitener @ s_b @ whitener
    fisher_eigs, fisher_vecs = torch.linalg.eigh(fisher_matrix)
    order = torch.argsort(fisher_eigs, descending=True)
    fisher_eigs = fisher_eigs[order]
    fisher_vecs = fisher_vecs[:, order]

    return {
        "n_samples": n_samples,
        "feature_dim": feature_dim,
        "n_classes": n_classes,
        "global_mean": global_mean,
        "class_means": class_means,
        "class_counts": class_counts,
        "s_w": s_w,
        "s_b": s_b,
        "whitener": whitener,
        "fisher_matrix": fisher_matrix,
        "fisher_eigs": fisher_eigs,
        "fisher_vecs": fisher_vecs,
    }


def summarize_tail(stats, rank_tol, tail_start_dim):
    features = stats["features"]
    global_mean = stats["global_mean"]
    whitener = stats["whitener"]
    fisher_vecs = stats["fisher_vecs"]
    fisher_eigs = stats["fisher_eigs"]
    feature_dim = stats["feature_dim"]

    centered = features - global_mean
    z_white = centered @ whitener

    disc_basis = fisher_vecs[:, :tail_start_dim]
    resid_basis = fisher_vecs[:, tail_start_dim:]
    z_resid = z_white @ resid_basis if resid_basis.numel() > 0 else torch.zeros(features.shape[0], 1, device=features.device)

    tail_eigs = fisher_eigs[tail_start_dim:]
    tail_sum = float(torch.clamp(tail_eigs, min=0).sum().item())
    tail_abs_sum = float(torch.abs(tail_eigs).sum().item())
    tail_max = float(tail_eigs.max().item()) if tail_eigs.numel() > 0 else 0.0
    tail_min = float(tail_eigs.min().item()) if tail_eigs.numel() > 0 else 0.0
    tail_nontrivial = int((torch.abs(tail_eigs) > rank_tol).sum().item())

    resid_energy_mean = float((z_resid.pow(2).sum(dim=1)).mean().item())
    resid_energy_median = float((z_resid.pow(2).sum(dim=1)).median().item())
    resid_norm_mean = float(torch.norm(z_resid, p=2, dim=1).mean().item())

    return {
        "tail_start_dim": tail_start_dim,
        "tail_dim": int(max(feature_dim - tail_start_dim, 0)),
        "tail_sum_positive": tail_sum,
        "tail_sum_abs": tail_abs_sum,
        "tail_max": tail_max,
        "tail_min": tail_min,
        "tail_nontrivial_count_abs_gt_tol": tail_nontrivial,
        "resid_energy_mean": resid_energy_mean,
        "resid_energy_median": resid_energy_median,
        "resid_norm_mean": resid_norm_mean,
    }


def effective_rank_desc(desc_eigs, rel_tol):
    if desc_eigs.numel() == 0:
        return 0
    threshold = rel_tol * float(abs(desc_eigs[0].item()))
    return int((torch.abs(desc_eigs) > threshold).sum().item()), threshold


def main():
    parser = argparse.ArgumentParser(description="Verify WDiscOOD Fisher tail structure")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_root", default="./Plankton_OOD_Dataset", type=str)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--rank_tol", default=1e-6, type=float)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_type, image_size = load_model(args.checkpoint, device)

    train_loader = create_id_train_loader_only(args.data_root, batch_size=64, image_size=image_size)
    print("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    stats = build_wlda_statistics(train_features, train_labels)
    stats["features"] = train_features

    s_b_eigs = torch.flip(torch.linalg.eigvalsh(stats["s_b"]), dims=[0])
    fisher_eigs = stats["fisher_eigs"]

    s_b_rank = int(torch.linalg.matrix_rank(stats["s_b"], tol=args.rank_tol).item())
    fisher_rank = int(torch.linalg.matrix_rank(stats["fisher_matrix"], tol=args.rank_tol).item())
    c_minus_1 = int(stats["n_classes"] - 1)

    s_b_rank_rel_1e3, s_b_thr_1e3 = effective_rank_desc(s_b_eigs, 1e-3)
    s_b_rank_rel_1e4, s_b_thr_1e4 = effective_rank_desc(s_b_eigs, 1e-4)
    fisher_rank_rel_1e3, fisher_thr_1e3 = effective_rank_desc(fisher_eigs, 1e-3)
    fisher_rank_rel_1e4, fisher_thr_1e4 = effective_rank_desc(fisher_eigs, 1e-4)

    tail_summary = summarize_tail(stats, args.rank_tol, c_minus_1)

    result = {
        "model_type": model_type,
        "n_samples": int(stats["n_samples"]),
        "feature_dim": int(stats["feature_dim"]),
        "n_classes": int(stats["n_classes"]),
        "c_minus_1": c_minus_1,
        "rank_tol": float(args.rank_tol),
        "rank_s_b": s_b_rank,
        "rank_fisher_matrix": fisher_rank,
        "effective_rank_s_b_rel_1e3": s_b_rank_rel_1e3,
        "effective_rank_s_b_rel_1e4": s_b_rank_rel_1e4,
        "effective_rank_fisher_rel_1e3": fisher_rank_rel_1e3,
        "effective_rank_fisher_rel_1e4": fisher_rank_rel_1e4,
        "effective_rank_thresholds": {
            "s_b_rel_1e3": s_b_thr_1e3,
            "s_b_rel_1e4": s_b_thr_1e4,
            "fisher_rel_1e3": fisher_thr_1e3,
            "fisher_rel_1e4": fisher_thr_1e4,
        },
        "rank_claim_holds_effective_rel_1e3": bool(
            s_b_rank_rel_1e3 <= c_minus_1 and fisher_rank_rel_1e3 <= c_minus_1
        ),
        "rank_claim_holds_effective_rel_1e4": bool(
            s_b_rank_rel_1e4 <= c_minus_1 and fisher_rank_rel_1e4 <= c_minus_1
        ),
        "top10_fisher_eigs": fisher_eigs[:10].detach().cpu().tolist(),
        "fisher_eigs_around_c_minus_1": fisher_eigs[max(c_minus_1 - 3, 0):c_minus_1 + 5].detach().cpu().tolist(),
        "s_b_eigs_around_c_minus_1": s_b_eigs[max(c_minus_1 - 3, 0):c_minus_1 + 5].detach().cpu().tolist(),
        "tail_summary_from_c_minus_1": tail_summary,
    }

    print("\n=== Verification Summary ===")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
