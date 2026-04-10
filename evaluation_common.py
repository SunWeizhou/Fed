#!/usr/bin/env python3
"""Shared helpers for FedViM evaluation entrypoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from data_utils import create_id_train_client_loaders_only, create_id_train_pooled_loader_only
from models import Backbone, FedAvg_Model
from utils.ood_utils import (
    aggregate_empirical_alpha_statistics,
    compute_empirical_alpha_local_stats,
)


DEFAULT_FIVE_MODELS = [
    "resnet101",
    "efficientnet_v2_s",
    "mobilenetv3_large",
    "densenet169",
    "resnet50",
]


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a torch device string with a CUDA fallback."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load checkpoint and adjacent config.json when available."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_dir = checkpoint_path.parent
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = dict(checkpoint.get("config", {}))

    merged_config = dict(checkpoint.get("config", {}))
    merged_config.update(config)
    return checkpoint, merged_config, checkpoint_dir


def infer_model_spec(checkpoint: dict[str, Any], config: dict[str, Any]) -> tuple[str, int, int]:
    """Infer model_type, num_classes, and feature_dim from checkpoint/config."""
    model_type = config.get("model_type", checkpoint.get("config", {}).get("model_type", "densenet169"))
    state_dict = checkpoint["global_model_state_dict"]

    if "classifier.weight" in state_dict:
        num_classes, feature_dim = state_dict["classifier.weight"].shape
    else:
        classifier_weight = next(value for key, value in state_dict.items() if key.endswith("weight"))
        num_classes = int(config.get("num_classes", 54))
        feature_dim = classifier_weight.shape[1]

    return model_type, int(num_classes), int(feature_dim)


def build_model_from_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device) -> tuple[FedAvg_Model, str, int, int]:
    """Instantiate a model skeleton without downloading pretrained weights."""
    model_type, num_classes, feature_dim = infer_model_spec(checkpoint, config)
    backbone = Backbone(model_type=model_type, pretrained=False)
    model = FedAvg_Model(backbone=backbone, num_classes=num_classes)
    model.load_state_dict(checkpoint["global_model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_type, num_classes, feature_dim


def resolve_image_size(config: dict[str, Any], override: int | None = None) -> int:
    """Use caller override first, otherwise trust checkpoint/config."""
    if override is not None:
        return int(override)
    return int(config.get("image_size", 320))


def reconstruct_covariance_from_vim_stats(vim_stats: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Reconstruct covariance from aggregated federated sufficient statistics."""
    if not vim_stats:
        raise ValueError("Checkpoint does not contain vim_stats.")

    required = ("sum_z", "sum_zzT", "count", "mu")
    missing = [key for key in required if vim_stats.get(key) is None]
    if missing:
        raise ValueError(
            "Checkpoint is missing aggregated statistics required for FedViM/ACT "
            f"evaluation: {', '.join(missing)}"
        )

    mu = vim_stats["mu"].to(device)
    sum_zzT = vim_stats["sum_zzT"].to(device)
    total_count = int(vim_stats["count"])
    cov = sum_zzT / total_count - torch.outer(mu, mu)
    cov = cov + torch.eye(cov.shape[0], device=device) * 1e-6
    return mu, cov, total_count


def load_empirical_alpha_loaders(
    config: dict[str, Any],
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
):
    """Create per-client ID train loaders for empirical alpha calibration."""
    return create_id_train_client_loaders_only(
        data_root=data_root,
        n_clients=int(config.get("n_clients", 5)),
        alpha=float(config.get("alpha", 0.1)),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        partition_seed=config.get("seed", 42),
    )


def load_pooled_train_loader(
    config: dict[str, Any],
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
):
    """Create one pooled ID-train loader aligned with the federated train split."""
    return create_id_train_pooled_loader_only(
        data_root=data_root,
        n_clients=int(config.get("n_clients", 5)),
        alpha=float(config.get("alpha", 0.1)),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        partition_seed=config.get("seed", 42),
    )


def calibrate_empirical_alpha(model, loaders, P: torch.Tensor, mu: torch.Tensor, device: torch.device) -> tuple[float, float, float]:
    """Aggregate client-local alpha statistics for a given subspace."""
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    local_stats = [
        compute_empirical_alpha_local_stats(
            model=model,
            loader=loader,
            P=P,
            mu=mu,
            device=device,
        )
        for loader in loaders
    ]
    return aggregate_empirical_alpha_statistics(local_stats)


def compute_loader_feature_statistics(model, loader, device: torch.device) -> dict[str, Any]:
    """Compute pooled first- and second-order feature sufficient statistics for one loader."""
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "feature_dim"):
        raise ValueError("Model does not expose backbone.feature_dim required for ViM statistics.")

    feature_dim = int(model.backbone.feature_dim)
    sum_z = torch.zeros(feature_dim, device=device)
    sum_zzT = torch.zeros(feature_dim, feature_dim, device=device)
    total_count = 0

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            _, features = model(images)
            sum_z += features.sum(dim=0)
            sum_zzT += torch.matmul(features.T, features)
            total_count += images.size(0)

    if was_training:
        model.train()

    if total_count == 0:
        raise ValueError("No samples were available for pooled feature statistics.")

    mu = sum_z / total_count
    cov = sum_zzT / total_count - torch.outer(mu, mu)
    cov = cov + torch.eye(feature_dim, device=device) * 1e-6

    return {
        "mu": mu,
        "cov": cov,
        "sum_z": sum_z,
        "sum_zzT": sum_zzT,
        "count": int(total_count),
    }


def aggregate_feature_statistics_from_loaders(model, loaders, device: torch.device) -> dict[str, Any]:
    """Aggregate sufficient statistics from one or more client ID-train loaders."""
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    total_sum_z = None
    total_sum_zzT = None
    total_count = 0

    for loader in loaders:
        stats = compute_loader_feature_statistics(model, loader, device)
        if total_sum_z is None:
            total_sum_z = torch.zeros_like(stats["sum_z"])
            total_sum_zzT = torch.zeros_like(stats["sum_zzT"])
        total_sum_z += stats["sum_z"]
        total_sum_zzT += stats["sum_zzT"]
        total_count += int(stats["count"])

    if total_sum_z is None or total_sum_zzT is None or total_count == 0:
        raise ValueError("No ID-train samples were available for feature-statistics aggregation.")

    mu = total_sum_z / total_count
    cov = total_sum_zzT / total_count - torch.outer(mu, mu)
    cov = cov + torch.eye(cov.shape[0], device=device) * 1e-6

    return {
        "mu": mu,
        "cov": cov,
        "sum_z": total_sum_z,
        "sum_zzT": total_sum_zzT,
        "count": int(total_count),
    }


def result_output_dir(checkpoint_dir: Path, output_dir: str | None) -> Path:
    """Resolve where evaluation artifacts should be written."""
    target = Path(output_dir).expanduser().resolve() if output_dir else checkpoint_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def json_default(value: Any):
    """Best-effort JSON serializer for numpy/torch scalar outputs."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_result_json(output_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    """Write a structured JSON evaluation record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = dict(payload)
    payload.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
    return path
