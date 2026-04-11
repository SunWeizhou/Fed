"""Utility helpers for the independent FOSTER baseline."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_utils import create_federated_loaders

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment_dir(output_dir: str, model_type: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_root = Path(output_dir).expanduser()
    if model_root.name != model_type:
        model_root = model_root / model_type
    experiment_dir = model_root / f"experiment_{timestamp}"
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return experiment_dir.resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_checkpoint_payload(
    model: torch.nn.Module,
    round_idx: int,
    config: dict[str, Any],
    training_history: dict[str, Any],
) -> dict[str, Any]:
    return {
        "global_model_state_dict": model_state_cpu(model),
        "round": int(round_idx),
        "config": dict(config),
        "training_history": dict(training_history),
    }


def infer_observed_classes(loader) -> list[int]:
    """Collect the class ids present in one client split."""
    labels = set()
    dataset = loader.dataset
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        base_labels = dataset.dataset.labels
        for idx in dataset.indices:
            labels.add(int(base_labels[idx]))
    else:
        for _, batch_labels in loader:
            labels.update(int(label) for label in batch_labels.tolist())
    return sorted(label for label in labels if label >= 0)


def cosine_round_lr(
    base_lr: float,
    current_round: int,
    total_rounds: int,
    warmup_rounds: int,
    min_lr_factor: float,
) -> float:
    warmup_start_lr = 1e-5
    min_lr = base_lr * min_lr_factor
    if current_round < warmup_rounds:
        return warmup_start_lr + (base_lr - warmup_start_lr) * (current_round / max(1, warmup_rounds))
    progress = (current_round - warmup_rounds) / max(1, total_rounds - warmup_rounds)
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return float(min_lr + (base_lr - min_lr) * decay)


def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def copy_state_to_model(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    device_state = {key: value.to(next(model.parameters()).device) for key, value in state_dict.items()}
    model.load_state_dict(device_state, strict=True)


def resolve_output_dir(output_dir: str | None, default_root: Path) -> Path:
    if output_dir is None:
        return default_root.resolve()
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def should_enable_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def create_foster_federated_loaders(
    data_root: str,
    n_clients: int,
    alpha: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    partition_seed: int,
):
    """Thin wrapper that keeps FOSTER on the same canonical split as the mainline."""
    return create_federated_loaders(
        data_root=data_root,
        n_clients=n_clients,
        alpha=alpha,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        partition_seed=partition_seed,
    )
