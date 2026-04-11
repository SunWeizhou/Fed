"""Utility helpers for the independent FOSTER baseline."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import PlanktonDataset, _build_loader_kwargs, get_transforms, partition_data

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


def canonicalize_dataset_order(dataset) -> None:
    """Sort cached dataset entries by path to remove filesystem-order drift."""
    if not hasattr(dataset, "image_paths") or not hasattr(dataset, "labels"):
        return
    pairs = sorted(zip(dataset.image_paths, dataset.labels), key=lambda item: item[0])
    if not pairs:
        return
    dataset.image_paths = [path for path, _ in pairs]
    dataset.labels = [label for _, label in pairs]


def create_foster_federated_loaders(
    data_root: str,
    n_clients: int,
    alpha: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    partition_seed: int,
):
    """Create deterministic FOSTER loaders without relying on cache insertion order."""
    train_transform, test_transform = get_transforms(image_size)

    full_train_dataset = PlanktonDataset(data_root, transform=train_transform, mode="train")
    canonicalize_dataset_order(full_train_dataset)

    total_len = len(full_train_dataset)
    val_len = int(total_len * 0.1)
    train_len = total_len - val_len

    train_dataset, val_dataset_raw = torch.utils.data.random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset = PlanktonDataset(data_root, transform=test_transform, mode="train")
    canonicalize_dataset_order(val_dataset)
    val_dataset = torch.utils.data.Subset(val_dataset, val_dataset_raw.indices)

    client_indices = partition_data(train_dataset, n_clients=n_clients, alpha=alpha, seed=partition_seed)
    client_loaders = []
    for client_id in range(n_clients):
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(
            client_dataset,
            **_build_loader_kwargs(
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            ),
        )
        client_loaders.append(client_loader)
        print(f"客户端 {client_id}: {len(client_dataset)} 样本")

    test_dataset = PlanktonDataset(data_root, transform=test_transform, mode="test")
    near_dataset = PlanktonDataset(data_root, transform=test_transform, mode="near_ood")
    far_dataset = PlanktonDataset(data_root, transform=test_transform, mode="far_ood")
    canonicalize_dataset_order(test_dataset)
    canonicalize_dataset_order(near_dataset)
    canonicalize_dataset_order(far_dataset)

    test_loader = DataLoader(
        test_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    near_loader = DataLoader(
        near_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    far_loader = DataLoader(
        far_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )

    print("FOSTER 确定性联邦数据加载器创建完成:")
    print(f"  - 客户端数量: {len(client_loaders)}")
    print(f"  - 训练集 (联邦): {len(train_dataset)} 样本")
    print(f"  - 验证集 (服务端): {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - Near-OOD: {len(near_dataset)} 样本")
    print(f"  - Far-OOD: {len(far_dataset)} 样本")
    return client_loaders, test_loader, near_loader, far_loader, val_loader
