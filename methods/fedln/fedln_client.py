"""Client-side training logic for the independent FedLN baseline."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from config import TrainingConfig
from methods.foster.foster_utils import cosine_round_lr, model_state_cpu

from .fedln_losses import LogitNormLoss


def create_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


class FedLNClient:
    """Federated client that optimizes the LogitNorm objective."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader,
        device: torch.device,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        temperature: float = 0.01,
        freeze_bn: bool = False,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.freeze_bn = bool(freeze_bn)
        self.sample_count = len(train_loader.dataset)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.scaler = create_grad_scaler(device, enabled=device.type == "cuda")
        self.loss_fn = LogitNormLoss(temperature=temperature)

    def load_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        target_device = next(self.model.parameters()).device
        device_state = {key: value.to(target_device) for key, value in state_dict.items()}
        self.model.load_state_dict(device_state, strict=True)

    def train_round(
        self,
        current_round: int,
        total_rounds: int,
        local_epochs: int,
        warmup_rounds: int,
        min_lr_factor: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        lr = cosine_round_lr(
            base_lr=self.base_lr,
            current_round=current_round,
            total_rounds=total_rounds,
            warmup_rounds=warmup_rounds,
            min_lr_factor=min_lr_factor,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        if self.freeze_bn:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

        self.model.train()
        total_samples = 0
        total_loss = 0.0
        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = labels.size(0)
                self.optimizer.zero_grad()

                with autocast_context(self.device, enabled=self.device.type == "cuda"):
                    logits, _ = self.model(images)
                    loss = self.loss_fn(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_samples += batch_size
                total_loss += loss.item() * batch_size

        metrics = {
            "loss": total_loss / max(1, total_samples),
            "lr": lr,
        }
        return model_state_cpu(self.model), metrics
