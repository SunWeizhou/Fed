"""Client-side training logic for the isolated FOSTER baseline."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from config import TrainingConfig

from .foster_losses import classifier_ce_loss, oe_loss
from .foster_utils import cosine_round_lr, infer_observed_classes, model_state_cpu


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


class FOSTERClient:
    """Federated client used only by the thesis FOSTER branch."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader,
        device: torch.device,
        num_classes: int,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        freeze_bn: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.num_classes = int(num_classes)
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.freeze_bn = bool(freeze_bn)
        self.label_smoothing = float(label_smoothing)
        self.observed_classes = infer_observed_classes(train_loader)
        observed = set(self.observed_classes)
        self.external_classes = [cls for cls in range(self.num_classes) if cls not in observed]
        self.sample_count = len(train_loader.dataset)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.scaler = create_grad_scaler(device, enabled=device.type == "cuda")

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
        generator: torch.nn.Module | None,
        oe_batch_size: int,
        loss_weight: float,
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
        total_cls_loss = 0.0
        total_oe_loss = 0.0
        oe_steps = 0
        use_generator = generator is not None and current_round >= warmup_rounds and len(self.external_classes) > 0

        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = labels.size(0)
                self.optimizer.zero_grad()

                with autocast_context(self.device, enabled=self.device.type == "cuda"):
                    logits, _ = self.model(images)
                    cls_loss = classifier_ce_loss(
                        logits,
                        labels,
                        label_smoothing=self.label_smoothing,
                    )
                    aux_oe_loss = torch.zeros((), device=self.device)
                    if use_generator:
                        sampled_labels = torch.tensor(
                            [self.external_classes[idx] for idx in torch.randint(
                                low=0,
                                high=len(self.external_classes),
                                size=(oe_batch_size,),
                                device=self.device,
                            ).tolist()],
                            device=self.device,
                            dtype=torch.long,
                        )
                        with torch.no_grad():
                            fake_features = generator(sampled_labels).detach()
                        fake_logits = self.model.classifier(fake_features)
                        aux_oe_loss = oe_loss(fake_logits)
                    loss = cls_loss + loss_weight * aux_oe_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_cls_loss += cls_loss.item() * batch_size
                total_oe_loss += aux_oe_loss.item() * batch_size
                if use_generator:
                    oe_steps += 1

        denom = max(1, total_samples)
        metrics = {
            "loss": total_loss / denom,
            "cls_loss": total_cls_loss / denom,
            "oe_loss": total_oe_loss / denom,
            "lr": lr,
            "use_generator": float(use_generator),
            "oe_steps": float(oe_steps),
        }
        return model_state_cpu(self.model), metrics
