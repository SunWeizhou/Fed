"""Client-side training logic for the thesis-oriented FOOGD baseline."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F

from config import TrainingConfig
from methods.foster.foster_utils import cosine_round_lr, model_state_cpu

from .foogd_ksd import compute_bandwidth, compute_ksd
from .foogd_losses import MMDLoss
from .foogd_sampling import anneal_langevin_dynamics, langevin_dynamics


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


class FOOGDClient:
    """Federated client with a backbone and a local score model."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        score_model: torch.nn.Module,
        train_loader,
        device: torch.device,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        num_classes: int,
        lambda_sag: float,
        lambda_sm3d: float,
        score_learning_rate: float,
        score_momentum: float,
        score_weight_decay: float,
        sample_steps: int,
        sample_eps: float,
        sigma_begin: float,
        sigma_end: float,
        anneal_power: float,
        noise_type: str = "gaussian",
        loss_type: str = "anneal_dsm",
        mmd_kernel_num: int = 5,
        freeze_bn: bool = False,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.score_model = score_model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.num_classes = int(num_classes)
        self.lambda_sag = float(lambda_sag)
        self.lambda_sm3d = float(lambda_sm3d)
        self.score_learning_rate = float(score_learning_rate)
        self.score_momentum = float(score_momentum)
        self.score_weight_decay = float(score_weight_decay)
        self.sample_steps = int(sample_steps)
        self.sample_eps = float(sample_eps)
        self.anneal_power = float(anneal_power)
        self.noise_type = noise_type
        self.loss_type = loss_type
        self.freeze_bn = bool(freeze_bn)
        self.sample_count = len(train_loader.dataset)
        self.mmd_loss = MMDLoss(kernel_num=mmd_kernel_num)
        self.sigmas = torch.logspace(
            start=math.log10(sigma_begin),
            end=math.log10(sigma_end),
            steps=self.num_classes,
            device=device,
            dtype=torch.float32,
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.score_optimizer = torch.optim.SGD(
            self.score_model.parameters(),
            lr=self.score_learning_rate,
            momentum=self.score_momentum,
            weight_decay=self.score_weight_decay,
        )
        self.scaler = create_grad_scaler(device, enabled=device.type == "cuda")

    def load_global_state(
        self,
        model_state: dict[str, torch.Tensor],
        score_state: dict[str, torch.Tensor],
    ) -> None:
        model_device_state = {key: value.to(self.device) for key, value in model_state.items()}
        score_device_state = {key: value.to(self.device) for key, value in score_state.items()}
        self.model.load_state_dict(model_device_state, strict=True)
        self.score_model.load_state_dict(score_device_state, strict=True)

    def get_random_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x, device=self.device)
        if self.noise_type == "radermacher":
            return noise.sign()
        if self.noise_type == "sphere":
            return noise / torch.norm(noise, dim=-1, keepdim=True).clamp_min(1e-8) * (x.shape[-1] ** 0.5)
        return noise

    def anneal_dsm_loss(self, x: torch.Tensor) -> torch.Tensor:
        labels = torch.randint(0, len(self.sigmas), (x.shape[0],), device=x.device)
        used_sigma = self.sigmas[labels].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = self.get_random_noise(x)
        x = x.requires_grad_()
        perturbed = x + noise * used_sigma
        scores = self.score_model(perturbed)
        loss = (
            torch.norm(scores.view(scores.shape[0], -1) + noise.view(noise.shape[0], -1), dim=-1) ** 2
            * used_sigma.squeeze() ** self.anneal_power
        )
        return loss.mean() / 2.0

    def score_matching_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss_type != "anneal_dsm":
            raise NotImplementedError(f"Unsupported FOOGD loss_type: {self.loss_type}")
        return self.anneal_dsm_loss(x)

    def sample_latents(self, noise: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "anneal_dsm":
            return anneal_langevin_dynamics(
                score_fn=self.score_model,
                x=noise,
                sigmas=self.sigmas,
                eps=self.sample_eps,
                n_steps=self.sample_steps,
            )
        return langevin_dynamics(
            score_fn=self.score_model,
            x=noise,
            eps=self.sample_eps,
            n_steps=self.sample_steps,
        )

    def train_round(
        self,
        current_round: int,
        total_rounds: int,
        local_epochs: int,
        warmup_rounds: int,
        min_lr_factor: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
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
        self.score_model.train()
        total_samples = 0
        total_backbone_loss = 0.0
        total_score_loss = 0.0
        total_ce_loss = 0.0
        total_ksd_loss = 0.0
        total_dsm_loss = 0.0
        total_mmd_loss = 0.0
        correct = 0

        for _ in range(local_epochs):
            for image_ori, image_aug, labels in self.train_loader:
                image_ori = image_ori.to(self.device, non_blocking=True)
                image_aug = image_aug.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = labels.size(0)
                if batch_size <= 1:
                    continue

                self.optimizer.zero_grad()
                data = torch.cat([image_ori, image_aug], dim=0)
                targets = torch.cat([labels, labels], dim=0)

                with autocast_context(self.device, enabled=self.device.type == "cuda"):
                    logits, latents = self.model(data)
                    logits = logits.float()
                    latents = latents.float()
                    ce_loss = F.cross_entropy(logits, targets)
                    latents_ori = latents[:batch_size]
                    latents_aug = latents[batch_size:]
                    ksd_loss = torch.zeros((), device=self.device)
                    if self.lambda_sag > 0.0 and batch_size > 1:
                        bandwidth = compute_bandwidth(latents_ori, latents_aug)
                        ksd_loss = compute_ksd(latents_ori, latents_aug, self.score_model, bandwidth=bandwidth)
                    backbone_loss = ce_loss + self.lambda_sag * ksd_loss

                self.scaler.scale(backbone_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    correct += (preds[:batch_size] == labels).sum().item()

                latents_detached = latents_ori.detach().float().requires_grad_(True)
                self.score_optimizer.zero_grad()
                dsm_loss = self.score_matching_loss(latents_detached)
                noise = self.get_random_noise(latents_detached).float()
                latents_gen = self.sample_latents(noise)
                mmd_loss = self.lambda_sm3d * self.mmd_loss(latents_detached, latents_gen)
                score_loss = dsm_loss + mmd_loss
                score_loss.backward()
                self.score_optimizer.step()

                total_samples += batch_size
                total_backbone_loss += backbone_loss.item() * batch_size
                total_score_loss += score_loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_ksd_loss += ksd_loss.item() * batch_size
                total_dsm_loss += dsm_loss.item() * batch_size
                total_mmd_loss += mmd_loss.item() * batch_size

        denom = max(1, total_samples)
        metrics = {
            "backbone_loss": total_backbone_loss / denom,
            "score_loss": total_score_loss / denom,
            "ce_loss": total_ce_loss / denom,
            "ksd_loss": total_ksd_loss / denom,
            "dsm_loss": total_dsm_loss / denom,
            "mmd_loss": total_mmd_loss / denom,
            "train_accuracy": correct / denom,
            "lr": lr,
        }
        return model_state_cpu(self.model), model_state_cpu(self.score_model), metrics
