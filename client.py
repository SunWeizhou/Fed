#!/usr/bin/env python3
"""
联邦学习客户端模块。

当前实现仅保留 Fed-ViM 主线训练逻辑。
"""

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from config import OptimizerConfig, TrainingConfig, should_use_adamw
from utils.ood_utils import compute_empirical_alpha_local_stats


def create_grad_scaler(device, enabled):
    """Create a GradScaler compatible with both new and old AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, enabled):
    """Return an autocast context compatible with both new and old AMP APIs."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


class FLClient:
    """联邦学习客户端。"""

    def __init__(
        self,
        client_id,
        model,
        train_loader,
        device,
        alpha_loader=None,
        freeze_bn=True,
        base_lr=0.001,
        use_fedvim=False,
        weight_decay=1e-5,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.alpha_loader = alpha_loader if alpha_loader is not None else train_loader
        self.device = device
        self.freeze_bn = freeze_bn
        self.base_lr = base_lr
        self.use_fedvim = use_fedvim
        self.weight_decay = weight_decay
        self.effective_lr = base_lr

        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "model_type"):
            model_type = self.model.backbone.model_type
        else:
            model_type = "unknown"

        if should_use_adamw(model_type):
            self.effective_lr = base_lr
            self.optimizer_main = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.effective_lr,
                betas=OptimizerConfig.ADAMW_BETAS,
                eps=OptimizerConfig.ADAMW_EPS,
                weight_decay=self.weight_decay,
            )
            print(f"Client {client_id}: Using AdamW for {model_type} (lr={self.effective_lr})")
        else:
            if self.base_lr < 0.005:
                self.effective_lr = self.base_lr * TrainingConfig.SGD_LR_MULTIPLIER
                lr_note = f" (Auto-scaled x10: {self.base_lr} -> {self.effective_lr})"
            else:
                self.effective_lr = self.base_lr
                lr_note = f" (lr={self.effective_lr})"

            self.optimizer_main = torch.optim.SGD(
                self.model.parameters(),
                lr=self.effective_lr,
                momentum=TrainingConfig.SGD_MOMENTUM,
                weight_decay=self.weight_decay,
            )
            print(f"Client {client_id}: Using SGD for {model_type}{lr_note}")

        self.scaler = create_grad_scaler("cuda", enabled=self.device.type == "cuda")

    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        """Mixup 数据增强。"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_step(
        self,
        local_epochs=1,
        current_round=0,
        global_stats=None,
        total_rounds=50,
        warmup_rounds=5,
        accumulation_steps=1,
    ):
        """执行客户端本地训练，并在需要时回传二阶统计量。"""
        warmup_start_lr = 1e-5
        min_lr_ratio = 0.1
        min_lr = self.effective_lr * min_lr_ratio

        if current_round < warmup_rounds:
            target_lr = warmup_start_lr + (self.effective_lr - warmup_start_lr) * (current_round / warmup_rounds)
            phase = "Warmup"
        else:
            progress = (current_round - warmup_rounds) / max(1, total_rounds - warmup_rounds)
            lr_decay = 0.5 * (1 + np.cos(np.pi * progress))
            target_lr = min_lr + (self.effective_lr - min_lr) * lr_decay
            phase = "Cosine"

        for param_group in self.optimizer_main.param_groups:
            param_group["lr"] = target_lr

        if current_round % 5 == 0:
            eff_bs = self.train_loader.batch_size * accumulation_steps
            print(f"  [Round {current_round}] LR ({phase}): {target_lr:.6f} | AccSteps: {accumulation_steps} (EffBS: {eff_bs})")

        self.model.train()
        if self.freeze_bn:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

        total_loss = 0.0
        total_samples = 0
        epoch_log = {"cls": 0.0}
        client_vim_stats = None

        is_last_epoch = lambda epoch: epoch == local_epochs - 1

        for epoch in range(local_epochs):
            compute_stats_this_epoch = self.use_fedvim and is_last_epoch(epoch)
            if compute_stats_this_epoch:
                feature_dim = self.model.backbone.feature_dim
                sum_z = torch.zeros(feature_dim, device=self.device)
                sum_zzT = torch.zeros(feature_dim, feature_dim, device=self.device)
                stats_count = 0

            self.optimizer_main.zero_grad()

            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                with autocast_context("cuda", enabled=self.device.type == "cuda"):
                    use_mixup = np.random.random() < TrainingConfig.MIXUP_PROB
                    if use_mixup:
                        data_mixed, y_a, y_b, lam = self.mixup_data(data, targets, alpha=TrainingConfig.MIXUP_ALPHA)
                        logits, features = self.model(data_mixed)
                        classification_loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
                    else:
                        logits, features = self.model(data)
                        classification_loss = F.cross_entropy(
                            logits,
                            targets,
                            label_smoothing=TrainingConfig.LABEL_SMOOTHING,
                        )

                    loss_for_main = classification_loss / accumulation_steps

                self.scaler.scale(loss_for_main).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer_main)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP_NORM)
                    self.scaler.step(self.optimizer_main)
                    self.scaler.update()
                    self.optimizer_main.zero_grad()

                batch_size = data.size(0)
                total_loss += classification_loss.item() * batch_size
                total_samples += batch_size
                epoch_log["cls"] += classification_loss.item() * batch_size

                if compute_stats_this_epoch:
                    with torch.no_grad():
                        sum_z += features.sum(dim=0)
                        sum_zzT += torch.matmul(features.T, features)
                        stats_count += features.size(0)

            if len(self.train_loader) % accumulation_steps != 0:
                self.scaler.unscale_(self.optimizer_main)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer_main)
                self.scaler.update()
                self.optimizer_main.zero_grad()

            if compute_stats_this_epoch:
                client_vim_stats = {"sum_z": sum_z, "sum_zzT": sum_zzT, "count": stats_count}

        if total_samples > 0:
            avg_cls = epoch_log["cls"] / total_samples
            print(f"Client {self.client_id} - Epochs {local_epochs} - Avg Loss: {total_loss / total_samples:.4f} | Cls: {avg_cls:.4f}")

        if self.use_fedvim and client_vim_stats is None:
            client_vim_stats = self._compute_local_statistics()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        generic_params = self.get_generic_parameters()
        return generic_params, avg_loss, client_vim_stats

    def compute_vim_alpha_statistics(self, P, mu):
        """在客户端本地 ID 训练分片上计算经验 alpha 所需统计量。"""
        return compute_empirical_alpha_local_stats(
            model=self.model,
            loader=self.alpha_loader,
            P=P,
            mu=mu,
            device=self.device,
        )

    def get_generic_parameters(self):
        params = {}
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            clean_key = key.replace("_orig_mod.", "")
            params[f"model.{clean_key}"] = value.clone()
        return params

    def set_generic_parameters(self, generic_params):
        model_params = {}
        for key, value in generic_params.items():
            if key.startswith("model."):
                model_params[key.replace("model.", "")] = value
            else:
                model_params[key] = value
        self.model.load_state_dict(model_params, strict=False)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, _ = self.model(data)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * data.size(0)
                _, pred = torch.max(logits, 1)
                correct += (pred == targets).sum().item()
                total_samples += data.size(0)
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "acc": correct / total_samples if total_samples > 0 else 0.0,
        }

    def _compute_local_statistics(self):
        """Fed-ViM: 计算本地二阶统计量。"""
        self.model.eval()
        feature_dim = self.model.backbone.feature_dim

        sum_z = torch.zeros(feature_dim, device=self.device)
        sum_zzT = torch.zeros(feature_dim, feature_dim, device=self.device)
        count = 0

        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                _, features = self.model(data)
                sum_z += features.sum(dim=0)
                sum_zzT += torch.matmul(features.T, features)
                count += features.size(0)

        return {"sum_z": sum_z, "sum_zzT": sum_zzT, "count": count}
