"""Configuration defaults for the independent FOSTER baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FOSTERConfig:
    warmup_rounds: int = 5
    loss_weight: float = 0.1
    generator_lr: float = 1e-4
    noise_dim: int = 128
    generator_hidden_dim: int = 512
    generator_steps_per_round: int = 50
    oe_batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 1e-4
    min_lr_factor: float = 0.1
    generator_label_smoothing: float = 0.0


def get_foster_defaults() -> dict[str, float | int]:
    cfg = FOSTERConfig()
    return {
        "warmup_rounds": cfg.warmup_rounds,
        "loss_weight": cfg.loss_weight,
        "generator_lr": cfg.generator_lr,
        "noise_dim": cfg.noise_dim,
        "generator_hidden_dim": cfg.generator_hidden_dim,
        "generator_steps_per_round": cfg.generator_steps_per_round,
        "oe_batch_size": cfg.oe_batch_size,
        "momentum": cfg.momentum,
        "weight_decay": cfg.weight_decay,
        "min_lr_factor": cfg.min_lr_factor,
        "generator_label_smoothing": cfg.generator_label_smoothing,
    }
