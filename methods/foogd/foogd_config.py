"""Configuration defaults for the independent FOOGD baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FOOGDConfig:
    fourier_mix_alpha: float = 1.0
    lambda_sag: float = 0.05
    lambda_sm3d: float = 0.5
    score_learning_rate: float = 0.01
    score_momentum: float = 0.0
    score_weight_decay: float = 0.0
    score_hidden_dim: int = 1024
    sample_steps: int = 10
    sample_eps: float = 0.01
    sigma_begin: float = 0.01
    sigma_end: float = 1.0
    anneal_power: float = 2.0
    noise_type: str = "gaussian"
    loss_type: str = "anneal_dsm"
    mmd_kernel_num: int = 5
    min_lr_factor: float = 0.1
    score_eval: str = "sm"


def get_foogd_defaults() -> dict[str, float | int | str]:
    cfg = FOOGDConfig()
    return {
        "fourier_mix_alpha": cfg.fourier_mix_alpha,
        "lambda_sag": cfg.lambda_sag,
        "lambda_sm3d": cfg.lambda_sm3d,
        "score_learning_rate": cfg.score_learning_rate,
        "score_momentum": cfg.score_momentum,
        "score_weight_decay": cfg.score_weight_decay,
        "score_hidden_dim": cfg.score_hidden_dim,
        "sample_steps": cfg.sample_steps,
        "sample_eps": cfg.sample_eps,
        "sigma_begin": cfg.sigma_begin,
        "sigma_end": cfg.sigma_end,
        "anneal_power": cfg.anneal_power,
        "noise_type": cfg.noise_type,
        "loss_type": cfg.loss_type,
        "mmd_kernel_num": cfg.mmd_kernel_num,
        "min_lr_factor": cfg.min_lr_factor,
        "score_eval": cfg.score_eval,
    }
