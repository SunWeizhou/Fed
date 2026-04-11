"""Minimal KSD utilities adapted from the official FOOGD implementation."""

from __future__ import annotations

import math

import torch


def trace_se_kernel_multi(sample1: torch.Tensor, sample2: torch.Tensor, bandwidth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    diff = sample1 - sample2
    return K * (
        2.0 / (bandwidth ** 2 + 1e-9) * sample1.shape[-1]
        - 4.0 / (bandwidth ** 4 + 1e-9) * torch.sum(diff * diff, dim=-1)
    )


def se_kernel_multi(sample1: torch.Tensor, sample2: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    if len(sample1.shape) == 4:
        bandwidth = bandwidth.unsqueeze(-1).unsqueeze(-1)
    sample_diff = sample1 - sample2
    norm_sample = torch.norm(sample_diff, dim=-1) ** 2
    return torch.exp(-norm_sample / (bandwidth ** 2 + 1e-9))


def median_heuristic(sample1: torch.Tensor, sample2: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        G = torch.sum(sample1 * sample1, dim=-1)
        H = torch.sum(sample2 * sample2, dim=-1)
        dist = G.unsqueeze(-2) + H.unsqueeze(-1) - 2 * sample2.matmul(torch.transpose(sample1, -1, -2))
        dist = (dist - torch.tril(dist)).view(-1)
        positive = dist[dist > 0.0]
        if positive.numel() == 0:
            return torch.tensor(1.0, device=sample1.device, dtype=sample1.dtype)
        return torch.median(positive).clone().detach()


def compute_ksd(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    score_func,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    samples1_exp = samples1.unsqueeze(1).repeat(1, samples2.shape[0], 1)
    samples2_exp = samples2.unsqueeze(0).repeat(samples1.shape[0], 1, 1)

    score_sample1 = score_func(samples1)
    score_sample2 = score_func(samples2)

    score_sample1_exp = score_sample1.unsqueeze(1)
    score_sample2_exp = score_sample2.unsqueeze(0)

    K = se_kernel_multi(samples1_exp, samples2_exp, bandwidth=bandwidth)
    term1 = K * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)

    grad_K_2 = torch.autograd.grad(torch.sum(K), samples2_exp, retain_graph=True, create_graph=False)[0]
    term2 = torch.sum(score_sample1_exp * grad_K_2, dim=-1)

    K = se_kernel_multi(samples1_exp, samples2_exp, bandwidth=bandwidth)
    grad_K_1 = torch.autograd.grad(torch.sum(K), samples1_exp, retain_graph=True, create_graph=False)[0]
    term3 = torch.sum(score_sample2_exp * grad_K_1, dim=-1)

    K = se_kernel_multi(samples1_exp, samples2_exp, bandwidth=bandwidth)
    term4 = trace_se_kernel_multi(samples1_exp, samples2_exp, bandwidth=bandwidth, K=K)

    divergence = torch.sum(term1 + term2 + term3 + term4)
    return divergence / max(1, samples1.shape[0] * samples2.shape[0])


def compute_bandwidth(samples1: torch.Tensor, samples2: torch.Tensor) -> torch.Tensor:
    split_size = max(2, samples1.shape[0])
    median_dist = median_heuristic(samples1, samples2)
    factor = 2.0 * math.sqrt(1.0 / (2.0 * math.log(split_size + 1.0)))
    return factor * torch.pow(0.5 * median_dist, 0.5)
