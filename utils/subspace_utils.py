#!/usr/bin/env python3
"""Helpers for selecting ViM/Fed-ViM subspace dimensions."""

from __future__ import annotations


def select_vim_paper_k(feature_dim: int, num_classes: int) -> int:
    """
    Reproduce the heuristic principal-subspace dimension rule described in ViM.

    Rules used here:
    - if N > 1500: D = 1000
    - elif N <= C: D = N/2 (middle of [N/3, 2N/3])
    - else: D = 512
    """
    feature_dim = int(feature_dim)
    num_classes = int(num_classes)

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    if feature_dim > 1500:
        k = 1000
    elif feature_dim <= num_classes:
        k = feature_dim // 2
    else:
        k = 512

    return max(1, min(k, feature_dim))
