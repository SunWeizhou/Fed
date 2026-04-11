"""Loss helpers used by the thesis-oriented FOOGD baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, kernel_type: str = "rbf", kernel_mul: float = 2.0, kernel_num: int = 5) -> None:
        super().__init__()
        self.kernel_num = int(kernel_num)
        self.kernel_mul = float(kernel_mul)
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / max(1, (n_samples ** 2 - n_samples))
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "linear":
            delta = source.float().mean(0) - target.float().mean(0)
            return delta.dot(delta.T)

        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target)
        xx = torch.mean(kernels[:batch_size, :batch_size])
        yy = torch.mean(kernels[batch_size:, batch_size:])
        xy = torch.mean(kernels[:batch_size, batch_size:])
        yx = torch.mean(kernels[batch_size:, :batch_size])
        return torch.mean(xx + yy - xy - yx)
