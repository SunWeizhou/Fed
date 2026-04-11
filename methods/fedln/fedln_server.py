"""Server-side FedAvg aggregation for the independent FedLN baseline."""

from __future__ import annotations

import copy

import torch

from methods.foster.foster_utils import model_state_cpu


class FedLNServer:
    """Simple FedAvg server used by the thesis FedLN baseline."""

    def __init__(self, global_model: torch.nn.Module, device: torch.device) -> None:
        self.global_model = global_model.to(device)
        self.device = device

    def global_state(self) -> dict[str, torch.Tensor]:
        return model_state_cpu(self.global_model)

    def load_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        target_device = next(self.global_model.parameters()).device
        device_state = {key: value.to(target_device) for key, value in state_dict.items()}
        self.global_model.load_state_dict(device_state, strict=True)

    def aggregate(
        self,
        updates: list[dict[str, torch.Tensor]],
        sample_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        total_samples = float(sum(sample_sizes))
        aggregated = copy.deepcopy(updates[0])
        for key in aggregated.keys():
            if "num_batches_tracked" in key:
                aggregated[key] = updates[0][key].clone()
                continue
            weighted = None
            for update, n_samples in zip(updates, sample_sizes):
                tensor = update[key].float()
                contribution = tensor * (n_samples / total_samples)
                weighted = contribution if weighted is None else weighted + contribution
            aggregated[key] = weighted
        return aggregated
