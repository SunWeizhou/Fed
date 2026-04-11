"""Server-side coordination for the thesis-oriented FOOGD baseline."""

from __future__ import annotations

import copy

import torch

from methods.foster.foster_utils import model_state_cpu


class FOOGDServer:
    """FedAvg aggregation over both the backbone and the score model."""

    def __init__(
        self,
        global_model: torch.nn.Module,
        global_score_model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.global_model = global_model.to(device)
        self.global_score_model = global_score_model.to(device)
        self.device = device

    def global_states(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return model_state_cpu(self.global_model), model_state_cpu(self.global_score_model)

    def load_global_states(
        self,
        model_state: dict[str, torch.Tensor],
        score_state: dict[str, torch.Tensor],
    ) -> None:
        model_device_state = {key: value.to(self.device) for key, value in model_state.items()}
        score_device_state = {key: value.to(self.device) for key, value in score_state.items()}
        self.global_model.load_state_dict(model_device_state, strict=True)
        self.global_score_model.load_state_dict(score_device_state, strict=True)

    @staticmethod
    def aggregate_state_dicts(
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
                contribution = update[key].float() * (n_samples / total_samples)
                weighted = contribution if weighted is None else weighted + contribution
            aggregated[key] = weighted
        return aggregated

    def aggregate(
        self,
        model_updates: list[dict[str, torch.Tensor]],
        score_updates: list[dict[str, torch.Tensor]],
        sample_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        model_state = self.aggregate_state_dicts(model_updates, sample_sizes)
        score_state = self.aggregate_state_dicts(score_updates, sample_sizes)
        return model_state, score_state
