"""Server-side coordination for the isolated FOSTER baseline."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .foster_external_generator import ClassConditionalFeatureGenerator
from .foster_utils import model_state_cpu


class FOSTERServer:
    """Server that aggregates backbone updates and trains the central generator."""

    def __init__(
        self,
        global_model: torch.nn.Module,
        num_classes: int,
        feature_dim: int,
        device: torch.device,
        generator_lr: float,
        noise_dim: int,
        hidden_dim: int,
        generator_steps_per_round: int,
        oe_batch_size: int,
        label_smoothing: float = 0.0,
    ) -> None:
        self.global_model = global_model.to(device)
        self.device = device
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.generator_steps_per_round = int(generator_steps_per_round)
        self.oe_batch_size = int(oe_batch_size)
        self.label_smoothing = float(label_smoothing)
        self.generator = ClassConditionalFeatureGenerator(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(0.9, 0.999),
        )

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

    def train_generator(self) -> float:
        """One round of central generator updates using the global classifier."""
        self.global_model.eval()
        classifier = self.global_model.classifier
        self.generator.train()
        running_loss = 0.0

        for _ in range(self.generator_steps_per_round):
            labels = torch.randint(
                low=0,
                high=self.num_classes,
                size=(self.oe_batch_size,),
                device=self.device,
            )
            features = self.generator(labels)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()
            running_loss += loss.item()

        return running_loss / max(1, self.generator_steps_per_round)
