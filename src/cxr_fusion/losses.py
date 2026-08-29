from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ASL(nn.Module):
    def __init__(self, positive_weights: torch.Tensor, label_smooth: float):
        super().__init__()
        self.positive_weights = positive_weights
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth
        probability = torch.sigmoid(logits)
        negative_probability = (1 - probability + 0.05).clamp(max=1)
        positive_loss = (
            targets * torch.log(probability.clamp_min(1e-8)) * self.positive_weights[None]
        )
        negative_loss = (1 - targets) * torch.log(negative_probability.clamp_min(1e-8))
        focal_weight = torch.pow(
            1 - (probability * targets + negative_probability * (1 - targets)),
            3 * (1 - targets),
        )
        return -((positive_loss + negative_loss) * focal_weight).mean()


def mixup(features: torch.Tensor, labels: torch.Tensor, alpha: float):
    beta = np.random.beta(alpha, alpha)
    permutation = torch.randperm(len(features), device=features.device)
    mixed_features = beta * features + (1 - beta) * features[permutation]
    mixed_labels = beta * labels + (1 - beta) * labels[permutation]
    return mixed_features, mixed_labels
