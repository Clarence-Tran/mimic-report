from __future__ import annotations

import torch
from torch import nn


class DenseFeatureAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features: torch.Tensor):
        latent = self.encoder(features)
        return latent, self.decoder(latent)


class LatentDenoiser(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, sigma], dim=1))


class FusionTokensPredictor(nn.Module):
    def __init__(self, input_dim: int, n_labels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 8, hidden_dim * 2, 0.30, batch_first=True, activation="gelu"
        )
        self.fusion = nn.TransformerEncoder(layer, 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        feature_token = self.feature_projection(features)
        latent_token = self.latent_projection(latent)
        tokens = torch.stack([feature_token, latent_token], dim=1)
        pooled = self.fusion(tokens).mean(dim=1)
        gate = self.gate(torch.cat([feature_token, latent_token], dim=1)).unsqueeze(-1)
        gated = (tokens * gate).sum(dim=1)
        return self.head(0.5 * pooled + 0.5 * gated)
