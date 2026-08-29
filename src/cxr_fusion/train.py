from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config
from .data import AlignedCohort
from .datasets import DenseDataset
from .losses import ASL, mixup
from .models import DenseFeatureAutoencoder, FusionTokensPredictor, LatentDenoiser
from .utils import set_seed

logger = logging.getLogger(__name__)

Indices = tuple[np.ndarray, np.ndarray, np.ndarray]


def _make_loaders(cfg: Config, x_train, x_val, x_test, y_train, y_val, y_test, device: str):
    loader_cfg = cfg.data_loader
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        DenseDataset(x_train, y_train), loader_cfg.batch_size, shuffle=True,
        num_workers=loader_cfg.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        DenseDataset(x_val, y_val), loader_cfg.batch_size, shuffle=False,
        num_workers=loader_cfg.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        DenseDataset(x_test, y_test), loader_cfg.batch_size, shuffle=False,
        num_workers=loader_cfg.num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def collect(autoencoder, denoiser, predictor, loader, device: str):
    autoencoder.eval()
    denoiser.eval()
    predictor.eval()
    labels, logits = [], []
    for features, targets in loader:
        features = features.to(device)
        latent, _ = autoencoder(features)
        latent = denoiser(latent, torch.zeros(len(features), 1, device=device))
        labels.append(targets.numpy())
        logits.append(predictor(features, latent).cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def average_states(states: list[dict], component: str) -> dict:
    return {
        key: torch.stack([state[component][key].float() for state in states]).mean(dim=0)
        for key in states[0][component]
    }


def _load_cached(prediction_path, aligned: AlignedCohort, val_idx, test_idx):
    if not prediction_path.exists():
        return None
    cached_file = np.load(prediction_path)
    cached = {key: cached_file[key] for key in cached_file.files}
    if np.array_equal(cached["val_ids"], aligned.sample_ids[val_idx]) and np.array_equal(
        cached["test_ids"], aligned.sample_ids[test_idx]
    ):
        return cached
    logger.info("Ignoring stale cache: %s", prediction_path.name)
    return None


def train_component(
    cfg: Config, aligned: AlignedCohort, name: str, all_features: np.ndarray, seed: int, indices: Indices
) -> dict:
    out_dir = cfg.out_dir
    prediction_path = out_dir / f"internal_{cfg.alignment_version}_{name}_seed{seed}.npz"
    train_idx, val_idx, test_idx = indices

    cached = _load_cached(prediction_path, aligned, val_idx, test_idx)
    if cached is not None:
        return cached

    device = cfg.resolved_device
    set_seed(seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(all_features[train_idx]).astype("float32")
    x_val = scaler.transform(all_features[val_idx]).astype("float32")
    x_test = scaler.transform(all_features[test_idx]).astype("float32")

    train_loader, val_loader, test_loader = _make_loaders(
        cfg, x_train, x_val, x_test, aligned.y[train_idx], aligned.y[val_idx], aligned.y[test_idx], device
    )

    hidden_dim, latent_dim = cfg.model.hidden_dim, cfg.model.latent_dim
    autoencoder = DenseFeatureAutoencoder(x_train.shape[1], hidden_dim, latent_dim).to(device)
    denoiser = LatentDenoiser(hidden_dim, latent_dim).to(device)
    predictor = FusionTokensPredictor(
        x_train.shape[1], len(aligned.label_cols), hidden_dim, latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    for epoch in range(cfg.train.ae_epochs):
        autoencoder.train()
        losses = []
        for features, _ in train_loader:
            features = features.to(device)
            noisy_features = features + cfg.regularization.feature_noise_std * torch.randn_like(features)
            _, reconstruction = autoencoder(noisy_features)
            loss = F.smooth_l1_loss(reconstruction, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch in (0, cfg.train.ae_epochs - 1):
            logger.info("%s seed=%d AE epoch=%02d loss=%.5f", name, seed, epoch + 1, np.mean(losses))

    autoencoder.eval()
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    for epoch in range(cfg.train.gen_epochs):
        denoiser.train()
        for features, _ in train_loader:
            features = features.to(device)
            with torch.no_grad():
                latent, _ = autoencoder(features)
            sigma = torch.rand(len(features), 1, device=device) * 0.65
            noisy_latent = latent + sigma * torch.randn_like(latent)
            loss = F.mse_loss(denoiser(noisy_latent, sigma), latent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    positive_rate = aligned.y[train_idx].mean(axis=0).clip(1e-5, 1 - 1e-5)
    positive_weights = ((1 - positive_rate) / positive_rate).clip(1, 8)
    loss_fn = ASL(torch.tensor(positive_weights, device=device), cfg.regularization.label_smooth)
    trainable_parameters = list(predictor.parameters()) + list(denoiser.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=cfg.train.pred_lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-6
    )

    top_states: list[tuple[float, dict]] = []
    best_score = -1.0
    stale_epochs = 0

    for epoch in range(cfg.train.pred_epochs):
        predictor.train()
        denoiser.train()
        losses = []
        for features, targets in train_loader:
            features, targets = mixup(
                features.to(device), targets.to(device), cfg.regularization.mixup_alpha
            )
            features = features + cfg.regularization.feature_noise_std * torch.randn_like(features)
            with torch.no_grad():
                latent, _ = autoencoder(features)
            refined_latent = denoiser(latent, torch.zeros(len(features), 1, device=device))
            loss = loss_fn(predictor(features, refined_latent), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1)
            optimizer.step()
            losses.append(loss.item())

        val_labels, val_logits = collect(autoencoder, denoiser, predictor, val_loader, device)
        score = np.mean(
            [
                average_precision_score(val_labels[:, index], val_logits[:, index])
                for index in range(len(aligned.label_cols))
            ]
        )
        scheduler.step(score)
        logger.info(
            "%s seed=%d PRED epoch=%02d loss=%.5f val_mAP=%.5f lr=%.2e",
            name, seed, epoch + 1, np.mean(losses), score, optimizer.param_groups[0]["lr"],
        )

        state = {
            "ae": {key: value.cpu().clone() for key, value in autoencoder.state_dict().items()},
            "gen": {key: value.cpu().clone() for key, value in denoiser.state_dict().items()},
            "pred": {key: value.cpu().clone() for key, value in predictor.state_dict().items()},
        }
        top_states = sorted(top_states + [(score, state)], key=lambda item: item[0], reverse=True)[
            : cfg.train.topk
        ]

        if score > best_score:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.train.patience:
                break

    states = [state for _, state in top_states]
    chosen = copy.deepcopy(states[0])
    chosen["gen"] = average_states(states, "gen")
    chosen["pred"] = average_states(states, "pred")

    autoencoder.load_state_dict(chosen["ae"])
    denoiser.load_state_dict(chosen["gen"])
    predictor.load_state_dict(chosen["pred"])

    y_val, val_logits = collect(autoencoder, denoiser, predictor, val_loader, device)
    y_test, test_logits = collect(autoencoder, denoiser, predictor, test_loader, device)

    checkpoint = {
        "model": chosen,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "labels": aligned.label_cols,
        "alignment_version": cfg.alignment_version,
    }
    torch.save(checkpoint, out_dir / f"internal_{cfg.alignment_version}_{name}_seed{seed}.pt")

    payload = {
        "val_ids": aligned.sample_ids[val_idx],
        "test_ids": aligned.sample_ids[test_idx],
        "y_val": y_val,
        "y_test": y_test,
        "val_logits": val_logits,
        "test_logits": test_logits,
        "alignment_version": np.asarray(cfg.alignment_version),
    }
    np.savez_compressed(prediction_path, **payload)
    return payload
