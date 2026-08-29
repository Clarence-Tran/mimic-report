from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import FusionConfig

MODE_ORDER = ["B0_RADDINO", "B1_BIOVILT", "B2_EARLY_FUSION", "B3_HYBRID"]
REPORT_METRICS = ["mAP", "mean_AUROC", "mean_F1", "mean_P", "mean_PRROC"]
CONFIRMATORY_METRICS = ["mAP", "mean_AUROC"]


def scale(val_logits: np.ndarray, test_logits: np.ndarray):
    mean = val_logits.mean(axis=0, keepdims=True)
    std = val_logits.std(axis=0, keepdims=True).clip(1e-6)
    return (val_logits - mean) / std, (test_logits - mean) / std


def search_weights(
    labels: np.ndarray,
    rad_logits: np.ndarray,
    bio_logits: np.ndarray,
    early_logits: np.ndarray,
    cfg: FusionConfig,
) -> np.ndarray:
    weights = np.tile(np.array([0, 0, 1], dtype="float32"), (labels.shape[1], 1))
    grid = np.arange(0, 1 + 1e-9, cfg.grid_step)

    for label_index in range(labels.shape[1]):
        if len(np.unique(labels[:, label_index])) < 2:
            continue

        best_ap = average_precision_score(labels[:, label_index], early_logits[:, label_index])
        best_weights = weights[label_index].copy()

        for rad_weight in grid:
            for bio_weight in grid:
                early_weight = 1 - rad_weight - bio_weight
                if early_weight < cfg.min_fusion_weight - 1e-9:
                    continue
                score = (
                    rad_weight * rad_logits[:, label_index]
                    + bio_weight * bio_logits[:, label_index]
                    + early_weight * early_logits[:, label_index]
                )
                ap = average_precision_score(labels[:, label_index], score)
                if ap > best_ap + 1e-12:
                    best_ap = ap
                    best_weights = np.array([rad_weight, bio_weight, early_weight], dtype="float32")

        shrunk_weights = (1 - cfg.shrinkage) * np.array([0, 0, 1], dtype="float32") + cfg.shrinkage * best_weights
        late_score = (
            rad_logits[:, label_index] * shrunk_weights[0]
            + bio_logits[:, label_index] * shrunk_weights[1]
            + early_logits[:, label_index] * shrunk_weights[2]
        )
        gain = average_precision_score(labels[:, label_index], late_score) - average_precision_score(
            labels[:, label_index], early_logits[:, label_index]
        )
        if gain >= cfg.min_val_ap_gain:
            weights[label_index] = shrunk_weights

    return weights


def uniform_weights(n_labels: int) -> np.ndarray:
    return np.full((n_labels, 3), 1 / 3, dtype="float32")


def apply_weights(rad_logits, bio_logits, early_logits, weights: np.ndarray) -> np.ndarray:
    return rad_logits * weights[:, 0] + bio_logits * weights[:, 1] + early_logits * weights[:, 2]


def tune_thresholds(labels: np.ndarray, scores: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    thresholds = np.zeros(labels.shape[1], dtype="float32")
    for label_index in range(labels.shape[1]):
        precision, recall, candidates = precision_recall_curve(labels[:, label_index], scores[:, label_index])
        if not len(candidates):
            continue
        precision, recall = precision[:-1], recall[:-1]
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        valid = np.where(recall >= cfg.min_recall)[0]
        best_index = int(valid[np.argmax(f1[valid])]) if len(valid) else int(np.argmax(f1))
        thresholds[label_index] = candidates[best_index]
    return thresholds
