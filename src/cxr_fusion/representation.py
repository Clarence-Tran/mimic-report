from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import AlignedCohort

PROBE_SEED = 42


def _probe_split(aligned: AlignedCohort, probe_seed: int = PROBE_SEED, max_n: int = 40000):
    rng = np.random.default_rng(probe_seed)
    probe_idx = rng.choice(len(aligned.y), min(max_n, len(aligned.y)), replace=False)
    strata = np.clip(aligned.y[probe_idx].sum(axis=1), 0, 4).astype(int)
    try:
        train_idx, val_idx = train_test_split(
            probe_idx, test_size=0.25, random_state=probe_seed, stratify=strata
        )
    except ValueError:
        train_idx, val_idx = train_test_split(probe_idx, test_size=0.25, random_state=probe_seed)
    return train_idx, val_idx


def probe_representation(
    features: np.ndarray,
    name: str,
    aligned: AlignedCohort,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    probe_seed: int = PROBE_SEED,
) -> pd.DataFrame:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_idx]).astype("float32")
    val_features = scaler.transform(features[val_idx]).astype("float32")

    average_precisions = []
    for label_index in range(len(aligned.label_cols)):
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=200,
            tol=1e-4,
            average=True,
            random_state=probe_seed + label_index,
        )
        classifier.fit(train_features, aligned.y[train_idx, label_index])
        scores = classifier.decision_function(val_features)
        average_precisions.append(average_precision_score(aligned.y[val_idx, label_index], scores))

    return pd.DataFrame(
        {
            "label": aligned.label_names,
            "representation": name,
            "linear_probe_AP": average_precisions,
            "val_prevalence": aligned.y[val_idx].mean(axis=0),
        }
    )


def run_representation_diagnostic(
    aligned: AlignedCohort, probe_seed: int = PROBE_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, val_idx = _probe_split(aligned, probe_seed)

    representation_probe = pd.concat(
        [
            probe_representation(aligned.x_new, "fixed RAD-DINO", aligned, train_idx, val_idx, probe_seed),
            probe_representation(aligned.x_bio, "cached BioViL-T", aligned, train_idx, val_idx, probe_seed),
        ],
        ignore_index=True,
    )
    representation_probe["gain_over_prevalence"] = (
        representation_probe.linear_probe_AP - representation_probe.val_prevalence
    )
    summary = (
        representation_probe.groupby("representation", sort=False)
        .agg(
            linear_mAP=("linear_probe_AP", "mean"),
            prevalence_mAP=("val_prevalence", "mean"),
            mean_gain=("gain_over_prevalence", "mean"),
        )
        .reset_index()
    )
    return representation_probe, summary
