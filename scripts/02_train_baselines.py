#!/usr/bin/env python
"""Train the three branches (RAD-DINO, BioViL-T, early-fusion concat) per seed and evaluate B0-B3."""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cxr_fusion.config import load_config
from cxr_fusion.data import load_aligned_cohort
from cxr_fusion.fusion import MODE_ORDER, apply_weights, scale, tune_thresholds, uniform_weights
from cxr_fusion.splits import split_for_seed
from cxr_fusion.train import train_component


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_dir = cfg.out_dir
    aligned = load_aligned_cohort(out_dir)

    result_rows = []
    weight_rows = []

    for seed in cfg.seeds:
        split_indices = split_for_seed(aligned, cfg.split, seed)
        np.savez(
            out_dir / f"split_seed{seed}.npz",
            train=split_indices[0], val=split_indices[1], test=split_indices[2],
        )

        early_features = np.concatenate([aligned.x_new, aligned.x_bio], axis=1)
        components = {
            "RAD_DINO": train_component(cfg, aligned, "RAD_DINO", aligned.x_new, seed, split_indices),
            "BIO": train_component(cfg, aligned, "BIO", aligned.x_bio, seed, split_indices),
            "EARLY": train_component(cfg, aligned, "EARLY", early_features, seed, split_indices),
        }
        del early_features
        gc.collect()

        reference = components["RAD_DINO"]
        val_scaled, test_scaled = {}, {}
        for name, component in components.items():
            val_scaled[name], test_scaled[name] = scale(component["val_logits"], component["test_logits"])

        fusion_weights = uniform_weights(reference["y_val"].shape[1])
        val_modes = {
            "B0_RADDINO": val_scaled["RAD_DINO"],
            "B1_BIOVILT": val_scaled["BIO"],
            "B2_EARLY_FUSION": val_scaled["EARLY"],
            "B3_HYBRID": apply_weights(
                val_scaled["RAD_DINO"], val_scaled["BIO"], val_scaled["EARLY"], fusion_weights
            ),
        }
        test_modes = {
            "B0_RADDINO": test_scaled["RAD_DINO"],
            "B1_BIOVILT": test_scaled["BIO"],
            "B2_EARLY_FUSION": test_scaled["EARLY"],
            "B3_HYBRID": apply_weights(
                test_scaled["RAD_DINO"], test_scaled["BIO"], test_scaled["EARLY"], fusion_weights
            ),
        }

        mode_thresholds, mode_predictions = {}, {}
        for mode in MODE_ORDER:
            thresholds = tune_thresholds(reference["y_val"], val_modes[mode], cfg.fusion)
            scores = test_modes[mode]
            predictions = scores >= thresholds[None]
            mode_thresholds[mode] = thresholds
            mode_predictions[mode] = predictions

            for label_index, label in enumerate(aligned.label_names):
                y_true = reference["y_test"][:, label_index]
                score = scores[:, label_index]
                curve_precision, curve_recall, _ = precision_recall_curve(y_true, score)
                result_rows.append(
                    {
                        "feature_mode": mode,
                        "seed": seed,
                        "label": label,
                        "test_n": len(y_true),
                        "test_pos_n": int(y_true.sum()),
                        "prevalence": y_true.mean(),
                        "threshold": thresholds[label_index],
                        "AP": average_precision_score(y_true, score),
                        "AUROC": roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan,
                        "PRROC": auc(curve_recall[::-1], curve_precision[::-1]),
                        "Precision": precision_score(y_true, predictions[:, label_index], zero_division=0),
                        "Recall": recall_score(y_true, predictions[:, label_index], zero_division=0),
                        "F1": f1_score(y_true, predictions[:, label_index], zero_division=0),
                        "F2": fbeta_score(y_true, predictions[:, label_index], beta=2, zero_division=0),
                    }
                )

        for label_index, label in enumerate(aligned.label_names):
            weight_rows.append(
                {
                    "seed": seed,
                    "label": label,
                    "w_RAD_DINO_LATENT": fusion_weights[label_index, 0],
                    "w_BioViL_T_LATENT": fusion_weights[label_index, 1],
                    "w_EARLY": fusion_weights[label_index, 2],
                }
            )

        b3_scores = test_modes["B3_HYBRID"]
        b3_thresholds = mode_thresholds["B3_HYBRID"]
        b3_predictions = mode_predictions["B3_HYBRID"]
        prediction_table = aligned.sample_meta.set_index("sample_id").loc[reference["test_ids"]].reset_index()
        prediction_table["seed"] = seed
        for label_index, label in enumerate(aligned.label_names):
            prediction_table[f"true_{label}"] = reference["y_test"][:, label_index].astype("int8")
            prediction_table[f"score_{label}"] = b3_scores[:, label_index].astype("float32")
            prediction_table[f"pred_{label}"] = b3_predictions[:, label_index].astype("int8")
        prediction_table.to_parquet(
            out_dir / f"b3_{cfg.report.sample_level}_level_predictions_seed{seed}.parquet", index=False
        )
        np.savez_compressed(
            out_dir / f"b3_hybrid_fusion_seed{seed}.npz",
            test_ids=reference["test_ids"],
            y_test=reference["y_test"],
            test_score=b3_scores,
            weights=fusion_weights,
            thresholds=b3_thresholds,
        )
        logging.info("Seed %d done.", seed)

    results = pd.DataFrame(result_rows)
    weights = pd.DataFrame(weight_rows)
    results.to_csv(out_dir / "B0_B3_results.csv", index=False)
    weights.to_csv(out_dir / "B3_uniform_weights.csv", index=False)
    logging.info("Wrote B0_B3_results.csv (%d rows) and B3_uniform_weights.csv", len(results))


if __name__ == "__main__":
    main()
