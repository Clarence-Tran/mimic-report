#!/usr/bin/env python
"""Build the B3 heatmap, ROC curves, bar chart, CI/paired tests, and split tables."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from cxr_fusion.config import load_config
from cxr_fusion.data import load_aligned_cohort
from cxr_fusion.evaluate import b3_label_metrics, compute_mode_ci, compute_mode_seed_means, paired_tests, weight_summary_tables
from cxr_fusion.figures import plot_b3_heatmap, plot_mode_comparison_bar, plot_roc_curves
from cxr_fusion.report import hyperparameter_table, split_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_dir = cfg.out_dir
    aligned = load_aligned_cohort(out_dir)

    results = pd.read_csv(out_dir / "B0_B3_results.csv")
    weights = pd.read_csv(out_dir / "B3_uniform_weights.csv")

    b3_results, b3_mean, b3_std = b3_label_metrics(results, aligned.label_names)
    b3_mean.to_csv(out_dir / "B3_14label_mean_metrics.csv")
    b3_std.to_csv(out_dir / "B3_14label_std_metrics.csv")
    plot_b3_heatmap(b3_mean, b3_std, out_dir)

    weight_summary, best_seed, best_seed_ap, best_seed_weights = weight_summary_tables(
        weights, b3_results, aligned.label_names
    )
    weight_summary.to_csv(out_dir / "B3_weight_summary.csv")
    best_seed_weights.to_csv(out_dir / f"B3_best_seed_{best_seed}_weights.csv")
    logging.info("Best seed by mAP: %d (mAP=%.6f)", best_seed, best_seed_ap)

    mode_seed_means = compute_mode_seed_means(results)
    mode_ci = compute_mode_ci(mode_seed_means, cfg.report.ci_level)
    tests = paired_tests(mode_seed_means, ci_level=cfg.report.ci_level)
    mode_seed_means.to_csv(out_dir / f"{cfg.report.sample_level}_level_B0_B3_mean_metrics_per_seed.csv", index=False)
    mode_ci.to_csv(out_dir / f"{cfg.report.sample_level}_level_B0_B3_metric_95ci.csv", index=False)
    tests.to_csv(out_dir / "paired_tests_B0_B3_across_seeds.csv", index=False)

    b3_outputs = {}
    for seed in cfg.seeds:
        npz = np.load(out_dir / f"b3_hybrid_fusion_seed{seed}.npz")
        b3_outputs[seed] = {"y": npz["y_test"], "score": npz["test_score"]}
    plot_roc_curves(b3_outputs, aligned.label_names, out_dir)

    plot_mode_comparison_bar(mode_seed_means, ["B0_RADDINO", "B1_BIOVILT", "B2_EARLY_FUSION", "B3_HYBRID"], out_dir)

    hyperparameter_table(cfg).to_csv(out_dir / "hyperparameter_table.csv", index=False)
    split_table, split_size_table, split_summary = split_tables(aligned, cfg)
    split_table.to_csv(out_dir / "train_validation_test_by_seed.csv", index=False)
    split_summary.to_csv(out_dir / "train_validation_test_summary.csv", index=False)

    logging.info("Figures and tables written to %s", out_dir)


if __name__ == "__main__":
    main()
