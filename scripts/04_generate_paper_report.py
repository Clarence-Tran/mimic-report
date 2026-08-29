#!/usr/bin/env python
"""Assemble the paper-candidate report tables from artifacts already on disk."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from cxr_fusion.config import load_config
from cxr_fusion.data import load_aligned_cohort
from cxr_fusion.report import generate_paper_report


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
    mode_ci = pd.read_csv(out_dir / f"{cfg.report.sample_level}_level_B0_B3_metric_95ci.csv")
    mode_seed_means = pd.read_csv(out_dir / f"{cfg.report.sample_level}_level_B0_B3_mean_metrics_per_seed.csv")
    paired = pd.read_csv(out_dir / "paired_tests_B0_B3_across_seeds.csv")

    generate_paper_report(cfg, aligned, results, weights, mode_ci, mode_seed_means, paired)


if __name__ == "__main__":
    main()
