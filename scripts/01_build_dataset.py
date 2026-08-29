#!/usr/bin/env python
"""Build the aligned RAD-DINO + BioViL-T study-level cohort and cache it to disk."""

from __future__ import annotations

import argparse
import logging

from cxr_fusion.config import load_config
from cxr_fusion.data import (
    align_rad_and_bio,
    build_biovilt_cache_audit,
    build_study_base,
    load_biovilt_cache,
    save_aligned_cohort,
)
from cxr_fusion.representation import run_representation_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    out_dir = cfg.out_dir

    study = build_study_base(cfg.paths)
    study.support_audit.to_csv(out_dir / "support_devices_match_audit.csv", index=False)

    bio_cache = load_biovilt_cache(cfg.paths, study.study_base)
    build_biovilt_cache_audit(cfg.paths, bio_cache).to_csv(out_dir / "biovilt_cache_audit.csv", index=False)

    aligned = align_rad_and_bio(cfg, study, bio_cache)
    save_aligned_cohort(aligned, out_dir)

    probe, summary = run_representation_diagnostic(aligned)
    probe.to_csv(out_dir / "representation_linear_probe.csv", index=False)
    summary.to_csv(out_dir / "representation_linear_probe_summary.csv", index=False)
    rad_gain = float(summary.loc[summary.representation.eq("fixed RAD-DINO"), "mean_gain"].iloc[0])
    if rad_gain <= 0:
        logging.warning("Fixed RAD-DINO representation stays near prevalence-level AP.")

    logging.info("Aligned cohort cached under %s", out_dir)


if __name__ == "__main__":
    main()
