from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .data import AlignedCohort
from .evaluate import b3_label_metrics, weight_summary_tables
from .fusion import MODE_ORDER
from .splits import split_for_seed

logger = logging.getLogger(__name__)

PAPER_MODE_NAMES = {
    "B0_RADDINO": "B0 RADDINO",
    "B1_BIOVILT": "B1 BioViL-T",
    "B2_EARLY_FUSION": "B2 Early fusion",
    "B3_HYBRID": "B3 Hybrid fusion",
}


def hyperparameter_table(cfg: Config) -> pd.DataFrame:
    rows = [
        ("Evaluation", "Seeds", ", ".join(map(str, cfg.seeds))),
        ("Evaluation", "Confidence interval", cfg.report.ci_level),
        ("Data split", "Test fraction", cfg.split.test_size),
        ("Data split", "Validation fraction of temporary set", cfg.split.val_size_from_temp),
        ("Data split", "Effective train/validation/test", "70% / 10% / 20%"),
        ("Data loader", "Batch size", cfg.data_loader.batch_size),
        ("Data loader", "Workers", cfg.data_loader.num_workers),
        ("Architecture", "Hidden dimension", cfg.model.hidden_dim),
        ("Architecture", "Latent dimension", cfg.model.latent_dim),
        ("Training", "Autoencoder epochs", cfg.train.ae_epochs),
        ("Training", "Denoiser epochs", cfg.train.gen_epochs),
        ("Training", "Predictor maximum epochs", cfg.train.pred_epochs),
        ("Training", "Early-stopping patience", cfg.train.patience),
        ("Training", "Top checkpoints averaged", cfg.train.topk),
        ("Optimization", "Autoencoder/denoiser learning rate", cfg.train.lr),
        ("Optimization", "Predictor learning rate", cfg.train.pred_lr),
        ("Optimization", "Weight decay", cfg.train.weight_decay),
        ("Regularization", "Feature noise SD", cfg.regularization.feature_noise_std),
        ("Regularization", "Mixup alpha", cfg.regularization.mixup_alpha),
        ("Regularization", "Label smoothing", cfg.regularization.label_smooth),
        ("Hybrid fusion", "Branch weights", "1/3 / 1/3 / 1/3"),
        ("Thresholding", "Minimum recall", cfg.fusion.min_recall),
    ]
    return pd.DataFrame(rows, columns=["Section", "Hyperparameter", "Value"])


def split_tables(aligned: AlignedCohort, cfg: Config):
    rows = []
    for seed in cfg.seeds:
        train_idx, val_idx, test_idx = split_for_seed(aligned, cfg.split, seed)
        for split_name, indices in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
            split_labels = aligned.y[indices]
            rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "n_studies": len(indices),
                    "cohort_percent": 100 * len(indices) / len(aligned.y),
                    "positive_label_instances": int(split_labels.sum()),
                    "mean_label_prevalence": float(split_labels.mean()),
                }
            )

    split_table = pd.DataFrame(rows)
    order = ["Train", "Validation", "Test"]
    split_table["split"] = pd.Categorical(split_table["split"], categories=order, ordered=True)
    split_table = split_table.sort_values(["seed", "split"]).reset_index(drop=True)

    split_size_table = (
        split_table.pivot(index="seed", columns="split", values="n_studies").reindex(columns=order).reset_index()
    )
    split_summary = (
        split_table.groupby("split", observed=True)
        .agg(
            n_studies=("n_studies", "first"),
            cohort_percent=("cohort_percent", "first"),
            positive_instances_mean=("positive_label_instances", "mean"),
            positive_instances_sd=("positive_label_instances", "std"),
            label_prevalence_mean=("mean_label_prevalence", "mean"),
            label_prevalence_sd=("mean_label_prevalence", "std"),
        )
        .reindex(order)
        .reset_index()
    )
    return split_table, split_size_table, split_summary


def _save(df: pd.DataFrame, path: Path, title: str) -> None:
    df.to_csv(path, index=False)
    logger.info("%s -> %s (%d rows)", title, path.name, len(df))


def generate_paper_report(
    cfg: Config,
    aligned: AlignedCohort,
    results: pd.DataFrame,
    weights: pd.DataFrame,
    mode_ci: pd.DataFrame,
    mode_seed_means: pd.DataFrame,
    paired: pd.DataFrame,
) -> Path:
    report_dir = cfg.out_dir / "paper_candidate_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    b3_results, _, _ = b3_label_metrics(results, aligned.label_names)
    _, best_seed, best_seed_map, _ = weight_summary_tables(weights, b3_results, aligned.label_names)

    cohort_overview = pd.DataFrame(
        [
            ("Analysis level", cfg.report.sample_level),
            ("Aligned studies", f"{len(aligned.y):,}"),
            ("Disease labels", len(aligned.label_names)),
            ("RAD-DINO dimensions", aligned.x_new.shape[1]),
            ("BioViL-T dimensions", aligned.x_bio.shape[1]),
            ("Random seeds", ", ".join(map(str, cfg.seeds))),
            ("Number of seeds", len(cfg.seeds)),
            ("Train / validation / test", "70% / 10% / 20%"),
            ("Cohort coverage", f"{aligned.coverage:.2%}"),
            ("Cohort hash", aligned.cohort_hash),
            ("Alignment version", cfg.alignment_version),
        ],
        columns=["Item", "Value"],
    )

    label_distribution = pd.DataFrame(
        {
            "Label": aligned.label_names,
            "Positive studies": aligned.y.sum(axis=0).astype(int),
            "Prevalence": aligned.y.mean(axis=0),
        }
    )

    configuration_table = pd.DataFrame(
        [
            ("B0", "RADDINO", "Branch model on standardized 768-d RAD-DINO features, evaluated standalone"),
            ("B1", "BioViL-T", "Branch model on standardized 128-d BioViL-T features, evaluated standalone"),
            ("B2", "Early fusion", "Branch model on the standardized 896-d RAD-DINO+BioViL-T concatenation"),
            ("B3", "Hybrid fusion", "Uniform (1/3, 1/3, 1/3) fusion of the normalized B0, B1, B2 logits"),
        ],
        columns=["Configuration", "Name", "Definition"],
    )

    train_idx, val_idx, test_idx = split_for_seed(aligned, cfg.split, cfg.seeds[0])
    paper_split_table = pd.DataFrame(
        [
            ("Train", f"{100 * len(train_idx) / len(aligned.y):.0f}%", len(train_idx)),
            ("Validation", f"{100 * len(val_idx) / len(aligned.y):.0f}%", len(val_idx)),
            ("Test", f"{100 * len(test_idx) / len(aligned.y):.0f}%", len(test_idx)),
        ],
        columns=["Split", "Ratio", "Samples"],
    )

    paper_hyperparameters = pd.DataFrame(
        [
            ("Batch size", cfg.data_loader.batch_size),
            ("Hidden / latent dimensions", f"{cfg.model.hidden_dim} / {cfg.model.latent_dim}"),
            (
                "AE / denoiser / predictor epochs",
                f"{cfg.train.ae_epochs} / {cfg.train.gen_epochs} / {cfg.train.pred_epochs}",
            ),
            ("Early-stopping patience", cfg.train.patience),
            ("Top checkpoints averaged", cfg.train.topk),
            ("AE-denoiser / predictor learning rates", f"{cfg.train.lr:.1e} / {cfg.train.pred_lr:.1e}"),
            ("Weight decay", f"{cfg.train.weight_decay:.1e}"),
            (
                "Noise SD / mixup alpha / label smoothing",
                f"{cfg.regularization.feature_noise_std:g} / {cfg.regularization.mixup_alpha:g} / "
                f"{cfg.regularization.label_smooth:g}",
            ),
            ("Hybrid-fusion branch weights", "1/3 / 1/3 / 1/3"),
            ("Minimum recall for thresholding", cfg.fusion.min_recall),
        ],
        columns=["Parameter", "Setting"],
    )

    overall_ci = mode_ci.copy()
    overall_ci["Configuration"] = overall_ci["feature_mode"].map(PAPER_MODE_NAMES)
    overall_ci["Estimate (95% CI)"] = overall_ci.apply(
        lambda row: f"{row['mean']:.3f} ({row['CI_low']:.3f}, {row['CI_high']:.3f})", axis=1
    )
    overall_results = (
        overall_ci.pivot(index="Configuration", columns="metric", values="Estimate (95% CI)")
        .reindex([PAPER_MODE_NAMES[mode] for mode in MODE_ORDER])
        .reset_index()[["Configuration", "mAP", "mean_AUROC", "mean_F1", "mean_PRROC", "mean_P"]]
        .rename(columns={"mean_AUROC": "AUROC", "mean_F1": "Macro F1", "mean_PRROC": "PRROC", "mean_P": "Precision"})
    )

    seed_results = mode_seed_means.copy()
    seed_results["Configuration"] = seed_results["feature_mode"].map(PAPER_MODE_NAMES)
    seed_results = seed_results[
        ["Configuration", "seed", "mAP", "mean_AUROC", "mean_F1", "mean_PRROC", "mean_P"]
    ].rename(
        columns={
            "seed": "Seed",
            "mean_AUROC": "AUROC",
            "mean_F1": "Macro F1",
            "mean_PRROC": "PRROC",
            "mean_P": "Precision",
        }
    )

    b3_paper_results = (
        b3_results.groupby("label", sort=False)
        .agg(
            Test_samples=("test_n", "mean"),
            Test_positives=("test_pos_n", "mean"),
            Prevalence=("prevalence", "mean"),
            Threshold=("threshold", "mean"),
            AP_mean=("AP", "mean"),
            AP_SD=("AP", "std"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_SD=("AUROC", "std"),
            Precision_mean=("Precision", "mean"),
            Recall_mean=("Recall", "mean"),
            F1_mean=("F1", "mean"),
            F2_mean=("F2", "mean"),
        )
        .reindex(aligned.label_names)
        .reset_index()
        .rename(columns={"label": "Label"})
    )

    paper_paired_tests = paired.copy()
    paper_paired_tests["comparison"] = paper_paired_tests["comparison"].replace(PAPER_MODE_NAMES, regex=True)
    paper_paired_tests = paper_paired_tests.rename(
        columns={
            "comparison": "Comparison",
            "metric": "Metric",
            "confirmatory": "Confirmatory",
            "n_pairs": "Pairs",
            "mean_delta": "Mean delta",
            "CI_level": "CI level",
            "CI_low": "CI low",
            "CI_high": "CI high",
            "cohens_dz": "Cohen's dz",
            "p_two_sided": "p value",
            "significant": "Significant (p<0.001, confirmatory only)",
        }
    )

    b3_weight_report = (
        weights.groupby("label")[["w_RAD_DINO_LATENT", "w_BioViL_T_LATENT", "w_EARLY"]]
        .agg(["mean", "std"])
        .reindex(aligned.label_names)
    )
    b3_weight_report.columns = [f"{component}_{stat}" for component, stat in b3_weight_report.columns]
    b3_weight_report = b3_weight_report.reset_index().rename(columns={"label": "Label"})

    b3_seed_diagnostic = (
        seed_results[seed_results["Configuration"].eq(PAPER_MODE_NAMES[MODE_ORDER[-1]])]
        .sort_values("mAP", ascending=False)
        .reset_index(drop=True)
    )

    tables = {
        "01_cohort_overview.csv": cohort_overview,
        "02_label_distribution.csv": label_distribution,
        "03_configurations.csv": configuration_table,
        "04_data_split.csv": paper_split_table,
        "05_key_hyperparameters.csv": paper_hyperparameters,
        "06_overall_results_95ci.csv": overall_results,
        "07_seed_level_results.csv": seed_results,
        "08_B3_label_results.csv": b3_paper_results,
        "09_paired_statistical_tests.csv": paper_paired_tests,
        "10_B3_fusion_weights.csv": b3_weight_report,
        "11_B3_seed_diagnostic.csv": b3_seed_diagnostic,
    }
    for filename, table in tables.items():
        _save(table, report_dir / filename, filename)

    manifest = {
        "primary_tables": ["06_overall_results_95ci.csv", "08_B3_label_results.csv", "09_paired_statistical_tests.csv"],
        "methods_tables": [
            "01_cohort_overview.csv",
            "02_label_distribution.csv",
            "03_configurations.csv",
            "04_data_split.csv",
            "05_key_hyperparameters.csv",
        ],
        "supplementary_tables": ["07_seed_level_results.csv", "10_B3_fusion_weights.csv", "11_B3_seed_diagnostic.csv"],
        "note": (
            "Primary claims use mean (95% CI) across seeds for mAP/AUROC and the three "
            "prespecified paired comparisons in 09_paired_statistical_tests.csv "
            "(significant at p<0.001, confirmatory rows only). Macro F1/Precision/PRROC "
            "and the best-seed table are exploratory / diagnostic only."
        ),
        "best_seed_diagnostic_only": {"seed": best_seed, "mAP": float(best_seed_map)},
    }
    (report_dir / "paper_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Paper candidate report saved to: %s", report_dir)
    return report_dir
