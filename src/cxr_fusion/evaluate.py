from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from .fusion import CONFIRMATORY_METRICS, MODE_ORDER, REPORT_METRICS

METRIC_COLS = ["AP", "AUROC", "PRROC", "Precision", "Recall", "F1", "F2"]

DEFAULT_COMPARISONS = [
    ("B2_EARLY_FUSION", "B1_BIOVILT"),
    ("B2_EARLY_FUSION", "B0_RADDINO"),
    ("B3_HYBRID", "B2_EARLY_FUSION"),
]

CONFIRMATORY_P_THRESHOLD = 0.001


def b3_label_metrics(results: pd.DataFrame, label_names: list[str]):
    b3_results = results[results["feature_mode"].eq("B3_HYBRID")]
    b3_mean = b3_results.groupby("label")[METRIC_COLS].mean().reindex(label_names)
    b3_std = b3_results.groupby("label")[METRIC_COLS].std().fillna(0).reindex(label_names)
    return b3_results, b3_mean, b3_std


def weight_summary_tables(weights: pd.DataFrame, b3_results: pd.DataFrame, label_names: list[str]):
    weight_columns = ["w_RAD_DINO_LATENT", "w_BioViL_T_LATENT", "w_EARLY"]
    weight_summary = weights.groupby("label")[weight_columns].agg(["mean", "std"]).reindex(label_names)

    seed_map = b3_results.groupby("seed")["AP"].mean().sort_values(ascending=False)
    best_seed = int(seed_map.index[0])
    best_seed_weights = (
        weights[weights["seed"].eq(best_seed)].set_index("label")[weight_columns].reindex(label_names)
    )
    return weight_summary, best_seed, seed_map.iloc[0], best_seed_weights


def compute_mode_seed_means(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["feature_mode", "seed"])
        .agg(
            mAP=("AP", "mean"),
            mean_P=("Precision", "mean"),
            mean_AUROC=("AUROC", "mean"),
            mean_PRROC=("PRROC", "mean"),
            mean_F1=("F1", "mean"),
        )
        .reset_index()
    )


def compute_mode_ci(
    mode_seed_means: pd.DataFrame,
    ci_level: float = 0.95,
    mode_order: list[str] = MODE_ORDER,
    report_metrics: list[str] = REPORT_METRICS,
) -> pd.DataFrame:
    alpha = 1 - (1 - ci_level) / 2
    rows = []
    for mode in mode_order:
        mode_values = mode_seed_means[mode_seed_means["feature_mode"].eq(mode)]
        for metric in report_metrics:
            values = mode_values[metric].dropna().to_numpy()
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1)
            half_width = student_t.ppf(alpha, n - 1) * std / np.sqrt(n) if n > 1 else np.nan
            rows.append(
                {
                    "feature_mode": mode,
                    "metric": metric,
                    "n_seeds": n,
                    "mean": mean,
                    "std": std,
                    "CI_low": mean - half_width,
                    "CI_high": mean + half_width,
                }
            )
    mode_ci = pd.DataFrame(rows)
    mode_ci["mean [95% CI]"] = mode_ci.apply(
        lambda row: f"{row['mean']:.6f} [{row['CI_low']:.6f}, {row['CI_high']:.6f}]", axis=1
    )
    return mode_ci


def paired_tests(
    mode_seed_means: pd.DataFrame,
    comparisons: list[tuple[str, str]] = DEFAULT_COMPARISONS,
    report_metrics: list[str] = REPORT_METRICS,
    ci_level: float = 0.95,
    confirmatory_metrics: list[str] = CONFIRMATORY_METRICS,
    confirmatory_p_threshold: float = CONFIRMATORY_P_THRESHOLD,
) -> pd.DataFrame:
    n_confirmatory = len(comparisons) * len(confirmatory_metrics)
    confirmatory_ci_level = 1 - 0.05 / n_confirmatory
    rows = []
    for model_a, model_b in comparisons:
        for metric in report_metrics:
            paired = (
                mode_seed_means[mode_seed_means["feature_mode"].isin([model_a, model_b])]
                .pivot(index="seed", columns="feature_mode", values=metric)
                .dropna()
            )
            differences = (paired[model_a] - paired[model_b]).to_numpy()
            n = len(differences)
            mean_delta = differences.mean()
            std_delta = differences.std(ddof=1)
            is_confirmatory = metric in confirmatory_metrics
            level = confirmatory_ci_level if is_confirmatory else ci_level
            alpha = 1 - (1 - level) / 2
            half_width = student_t.ppf(alpha, n - 1) * std_delta / np.sqrt(n) if n > 1 else np.nan
            p_value = ttest_rel(paired[model_a], paired[model_b]).pvalue if n > 1 else np.nan
            significant = bool(is_confirmatory and np.isfinite(p_value) and p_value < confirmatory_p_threshold)
            rows.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "metric": metric,
                    "confirmatory": is_confirmatory,
                    "n_pairs": n,
                    "mean_delta": mean_delta,
                    "CI_level": level,
                    "CI_low": mean_delta - half_width,
                    "CI_high": mean_delta + half_width,
                    "cohens_dz": mean_delta / std_delta if std_delta > 0 else np.inf,
                    "p_two_sided": p_value,
                    "significant": significant,
                }
            )
    return pd.DataFrame(rows)
