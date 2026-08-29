from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#4D4D4D"]


def plot_b3_heatmap(b3_mean: pd.DataFrame, b3_std: pd.DataFrame, out_dir: Path) -> Path:
    annotations = b3_mean.copy().astype(str)
    for label in b3_mean.index:
        for metric in b3_mean.columns:
            annotations.loc[label, metric] = f"{b3_mean.loc[label, metric]:.3f}±{b3_std.loc[label, metric]:.3f}"

    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    sns.heatmap(b3_mean, annot=annotations, fmt="", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("B3 hybrid fusion — 14 labels (mean ± SD)")
    ax.set_ylabel("CXR finding")
    plt.tight_layout()
    out_path = out_dir / "heatmap_B3_14labels.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(
    b3_outputs: dict[int, dict[str, np.ndarray]], label_names: list[str], out_dir: Path
) -> tuple[Path, pd.DataFrame]:
    with plt.rc_context(_PUBLICATION_RC):
        mean_fpr = np.linspace(0, 1, 301)
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
        line_styles = ["-", "--"]

        roc_summary_rows = []
        for label_index, label in enumerate(label_names):
            interpolated_tprs, seed_aurocs = [], []
            for seed, outputs in b3_outputs.items():
                y_true = outputs["y"][:, label_index]
                score = outputs["score"][:, label_index]
                if len(np.unique(y_true)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, score)
                interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
                interpolated_tpr[0] = 0.0
                interpolated_tprs.append(interpolated_tpr)
                seed_aurocs.append(roc_auc_score(y_true, score))

            if not interpolated_tprs:
                continue

            mean_tpr = np.mean(interpolated_tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = float(np.mean(seed_aurocs))
            std_auc = float(np.std(seed_aurocs, ddof=1)) if len(seed_aurocs) > 1 else 0.0
            color = OKABE_ITO[label_index % len(OKABE_ITO)]
            line_style = line_styles[label_index // len(OKABE_ITO)]

            ax.plot(
                mean_fpr, mean_tpr, color=color, linestyle=line_style, linewidth=2.2, alpha=0.98,
                label=f"{label} ({mean_auc:.3f})",
            )
            roc_summary_rows.append(
                {"label": label, "mean_AUROC": mean_auc, "std_AUROC": std_auc, "n_seeds": len(seed_aurocs)}
            )

        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=(0, (2.5, 2.5)), linewidth=1.6, label="Chance (0.500)", zorder=0)
        ax.set_title("ROC curves — B3 hybrid fusion", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_box_aspect(1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=5, width=1.2, pad=4)
        ax.legend(
            title="Finding (mean AUROC)", loc="lower right", frameon=True, facecolor="white",
            edgecolor="#D1D5DB", framealpha=0.94, fontsize=8.2, title_fontsize=9.0,
            handlelength=2.2, handletextpad=0.45, labelspacing=0.28, borderpad=0.6,
        )
        plt.tight_layout(pad=0.8)

        stem = out_dir / "B3_ROC_curves_14labels"
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None, facecolor="white")
        plt.close(fig)

    roc_summary = pd.DataFrame(roc_summary_rows).set_index("label").reindex(label_names)
    roc_summary.to_csv(out_dir / "B3_ROC_AUROC_14labels.csv")
    return stem.with_suffix(".png"), roc_summary


def plot_mode_comparison_bar(mode_seed_means: pd.DataFrame, mode_order: list[str], out_dir: Path) -> Path:
    mode_display_names = {
        "B0_RADDINO": "RADDINO",
        "B1_BIOVILT": "BioViL-T",
        "B2_EARLY_FUSION": "Early fusion",
        "B3_HYBRID": "Hybrid fusion",
    }
    bar_summary = mode_seed_means.groupby("feature_mode")[["mAP", "mean_AUROC"]].agg(["mean", "std"]).reindex(mode_order)

    with plt.rc_context(_PUBLICATION_RC):
        x = np.arange(len(mode_order))
        width = 0.34
        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
        bar_specs = [("mAP", "mAP", "#3C5488", -width / 2), ("mean_AUROC", "AUROC", "#E64B35", width / 2)]

        for metric, legend_label, color, offset in bar_specs:
            means = bar_summary[(metric, "mean")].to_numpy()
            stds = bar_summary[(metric, "std")].fillna(0).to_numpy()
            bars = ax.bar(
                x + offset, means, width, yerr=stds, capsize=4, color=color, edgecolor="white", linewidth=0.8,
                error_kw={"ecolor": "#374151", "elinewidth": 1.2, "capthick": 1.2}, label=legend_label, zorder=3,
            )
            for bar, mean, std in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, mean + std + 0.018, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1F2937", zorder=5,
                )

        ax.set_title("Performance comparison across configurations", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([mode_display_names[mode] for mode in mode_order], fontsize=10, fontweight="semibold")
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="y", labelsize=10, width=1.0, length=4)
        ax.tick_params(axis="x", width=1.0, length=4, pad=7)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=10, handlelength=1.6, columnspacing=1.5)
        plt.tight_layout()

        out_path = out_dir / "B0_B3_mAP_AUROC_comparison.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(out_dir / "B0_B3_mAP_AUROC_comparison.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    bar_summary.to_csv(out_dir / "B0_B3_mAP_AUROC_bar_summary.csv")
    return out_path
