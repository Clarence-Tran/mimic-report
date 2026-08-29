# CXR Fusion: Multi-view Chest X-ray Hybrid Fusion (RAD-DINO + BioViL-T)

Code for a multi-label chest X-ray classification method (14 CheXpert-style
labels, MIMIC-CXR) that refines unimodal RAD-DINO and vision-language
BioViL-T representations separately in latent space before fusing them.

| Config | Input | Description |
| --- | --- | --- |
| B0 | RAD-DINO (768-d) | branch model, evaluated standalone |
| B1 | BioViL-T (128-d) | branch model, evaluated standalone |
| B2 | RAD-DINO &#124;&#124; BioViL-T (896-d) | branch model, evaluated standalone |
| B3 | — | uniform (1/3, 1/3, 1/3) fusion of B0, B1, B2 |

B0, B1, and B2 are three independent instances of the *same* branch
architecture. The paper's own parameter/FLOP counts (Table 10: approx. 6-7M
params per single-branch configuration, approx. 20M for the fused model)
only match a shared autoencoder + transformer architecture, not a bare
linear layer (approx. 10 thousand params for a 768&rarr;14 linear map).

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Scripts and outputs](#scripts-and-outputs)
- [Reproducibility](#reproducibility)
- [Repository structure](#repository-structure)
- [Status](#status)
- [Citation](#citation)
- [License](#license)

## Architecture

Inputs are standardized with train-split statistics, then each one is passed
through the same branch architecture: an autoencoder refines the embedding
in a 256-d latent space, a denoiser cleans the latent, and a 2-token
transformer with a learned gate combines the raw embedding with the refined
latent before the classification head.

```
RAD-DINO (768-d)              --->  autoencoder -> latent denoiser -> transformer + gate  --->  l_R  (B0)
BioViL-T (128-d)              --->  autoencoder -> latent denoiser -> transformer + gate  --->  l_B  (B1)
RAD-DINO || BioViL-T (896-d)  --->  autoencoder -> latent denoiser -> transformer + gate  --->  l_E  (B2)

B3 = uniform fusion (1/3, 1/3, 1/3) of the per-label-normalized l_R, l_B, l_E
```

Each branch is trained independently per seed with asymmetric focal loss
(label smoothing + per-label positive weighting), mixup, feature noise, and
top-k checkpoint averaging. Per-label decision thresholds are tuned on the
validation split subject to a minimum-recall floor. Logits are z-normalized
per label using validation-split statistics before fusion.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python 3.10+. `requirements.txt` lists dependency floors; after a
successful run in your training environment, capture exact versions with
`pip freeze > requirements-lock.txt` (`scipy`/`scikit-learn` minor-version
drift can shift `ttest_rel`/AP values at the fourth decimal).

## Data

This repository ships no data. See [`data/README.md`](data/README.md) for
what a RAD-DINO / BioViL-T study-level cache must contain and for
MIMIC-CXR's credentialed access requirements.

## Usage

```bash
cp configs/default.yaml configs/my_env.yaml     # point `paths:` at your caches

python scripts/01_build_dataset.py            --config configs/my_env.yaml
python scripts/02_train_baselines.py          --config configs/my_env.yaml
python scripts/03_make_figures_and_tables.py  --config configs/my_env.yaml
python scripts/04_generate_paper_report.py    --config configs/my_env.yaml
```

Each script is idempotent and resumable: `02_train_baselines.py` caches
per-(branch, seed) checkpoints and predictions, and skips any that already
match the current split's validation/test sample ids.

## Scripts and outputs

| Script | Produces |
| --- | --- |
| `01_build_dataset.py` | aligned cohort cache; cohort/label/coverage audits; a linear-probe sanity-check diagnostic |
| `02_train_baselines.py` | `B0_B3_results.csv`, `B3_uniform_weights.csv` — the source of every B0-B3 number |
| `03_make_figures_and_tables.py` | B3 per-label heatmap, ROC curves, mAP/AUROC bar chart, confidence intervals, paired tests, hyperparameter/split tables |
| `04_generate_paper_report.py` | manuscript-ready tables under `outputs/checkpoints/<case_name>/paper_candidate_report/`, plus `paper_report_manifest.json` |

`paper_report_manifest.json` tags each table as `primary`, `methods`, or
`supplementary`, matching how they are intended to appear in the paper.

## Reproducibility

- Five fixed seeds (`configs/default.yaml: seeds`); `set_seed()` is applied
  per branch/seed before any randomness (torch, numpy, and `random`, across
  all CUDA devices).
- The train/validation/test split is a deterministic temporal cut on each
  subject's earliest study (`cxr_fusion.splits.split_for_seed`), not
  reshuffled per seed. This avoids patient-level leakage across splits, at
  the cost of the split itself not varying across seeds. `index_time` is
  MIMIC-CXR's per-subject date-shifted timestamp: ordering is only
  meaningful within one subject, never compared across subjects.
- Primary results are reported as mean (95% CI) across seeds, together with
  the three prespecified paired comparisons in
  `09_paired_statistical_tests.csv` (significant at p < 0.001, restricted to
  the confirmatory family: mAP and macro AUROC). The best-seed table is a
  diagnostic only.

## Repository structure

```
configs/default.yaml   # every hyperparameter and path; nothing hardcoded in code
data/README.md         # how to obtain/build the RAD-DINO and BioViL-T caches
pyproject.toml         # packaging (pip install -e .)
requirements.txt       # dependency version floors

scripts/
    01_build_dataset.py             # build and cache the aligned cohort
    02_train_baselines.py           # train the three branches, evaluate B0-B3
    03_make_figures_and_tables.py   # figures, confidence intervals, paired tests
    04_generate_paper_report.py     # assemble manuscript-ready tables

src/cxr_fusion/
    config.py           # YAML -> typed Config
    utils.py            # id_str / set_seed / canonical_label helpers
    data.py             # RAD-DINO + BioViL-T cache loading and alignment
    representation.py   # linear-probe sanity-check diagnostic (not part of B0-B3)
    datasets.py         # torch Dataset wrapper
    models.py           # autoencoder, latent denoiser, fusion transformer (one shared branch model)
    losses.py           # asymmetric focal loss, mixup
    splits.py           # subject-anchored temporal split
    train.py            # branch training loop (autoencoder -> denoiser -> fusion transformer)
    fusion.py           # late-fusion weight search/application, thresholding
    evaluate.py         # confidence intervals, paired tests, per-label aggregation
    figures.py          # heatmap, ROC curves, bar chart
    report.py           # hyperparameter/split tables, paper-candidate report
```

## Status

The core B0-B3 pipeline — dataset alignment, training, evaluation, figures,
and the paper-candidate report — is fully implemented and covered by an
end-to-end smoke test. Supplementary robustness studies described in the
paper's appendices (architecture ablations, calibration and rare-label
analysis, inference latency/FLOPs, representation-similarity/CKA analysis,
noise-ceiling sensitivity) are not included in this release.

## Citation

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026},
}
```

## License

MIT — see [`LICENSE`](LICENSE). This covers the code only; MIMIC-CXR,
RAD-DINO, and BioViL-T remain under their own respective licenses and data
use agreements.
