# Data

This repository does **not** ship any MIMIC-CXR images, embeddings, or
derived tables. MIMIC-CXR is a credentialed PhysioNet dataset and its
[data use agreement](https://physionet.org/content/mimic-cxr/view-dua/2.1.0/)
prohibits redistribution — that applies to the raw JPEGs as well as any
per-study feature cache derived from them.

## What you need to provide

1. **Credentialed MIMIC-CXR / MIMIC-CXR-JPG access.** Complete PhysioNet's
   credentialing (CITI training + data use agreement) and download the
   dataset yourself.
2. **A RAD-DINO study-level feature cache**, matching the schema this
   pipeline expects at `paths.raddino_dir` (see `configs/default.yaml`):
   - `v2_embedding.npy` (or `v1_onehot.npy`) — the embedding matrix, one row
     per RAD-DINO-encoded image.
   - `index.parquet` — maps each matrix row to a `study_id`.
   - `columns.json` — per-variant column names (`x_cxr_0..767` for the image
     features) and the ordered list of 14 CheXpert-style labels.
   - `cohort_study_level.parquet` — the study-level label table
     (`subject_id`, `study_id`, one column per label).
   - `support_devices_update_audit.json` *(optional)* — provenance for a
     patched-in "Support Devices" label; omit it if your cohort already
     ships that label natively (`build_study_base` falls back automatically).
3. **A BioViL-T study-level feature cache** at `paths.bio_cache_dir`:
   - `biovilt_study_embeddings.npy` — the embedding matrix.
   - `biovilt_study_manifest.parquet` — maps each row (`bio_row`) to
     `subject_id` / `study_id`.
   - `cache_spec.json` — cache version/provenance metadata.

> **Note:** the notebook this repository was extracted from consumed these
> two caches as a given — it does not itself contain the RAD-DINO/BioViL-T
> feature-extraction code. If you are reproducing this work end-to-end from
> raw MIMIC-CXR images, you will need RAD-DINO and BioViL-T inference
> scripts to produce the caches above before running `scripts/01_build_dataset.py`.

## Point the pipeline at your data

Copy `configs/default.yaml`, edit the `paths:` block to point at your local
cache directories, and pass `--config` to every script:

```bash
python scripts/01_build_dataset.py --config configs/my_env.yaml
```

## Frontal/lateral (multi-view) cohorts

The default config matches the single-view (frontal) cohort used for the
paper's primary numbers. If you built separate caches for other view
counts or a frontal-only cohort, create one config file per variant —
everything else in the pipeline is identical.
