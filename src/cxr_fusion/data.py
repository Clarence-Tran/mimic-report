from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, PathsConfig
from .utils import canonical_label, id_str

logger = logging.getLogger(__name__)


def resolve_feature_path(raddino_dir: Path) -> Path:
    v2_path = raddino_dir / "v2_embedding.npy"
    v1_path = raddino_dir / "v1_onehot.npy"
    return v2_path if v2_path.exists() else v1_path


@dataclass
class StudyBase:
    study_base: pd.DataFrame
    study_index_time: np.ndarray
    matrix: np.ndarray
    rad_indices: list[int]
    label_cols: list[str]
    feature_path: Path
    parquet_audit: dict
    support_audit: pd.DataFrame


def build_study_base(paths: PathsConfig) -> StudyBase:
    columns_doc = json.loads(paths.columns_json.read_text(encoding="utf-8"))
    feature_path = resolve_feature_path(paths.raddino_dir)
    variant = feature_path.stem
    schema = columns_doc["variants"][variant]
    feature_columns = list(map(str, schema["columns"]))
    rad_indices = [
        index
        for index, column in enumerate(feature_columns)
        if re.fullmatch(r"x_cxr_\d+", column)
    ]

    matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)

    source_parquet = paths.raddino_dir / f"{variant}.parquet"
    source = pd.read_parquet(source_parquet) if source_parquet.exists() else None

    index_meta = pd.read_parquet(paths.index_path)
    row_col = columns_doc.get("row_col", "row")
    index_name_map = {str(column).lower(): column for column in index_meta.columns}
    index_study_col = index_name_map.get("study_id") or index_name_map.get("studyid")

    index_meta[row_col] = pd.to_numeric(index_meta[row_col], errors="raise").astype(int)
    index_order = index_meta.sort_values(row_col).reset_index(drop=True)
    index_order["study_id"] = index_order[index_study_col].map(id_str)

    parquet_audit = {"performed": False}
    if source is not None:
        probe = np.unique(np.linspace(0, len(matrix) - 1, 257, dtype=int))
        source_name_map = {str(column).lower(): column for column in source.columns}
        source_study_col = source_name_map.get("study_id") or source_name_map.get("studyid")

        if source_study_col:
            source_probe = source[[source_study_col] + feature_columns].copy()
            source_probe[source_study_col] = source_probe[source_study_col].map(id_str)
            wanted = (
                index_order.iloc[probe][["study_id", row_col]]
                .merge(
                    source_probe,
                    left_on="study_id",
                    right_on=source_study_col,
                    how="left",
                    validate="one_to_one",
                )
                .sort_values(row_col)
            )
            parquet_values = wanted[feature_columns].to_numpy("float32")
        else:
            parquet_values = source.iloc[probe][feature_columns].to_numpy("float32")

        npy_values = np.asarray(matrix[probe], dtype="float32")
        max_diff = float(np.nanmax(np.abs(parquet_values - npy_values)))
        matches = bool(
            np.allclose(parquet_values, npy_values, rtol=1e-5, atol=1e-6, equal_nan=False)
        )
        parquet_audit = {
            "performed": True,
            "join_key": "study_id" if source_study_col else "position",
            "max_abs_diff": max_diff,
            "allclose": matches,
        }
        logger.info("Parquet audit: %s", parquet_audit)
        if not matches:
            logger.warning("Parquet export differs from the NPY/index pair; ignoring it.")
    else:
        logger.info("Parquet audit skipped: training uses NPY + index.parquet only.")

    cohort = pd.read_parquet(paths.cohort_path).reset_index(drop=True)
    cohort_name_map = {str(column).lower(): column for column in cohort.columns}
    cohort_subject_col = cohort_name_map.get("subject_id") or cohort_name_map.get("subjectid")
    cohort_study_col = cohort_name_map.get("study_id") or cohort_name_map.get("studyid")
    cohort = cohort.rename(columns={cohort_subject_col: "subject_id", cohort_study_col: "study_id"})
    cohort[["subject_id", "study_id"]] = cohort[["subject_id", "study_id"]].apply(
        lambda column: column.map(id_str)
    )

    selected = (
        index_order[["study_id", row_col]]
        .merge(cohort, on="study_id", how="left", validate="one_to_one")
        .sort_values(row_col)
        .reset_index(drop=True)
    )

    schema_labels = columns_doc["labels"]
    native_label_source = [
        next(
            (column for column in selected.columns if canonical_label(column) == canonical_label(label)),
            None,
        )
        for label in schema_labels
    ]
    label_cols = ["y_" + label.replace("_", " ") for label in schema_labels]

    study_base = selected[["subject_id", "study_id"] + native_label_source].copy()
    study_base.columns = ["subject_id", "study_id"] + label_cols
    study_base[["subject_id", "study_id"]] = study_base[["subject_id", "study_id"]].apply(
        lambda column: column.map(id_str)
    )
    study_base["feature_row"] = np.arange(len(study_base))

    for column in label_cols:
        study_base[column] = (
            pd.to_numeric(study_base[column], errors="coerce").fillna(0).clip(0, 1).astype("float32")
        )

    if paths.support_audit_path.exists():
        support_spec = json.loads(paths.support_audit_path.read_text(encoding="utf-8"))
        support_source_path = support_spec["official_chexpert_source"]
        support_coverage = float(support_spec["study_coverage"])
        support_mapping = support_spec["support_policy"]
        support_cohort_source = "patched via add_label_support_devices_RADDINO (audit found)"
    else:
        support_source_path = str(paths.cohort_path)
        support_coverage = float(study_base["y_Support Devices"].notna().mean())
        support_mapping = "native to cohort_study_level.parquet (no audit file; assumed already binary 0/1)"
        support_cohort_source = "native 14-label cohort (no add_label audit found)"
        logger.info(
            "%s not found; treating Support Devices as a native cohort label.",
            paths.support_audit_path,
        )

    support_audit = pd.DataFrame(
        [
            {
                "source": support_source_path,
                "cohort_studies": len(study_base),
                "matched_studies": len(study_base),
                "coverage": support_coverage,
                "positive_studies": int(study_base["y_Support Devices"].sum()),
                "prevalence": float(study_base["y_Support Devices"].mean()),
                "join_keys": "subject_id+study_id",
                "mapping": support_mapping,
                "cohort_source": support_cohort_source,
            }
        ]
    )

    study_index_time = selected["index_time"].to_numpy()

    logger.info(
        "RAD-DINO matrix %s -> image embedding (%d, %d); native cohort %d studies, %d labels.",
        matrix.shape,
        len(matrix),
        len(rad_indices),
        len(study_base),
        len(label_cols),
    )

    return StudyBase(
        study_base=study_base,
        study_index_time=study_index_time,
        matrix=matrix,
        rad_indices=rad_indices,
        label_cols=label_cols,
        feature_path=feature_path,
        parquet_audit=parquet_audit,
        support_audit=support_audit,
    )


@dataclass
class BiovilCache:
    manifest: pd.DataFrame
    embeddings: np.ndarray
    cache_spec: dict
    coverage: float
    manifest_hash: str


def load_biovilt_cache(paths: PathsConfig, study_base: pd.DataFrame) -> BiovilCache:
    cache_spec = json.loads(paths.bio_cache_spec.read_text(encoding="utf-8"))
    manifest = pd.read_parquet(paths.bio_manifest_path)
    embeddings = np.load(paths.bio_emb_path, mmap_mode="r", allow_pickle=False)

    manifest[["subject_id", "study_id"]] = manifest[["subject_id", "study_id"]].apply(
        lambda column: column.map(id_str)
    )
    manifest["bio_row"] = pd.to_numeric(manifest["bio_row"], errors="raise").astype(int)
    manifest["row"] = pd.to_numeric(manifest["row"], errors="raise").astype(int)

    matched = study_base[["subject_id", "study_id", "feature_row"]].merge(
        manifest, on=["subject_id", "study_id"], how="inner", validate="one_to_one", sort=False
    )
    coverage = len(matched) / len(study_base)

    study_hash = cache_spec.get("study_manifest_hash")
    if not study_hash:
        study_ids = (manifest.subject_id + "::" + manifest.study_id).astype(str)
        study_hash = hashlib.sha256("|".join(study_ids).encode()).hexdigest()[:16]

    logger.info(
        "Loaded BioViL-T v3: %d studies | coverage: %.2f%% | embeddings: %s | version: %s",
        len(manifest),
        coverage * 100,
        embeddings.shape,
        cache_spec.get("version"),
    )
    return BiovilCache(
        manifest=manifest,
        embeddings=embeddings,
        cache_spec=cache_spec,
        coverage=coverage,
        manifest_hash=study_hash,
    )


def build_biovilt_cache_audit(paths: PathsConfig, bio_cache: BiovilCache) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cache_dir": str(paths.bio_cache_dir),
                "cache_version": bio_cache.cache_spec.get("version"),
                "cached_studies_total": len(bio_cache.manifest),
                "cohort_study_coverage": bio_cache.coverage,
                "embedding_dim": bio_cache.embeddings.shape[1],
                "embedding_dtype": str(bio_cache.embeddings.dtype),
                "source_jpg_accessed": bool(bio_cache.cache_spec.get("source_jpg_accessed", False)),
                "study_manifest_hash": bio_cache.manifest_hash,
            }
        ]
    )


@dataclass
class AlignedCohort:
    common: pd.DataFrame
    x_new: np.ndarray
    x_bio: np.ndarray
    y: np.ndarray
    study_dates: np.ndarray
    sample_ids: np.ndarray
    sample_meta: pd.DataFrame
    label_names: list[str]
    label_cols: list[str]
    coverage: float
    cohort_hash: str
    cohort_spec: dict


def align_rad_and_bio(
    cfg: Config,
    study: StudyBase,
    bio_cache: BiovilCache,
) -> AlignedCohort:
    common = study.study_base.merge(
        bio_cache.manifest, on=["subject_id", "study_id"], how="inner", validate="one_to_one", sort=False
    )
    coverage = len(common) / len(study.study_base)

    x_new = np.asarray(study.matrix[common.feature_row.to_numpy(int)], dtype="float32")[:, study.rad_indices]
    x_bio = bio_cache.embeddings[common.bio_row.to_numpy(int)]
    y = common[study.label_cols].to_numpy(dtype="float32")

    study_dates = study.study_index_time[common.feature_row.to_numpy(int)]

    sample_ids = (common.subject_id + "::" + common.study_id).to_numpy(str)
    sample_meta = common[["subject_id", "study_id"]].copy()
    sample_meta.insert(0, "sample_id", sample_ids)
    label_names = [column[2:] for column in study.label_cols]

    cohort_hash = hashlib.sha256("|".join(sample_ids).encode()).hexdigest()[:16]
    cohort_spec = {
        "case": cfg.case_name,
        "alignment_version": cfg.alignment_version,
        "level": "study",
        "n_native_cohort": len(study.study_base),
        "n_common": len(common),
        "coverage": coverage,
        "rad_source": str(study.feature_path),
        "rad_rule": "x_cxr_0..767 aligned by index.parquet[row, study_id]",
        "bio_source": "BioViL-T study cache aligned by bio_row and row",
        "bio_cache_dir": str(cfg.paths.bio_cache_dir),
        "bio_cache_version": bio_cache.cache_spec.get("version"),
        "bio_cache_cohort_coverage": bio_cache.coverage,
        "bio_manifest_hash": bio_cache.manifest_hash,
        "labels": study.label_cols,
        "n_labels": len(study.label_cols),
        "component_architectures": {
            "RAD_DINO": "DenseFeatureAutoencoder + LatentDenoiser + FusionTokensPredictor (768-d input)",
            "BIO": "DenseFeatureAutoencoder + LatentDenoiser + FusionTokensPredictor (128-d input)",
            "EARLY": "DenseFeatureAutoencoder + LatentDenoiser + FusionTokensPredictor (896-d concatenated input)",
        },
        "support_devices": True,
        "split": (
            "subject-anchored pseudo-temporal split by index_time. index_time is "
            "MIMIC's date-shifted timestamp, shifted independently per subject_id - "
            "ordering is only meaningful within one subject, not across subjects."
        ),
        "cohort_hash": cohort_hash,
    }

    logger.info(
        "FINAL COHORT: %d/%d studies (%.2f%%) | RAD-DINO %s | BioViL-T %s | labels=%d",
        len(common),
        len(study.study_base),
        coverage * 100,
        x_new.shape,
        x_bio.shape,
        len(study.label_cols),
    )

    return AlignedCohort(
        common=common,
        x_new=x_new,
        x_bio=x_bio,
        y=y,
        study_dates=study_dates,
        sample_ids=sample_ids,
        sample_meta=sample_meta,
        label_names=label_names,
        label_cols=study.label_cols,
        coverage=coverage,
        cohort_hash=cohort_hash,
        cohort_spec=cohort_spec,
    )


def save_aligned_cohort(aligned: AlignedCohort, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / "aligned_cohort.npz",
        x_new=aligned.x_new,
        x_bio=aligned.x_bio,
        y=aligned.y,
        study_dates=aligned.study_dates,
        sample_ids=aligned.sample_ids,
        label_names=np.asarray(aligned.label_names),
        label_cols=np.asarray(aligned.label_cols),
    )
    aligned.sample_meta.to_parquet(out_dir / "aligned_sample_meta.parquet", index=False)
    (out_dir / "cohort_spec.json").write_text(json.dumps(aligned.cohort_spec, indent=2), encoding="utf-8")


def load_aligned_cohort(out_dir: Path) -> AlignedCohort:
    npz = np.load(out_dir / "aligned_cohort.npz", allow_pickle=False)
    sample_meta = pd.read_parquet(out_dir / "aligned_sample_meta.parquet")
    cohort_spec = json.loads((out_dir / "cohort_spec.json").read_text(encoding="utf-8"))
    return AlignedCohort(
        common=sample_meta,
        x_new=npz["x_new"],
        x_bio=npz["x_bio"],
        y=npz["y"],
        study_dates=npz["study_dates"],
        sample_ids=npz["sample_ids"],
        sample_meta=sample_meta,
        label_names=list(npz["label_names"]),
        label_cols=list(npz["label_cols"]),
        coverage=cohort_spec["coverage"],
        cohort_hash=cohort_spec["cohort_hash"],
        cohort_spec=cohort_spec,
    )
