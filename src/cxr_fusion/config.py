from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    raddino_dir: Path
    cohort_path: Path
    index_path: Path
    columns_json: Path
    support_audit_path: Path

    bio_cache_dir: Path
    bio_emb_path: Path
    bio_manifest_path: Path
    bio_cache_spec: Path

    out_root: Path

    def out_dir(self, case_name: str) -> Path:
        out_dir = self.out_root / "checkpoints" / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


@dataclass
class SplitConfig:
    test_size: float = 0.20
    val_size_from_temp: float = 0.125


@dataclass
class DataLoaderConfig:
    batch_size: int = 1024
    num_workers: int = 0


@dataclass
class ModelConfig:
    latent_dim: int = 256
    hidden_dim: int = 512


@dataclass
class TrainConfig:
    ae_epochs: int = 12
    gen_epochs: int = 10
    pred_epochs: int = 60
    patience: int = 8
    topk: int = 3
    lr: float = 2e-4
    pred_lr: float = 1.5e-4
    weight_decay: float = 3e-4


@dataclass
class RegularizationConfig:
    feature_noise_std: float = 0.025
    mixup_alpha: float = 0.20
    label_smooth: float = 0.02


@dataclass
class FusionConfig:
    grid_step: float = 0.10
    min_fusion_weight: float = 0.30
    shrinkage: float = 0.70
    min_val_ap_gain: float = 0.002
    min_recall: float = 0.30


@dataclass
class ReportConfig:
    ci_level: float = 0.95
    sample_level: str = "study"


@dataclass
class Config:
    case_name: str
    alignment_version: str
    seeds: list[int]
    device: str
    paths: PathsConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @property
    def out_dir(self) -> Path:
        return self.paths.out_dir(self.case_name)

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def _paths_from_dict(raw: dict[str, Any], config_dir: Path) -> PathsConfig:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (config_dir / path).resolve()

    return PathsConfig(**{key: resolve(value) for key, value in raw.items()})


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent

    return Config(
        case_name=raw["case_name"],
        alignment_version=raw["alignment_version"],
        seeds=list(raw["seeds"]),
        device=raw.get("device", "auto"),
        paths=_paths_from_dict(raw["paths"], config_dir),
        split=SplitConfig(**raw.get("split", {})),
        data_loader=DataLoaderConfig(**raw.get("data_loader", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        regularization=RegularizationConfig(**raw.get("regularization", {})),
        fusion=FusionConfig(**raw.get("fusion", {})),
        report=ReportConfig(**raw.get("report", {})),
    )
