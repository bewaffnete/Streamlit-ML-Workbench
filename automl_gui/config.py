"""Configuration DTOs for app behavior and model workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any


@dataclass(slots=True)
class DataLoadConfig:
    max_upload_mb: int = 200
    csv_low_memory: bool = True
    csv_on_bad_lines: str = "warn"
    row_sampling_threshold: int = 500_000
    memory_sampling_threshold_mb: int = 1024
    default_sample_size: int = 50_000
    csv_encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WarningsConfig:
    target_corr_threshold: float = 0.92
    feature_corr_threshold: float = 0.95
    missing_threshold: float = 0.65
    high_cardinality_threshold: int = 50
    max_rows_for_warning_scan: int = 120_000
    top_numeric_features_for_pairwise: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    task_type: str = "Classification"
    evaluation_mode: str = "Train/Test Split"
    train_size: float = 0.8
    cv_folds: int = 5
    stratify: bool = True
    enable_hyperparameter_tuning: bool = False
    tuning_iterations: int = 25
    enable_early_stopping: bool = True
    random_state: int = 42
    n_clusters: int = 3
    selected_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    classification: dict[str, dict[str, Any]] = field(default_factory=dict)
    regression: dict[str, dict[str, Any]] = field(default_factory=dict)
    clustering: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_params(self, task_type: str, model_name: str) -> dict[str, Any]:
        task_key = task_type.lower()
        catalog = getattr(self, task_key, {})
        return dict(catalog.get(model_name, {}))


@dataclass(slots=True)
class RuntimeConfig:
    max_n_jobs: int = int(os.getenv("AUTOML_MAX_N_JOBS", "2"))
    background_workers: int = int(os.getenv("AUTOML_BG_WORKERS", "1"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
