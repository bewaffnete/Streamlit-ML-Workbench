"""Application services (business logic), independent from Streamlit widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DataLoadConfig, RuntimeConfig, TrainConfig, WarningsConfig
from .data_utils import (
    dataframe_fingerprint,
    load_uploaded_file,
    maybe_sample_large_dataframe,
    register_dataset,
    validate_uploaded_file,
)
from .evaluation import MetricsService, feature_importance_frame, permutation_importance_frame
from .logging_utils import get_logger
from .models import (
    ModelRegistry,
    estimate_training_cost,
    get_default_registry_cached,
    model_param_distributions,
)
from .preprocessing import (
    DataCleaningRequest,
    DataCleaningResult,
    PreprocessorConfig,
    PreprocessorFactory,
    clean_dataset_before_split,
    sanitize_selected_features,
)
from .serialization import build_model_metadata
from .training import ModelTrainer, RandomizedSearchOptimizer, TrainingRequest
from .warnings_utils import SmartWarning, analyze_smart_warnings

LOGGER = get_logger("services")


@dataclass(slots=True)
class LoadedDataset:
    raw_df: pd.DataFrame
    working_df: pd.DataFrame
    was_sampled: bool
    raw_fingerprint: str
    working_fingerprint: str


@dataclass(slots=True)
class PreparedTrainingData:
    X: pd.DataFrame
    y: pd.Series | None
    feature_cols: list[str]
    dropped_columns: list[str]
    cleaned_result: DataCleaningResult


@dataclass(slots=True)
class ModelRunResult:
    model_name: str
    training_result: Any
    metrics: dict[str, float]
    predictions_df: pd.DataFrame
    builtin_importance: pd.DataFrame | None
    permutation_importance: pd.DataFrame | None
    metadata: dict[str, Any]
    dropped_columns: list[str]
    feature_cols: list[str]


@dataclass(slots=True)
class TrainingSummary:
    leaderboard: pd.DataFrame
    runs: dict[str, ModelRunResult]
    best_model_name: str


class DataService:
    def __init__(self, config: DataLoadConfig) -> None:
        self.config = config

    def load_dataset(
        self,
        *,
        file_name: str,
        file_size: int,
        content_type: str | None,
        file_bytes: bytes,
        use_sampling: bool,
        sample_rows: int,
    ) -> LoadedDataset:
        validate_uploaded_file(
            file_name=file_name,
            file_size_bytes=file_size,
            content_type=content_type,
            max_upload_mb=self.config.max_upload_mb,
            file_bytes=file_bytes,
        )
        raw_df = load_uploaded_file(
            file_name=file_name,
            file_bytes=file_bytes,
            config=self.config,
        )
        sampled_df, was_large = maybe_sample_large_dataframe(
            raw_df,
            row_threshold=self.config.row_sampling_threshold,
            memory_threshold_mb=self.config.memory_sampling_threshold_mb,
            sample_size=sample_rows,
            random_state=42,
        )
        working_df = sampled_df if (was_large and use_sampling) else raw_df
        raw_fp = dataframe_fingerprint(raw_df)
        working_fp = dataframe_fingerprint(working_df)
        register_dataset(raw_fp, raw_df)
        register_dataset(working_fp, working_df)
        return LoadedDataset(
            raw_df=raw_df,
            working_df=working_df,
            was_sampled=(working_df is sampled_df),
            raw_fingerprint=raw_fp,
            working_fingerprint=working_fp,
        )


class WarningService:
    def __init__(self, config: WarningsConfig) -> None:
        self.config = config

    def analyze(
        self,
        *,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None,
        task_type: str,
    ) -> list[SmartWarning]:
        return analyze_smart_warnings(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            task_type=task_type,
            config=self.config,
        )


class TrainingService:
    """Coordinates cleaning, preprocessing, model creation, and evaluation."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        preprocessor_factory: PreprocessorFactory,
        trainer: ModelTrainer,
        metrics_service: MetricsService,
    ) -> None:
        self.model_registry = model_registry
        self.preprocessor_factory = preprocessor_factory
        self.trainer = trainer
        self.metrics_service = metrics_service

    def estimate_cost(
        self,
        *,
        n_rows: int,
        n_features: int,
        train_config: TrainConfig,
    ) -> str:
        return estimate_training_cost(
            n_rows=n_rows,
            n_features=n_features,
            n_models=len(train_config.selected_models),
            tuning_enabled=train_config.enable_hyperparameter_tuning,
            tuning_iterations=train_config.tuning_iterations,
        )

    def available_models(self, task_type: str) -> list[str]:
        return self.model_registry.available_models(task_type)

    def prepare_training_data(
        self,
        *,
        df: pd.DataFrame,
        task_type: str,
        target_col: str | None,
        selected_features: list[str],
        cleaning_request: DataCleaningRequest,
    ) -> PreparedTrainingData:
        features = sanitize_selected_features(df, selected_features)
        if not features:
            raise ValueError("No valid features selected.")
        if task_type != "Clustering" and (target_col is None or target_col not in df.columns):
            raise ValueError("Please select a valid target column.")
        if target_col in features:
            features = [f for f in features if f != target_col]
        if not features:
            raise ValueError("No features remain after removing target.")

        cleaning_request.feature_cols = features
        cleaning_request.target_col = target_col if task_type != "Clustering" else None
        cleaned = clean_dataset_before_split(df, cleaning_request)
        if not cleaned.feature_cols:
            raise ValueError("All features were dropped by preprocessing settings.")
        if cleaned.frame.empty:
            raise ValueError("Dataset is empty after preprocessing rules.")

        X = cleaned.frame[cleaned.feature_cols].copy()
        y = None
        if task_type != "Clustering":
            y = cleaned.frame[target_col].copy()  # type: ignore[index]
            if task_type == "Regression":
                y = pd.to_numeric(y, errors="coerce")
                valid = y.notna()
                X = X.loc[valid]
                y = y.loc[valid]
            if y.empty:
                raise ValueError("No valid target rows remain after cleaning.")
            if task_type == "Classification" and y.nunique(dropna=True) < 2:
                raise ValueError("Classification requires at least two target classes.")

        return PreparedTrainingData(
            X=X,
            y=y,
            feature_cols=cleaned.feature_cols,
            dropped_columns=cleaned.dropped_columns,
            cleaned_result=cleaned,
        )

    def train_models(
        self,
        *,
        prepared_data: PreparedTrainingData,
        preprocessor_config: PreprocessorConfig,
        train_config: TrainConfig,
        dataset_fingerprint: str | None,
        run_permutation_importance: bool,
    ) -> TrainingSummary:
        selected_models = list(train_config.selected_models)
        if not selected_models:
            raise ValueError("Select at least one model.")

        X, y = prepared_data.X, prepared_data.y
        if y is None and train_config.task_type != "Clustering":
            raise ValueError("Target series is required for supervised training.")

        leaderboard_rows: list[dict[str, Any]] = []
        runs: dict[str, ModelRunResult] = {}

        shared_bundle = None
        if (
            train_config.task_type in {"Classification", "Regression"}
            and train_config.evaluation_mode == "Train/Test Split"
            and y is not None
        ):
            pre = self.preprocessor_factory.build(X, preprocessor_config)
            shared_bundle = self.trainer.fit_preprocessor_once(
                X=X,
                y=y,
                preprocessor=pre,
                config=train_config,
                task_type=train_config.task_type,
            )

        for model_name in selected_models:
            if model_name not in self.available_models(train_config.task_type):
                LOGGER.warning("Skipping unknown model '%s' for task '%s'", model_name, train_config.task_type)
                continue

            override_params = (
                {"n_clusters": train_config.n_clusters}
                if train_config.task_type == "Clustering"
                else None
            )
            model = self.model_registry.create_model(
                task_type=train_config.task_type,
                name=model_name,
                override_params=override_params,
            )
            param_space = model_param_distributions(train_config.task_type, model_name)

            if train_config.task_type in {"Classification", "Regression"}:
                if y is None:
                    continue
                if shared_bundle is not None:
                    result = self.trainer.train_supervised_with_shared_preprocessor(
                        model=model,
                        task_type=train_config.task_type,
                        shared_bundle=shared_bundle,
                        config=train_config,
                        param_distributions=param_space,
                    )
                else:
                    pre = self.preprocessor_factory.build(X, preprocessor_config)
                    request = TrainingRequest(
                        X=X,
                        y=y,
                        task_type=train_config.task_type,
                        model_name=model_name,
                        model=model,
                        preprocessor=pre,
                        config=train_config,
                        param_distributions=param_space,
                    )
                    result = self.trainer.train_supervised(request)

                bundle = self.metrics_service.collect_and_format_metrics(
                    task_type=train_config.task_type,
                    y_true=result.y_true,
                    y_pred=result.y_pred,
                    y_prob=result.y_prob,
                )
                metrics = bundle.metrics
                predictions = result.X_eval.copy()
                predictions["actual"] = result.y_true
                predictions["prediction"] = result.y_pred
                if result.y_prob is not None and np.ndim(result.y_prob) == 2:
                    model_obj = result.pipeline.named_steps["model"]
                    classes = (
                        model_obj.classes_
                        if hasattr(model_obj, "classes_")
                        else [f"class_{i}" for i in range(result.y_prob.shape[1])]
                    )
                    for idx, cls in enumerate(classes):
                        predictions[f"proba_{cls}"] = result.y_prob[:, idx]

                builtin_importance = feature_importance_frame(result.pipeline)
                perm_importance = None
                if run_permutation_importance:
                    perm_importance = permutation_importance_frame(
                        result.pipeline,
                        result.X_eval,
                        result.y_eval,  # type: ignore[arg-type]
                        task_type=train_config.task_type,
                    )
            else:
                pre = self.preprocessor_factory.build(X, preprocessor_config)
                request = TrainingRequest(
                    X=X,
                    y=None,
                    task_type=train_config.task_type,
                    model_name=model_name,
                    model=model,
                    preprocessor=pre,
                    config=train_config,
                )
                result = self.trainer.train_clustering(request)
                bundle = self.metrics_service.collect_and_format_metrics(
                    task_type="Clustering",
                    y_true=None,
                    y_pred=None,
                    transformed_x=result.transformed,
                    labels=result.labels,
                )
                metrics = bundle.metrics
                predictions = result.X_eval.copy()
                predictions["cluster"] = result.labels
                builtin_importance = feature_importance_frame(result.pipeline)
                perm_importance = None

            metadata = build_model_metadata(
                model_name=model_name,
                task_type=train_config.task_type,
                metrics=metrics,
                dataset_fingerprint=dataset_fingerprint,
                feature_cols=prepared_data.feature_cols,
                preprocessing_summary=preprocessor_config.to_dict(),
                train_seconds=float(result.train_seconds or 0.0),
            )
            runs[model_name] = ModelRunResult(
                model_name=model_name,
                training_result=result,
                metrics=metrics,
                predictions_df=predictions,
                builtin_importance=builtin_importance,
                permutation_importance=perm_importance,
                metadata=metadata,
                dropped_columns=prepared_data.dropped_columns,
                feature_cols=prepared_data.feature_cols,
            )
            leaderboard_rows.append({"Model": model_name, **metrics})

        if not leaderboard_rows:
            raise RuntimeError("All selected models failed to train.")
        leaderboard = pd.DataFrame(leaderboard_rows)
        if train_config.task_type == "Classification":
            by_cols = [c for c in ["F1-Score", "ROC-AUC", "Accuracy"] if c in leaderboard.columns]
            leaderboard = leaderboard.sort_values(by=by_cols, ascending=False)
        elif train_config.task_type == "Regression":
            by_cols = [c for c in ["R2", "MAE"] if c in leaderboard.columns]
            leaderboard = leaderboard.sort_values(by=by_cols, ascending=[False, True][: len(by_cols)])
        else:
            leaderboard = leaderboard.sort_values(by=["Silhouette Score"], ascending=False)
        leaderboard = leaderboard.reset_index(drop=True)
        return TrainingSummary(
            leaderboard=leaderboard,
            runs=runs,
            best_model_name=str(leaderboard.iloc[0]["Model"]),
        )


@dataclass(slots=True)
class ServiceContainer:
    data_service: DataService
    warning_service: WarningService
    training_service: TrainingService


def build_service_container(
    *,
    data_config: DataLoadConfig,
    warning_config: WarningsConfig,
    runtime_config: RuntimeConfig,
    n_clusters: int = 3,
) -> ServiceContainer:
    registry = get_default_registry_cached(
        n_clusters=n_clusters,
        max_n_jobs=runtime_config.max_n_jobs,
    )
    trainer = ModelTrainer(
        optimizer=RandomizedSearchOptimizer(n_jobs=runtime_config.max_n_jobs),
        n_jobs=runtime_config.max_n_jobs,
    )
    training_service = TrainingService(
        model_registry=registry,
        preprocessor_factory=PreprocessorFactory(),
        trainer=trainer,
        metrics_service=MetricsService(),
    )
    return ServiceContainer(
        data_service=DataService(data_config),
        warning_service=WarningService(warning_config),
        training_service=training_service,
    )
