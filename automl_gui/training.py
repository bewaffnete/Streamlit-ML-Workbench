"""Training orchestration primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .config import TrainConfig


class TrainStrategy(str, Enum):
    HOLDOUT = "Train/Test Split"
    CV = "Cross Validation"


@dataclass(slots=True)
class TrainingRequest:
    X: pd.DataFrame
    y: pd.Series | None
    task_type: str
    model_name: str
    model: Any
    preprocessor: Any
    config: TrainConfig
    param_distributions: dict[str, list] | None = None


@dataclass(slots=True)
class TrainingResult:
    pipeline: Pipeline
    y_true: pd.Series | None
    y_pred: np.ndarray | None
    y_prob: np.ndarray | None
    X_eval: pd.DataFrame
    y_eval: pd.Series | None
    mode_note: str
    train_seconds: float | None
    transformed: Any | None = None
    labels: np.ndarray | None = None


class HyperparameterOptimizer(Protocol):
    def optimize(
        self,
        estimator,
        *,
        X,
        y,
        task_type: str,
        param_distributions: dict[str, list],
        n_iter: int,
        cv_folds: int,
    ):
        ...


class RandomizedSearchOptimizer:
    """RandomizedSearchCV wrapper for DI-friendly tuning."""

    def __init__(self, *, n_jobs: int = 2) -> None:
        self.n_jobs = n_jobs

    def optimize(
        self,
        estimator,
        *,
        X,
        y,
        task_type: str,
        param_distributions: dict[str, list],
        n_iter: int,
        cv_folds: int,
    ):
        if not param_distributions:
            return estimator
        scoring = "f1_weighted" if task_type == "Classification" else "r2"
        cv = min(max(cv_folds, 2), 5)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=min(max(n_iter, 1), 50),
            scoring=scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            random_state=42,
            error_score="raise",
        )
        search.fit(X, y)
        return search.best_estimator_


def _safe_predict_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except ValueError:
            return None
    return None


def _get_cv_splitter(task_type: str, y: pd.Series, cv_folds: int):
    if task_type == "Classification":
        min_class_count = y.value_counts().min()
        if min_class_count < 2:
            raise ValueError("At least 2 samples per class are required for cross-validation.")
        n_splits = min(cv_folds, int(min_class_count))
        if n_splits < 2:
            raise ValueError("Cross-validation requires at least 2 folds.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_splits = min(cv_folds, len(y))
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 rows.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


class ModelTrainer:
    """Unified trainer for holdout and CV with optional tuning plugin."""

    def __init__(self, optimizer: HyperparameterOptimizer | None = None, *, n_jobs: int = 2) -> None:
        self.optimizer = optimizer
        self.n_jobs = n_jobs

    def _fit_with_optional_early_stopping(
        self,
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        *,
        use_early_stopping: bool,
    ):
        model_name = model.__class__.__name__.lower()
        if not use_early_stopping or X_valid is None or y_valid is None:
            model.fit(X_train, y_train)
            return model

        if "xgb" in model_name:
            try:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                return model
            except TypeError:
                model.fit(X_train, y_train)
                return model
        if "lgbm" in model_name:
            try:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                return model
            except TypeError:
                model.fit(X_train, y_train)
                return model
        if "catboost" in model_name:
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    use_best_model=True,
                    verbose=False,
                )
                return model
            except TypeError:
                model.fit(X_train, y_train)
                return model
        model.fit(X_train, y_train)
        return model

    def train_supervised(self, request: TrainingRequest) -> TrainingResult:
        if request.y is None:
            raise ValueError("Supervised training requires target values.")

        cfg = request.config
        strategy = TrainStrategy(cfg.evaluation_mode)
        pipeline = Pipeline([("preprocessor", request.preprocessor), ("model", request.model)])

        if (
            cfg.enable_hyperparameter_tuning
            and self.optimizer is not None
            and request.param_distributions
        ):
            prefixed = {f"model__{k}": v for k, v in request.param_distributions.items()}
            pipeline = self.optimizer.optimize(
                pipeline,
                X=request.X,
                y=request.y,
                task_type=request.task_type,
                param_distributions=prefixed,
                n_iter=cfg.tuning_iterations,
                cv_folds=min(cfg.cv_folds, 3),
            )

        if strategy == TrainStrategy.CV:
            cv = _get_cv_splitter(request.task_type, request.y, cfg.cv_folds)
            y_pred = cross_val_predict(pipeline, request.X, request.y, cv=cv, method="predict")
            y_prob = None
            if request.task_type == "Classification":
                try:
                    y_prob = cross_val_predict(
                        pipeline, request.X, request.y, cv=cv, method="predict_proba"
                    )
                except ValueError:
                    y_prob = None
            pipeline.fit(request.X, request.y)
            return TrainingResult(
                pipeline=pipeline,
                y_true=request.y,
                y_pred=y_pred,
                y_prob=y_prob,
                X_eval=request.X,
                y_eval=request.y,
                mode_note=f"{cv.get_n_splits()}-fold cross-validation (OOF predictions).",
                train_seconds=None,
            )

        can_stratify = (
            request.task_type == "Classification"
            and cfg.stratify
            and request.y.nunique() > 1
            and request.y.value_counts().min() >= 2
        )
        split_stratify = request.y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            request.X,
            request.y,
            train_size=cfg.train_size,
            random_state=cfg.random_state,
            stratify=split_stratify,
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        y_pred = pipeline.predict(X_test)
        y_prob = _safe_predict_proba(pipeline, X_test) if request.task_type == "Classification" else None
        return TrainingResult(
            pipeline=pipeline,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            X_eval=X_test,
            y_eval=y_test,
            mode_note=f"Holdout split: {int(cfg.train_size * 100)}% train / {int((1-cfg.train_size) * 100)}% test.",
            train_seconds=train_seconds,
        )

    def fit_preprocessor_once(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor,
        config: TrainConfig,
        task_type: str,
    ) -> dict[str, Any]:
        can_stratify = (
            task_type == "Classification"
            and config.stratify
            and y.nunique() > 1
            and y.value_counts().min() >= 2
        )
        split_stratify = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=config.train_size,
            random_state=config.random_state,
            stratify=split_stratify,
        )
        fitted_pre = clone(preprocessor)
        fitted_pre.fit(X_train, y_train)
        return {
            "preprocessor": fitted_pre,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def train_supervised_with_shared_preprocessor(
        self,
        *,
        model,
        task_type: str,
        shared_bundle: dict[str, Any],
        config: TrainConfig,
        param_distributions: dict[str, list] | None = None,
    ) -> TrainingResult:
        X_train = shared_bundle["X_train"]
        X_test = shared_bundle["X_test"]
        y_train = shared_bundle["y_train"]
        y_test = shared_bundle["y_test"]
        preprocessor = shared_bundle["preprocessor"]
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        fitted_model = clone(model)
        if config.enable_hyperparameter_tuning and self.optimizer and param_distributions:
            fitted_model = self.optimizer.optimize(
                fitted_model,
                X=X_train_t,
                y=y_train,
                task_type=task_type,
                param_distributions=param_distributions,
                n_iter=config.tuning_iterations,
                cv_folds=3,
            )
            start = time.perf_counter()
            fitted_model.fit(X_train_t, y_train)
            train_seconds = time.perf_counter() - start
        else:
            start = time.perf_counter()
            fitted_model = self._fit_with_optional_early_stopping(
                fitted_model,
                X_train_t,
                y_train,
                X_test_t,
                y_test,
                use_early_stopping=config.enable_early_stopping,
            )
            train_seconds = time.perf_counter() - start

        y_pred = fitted_model.predict(X_test_t)
        y_prob = _safe_predict_proba(fitted_model, X_test_t) if task_type == "Classification" else None
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", fitted_model)])
        return TrainingResult(
            pipeline=pipeline,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            X_eval=X_test,
            y_eval=y_test,
            mode_note="Holdout split with shared fitted preprocessor.",
            train_seconds=train_seconds,
        )

    def train_clustering(self, request: TrainingRequest) -> TrainingResult:
        pipeline = Pipeline([("preprocessor", request.preprocessor), ("model", request.model)])
        start = time.perf_counter()
        labels = pipeline.fit_predict(request.X)
        train_seconds = time.perf_counter() - start
        transformed = pipeline.named_steps["preprocessor"].transform(request.X)
        return TrainingResult(
            pipeline=pipeline,
            y_true=None,
            y_pred=None,
            y_prob=None,
            X_eval=request.X,
            y_eval=None,
            mode_note="Clustering fit on full dataset.",
            train_seconds=train_seconds,
            transformed=transformed,
            labels=labels,
        )
