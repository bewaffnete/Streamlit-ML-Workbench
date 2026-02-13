"""Extensible model registry, defaults, and tuning spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from .config import ModelConfig
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ModelFactory = Callable[[dict[str, Any]], object]


@dataclass(slots=True)
class RegisteredModel:
    name: str
    task_type: str
    factory: ModelFactory
    default_params: dict[str, Any]


class ModelRegistry:
    """Plugin-style registry for model factory functions."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, RegisteredModel]] = {
            "Classification": {},
            "Regression": {},
            "Clustering": {},
        }

    def register_model(
        self,
        *,
        task_type: str,
        name: str,
        factory_func: ModelFactory,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        if task_type not in self._registry:
            raise ValueError(f"Unsupported task_type: {task_type}")
        self._registry[task_type][name] = RegisteredModel(
            name=name,
            task_type=task_type,
            factory=factory_func,
            default_params=default_params or {},
        )

    def available_models(self, task_type: str) -> list[str]:
        if task_type not in self._registry:
            raise ValueError(f"Unsupported task_type: {task_type}")
        return sorted(self._registry[task_type].keys())

    def create_model(
        self,
        *,
        task_type: str,
        name: str,
        override_params: dict[str, Any] | None = None,
    ):
        if task_type not in self._registry or name not in self._registry[task_type]:
            raise ValueError(f"Model '{name}' not registered for {task_type}.")
        reg = self._registry[task_type][name]
        params = dict(reg.default_params)
        params.update(override_params or {})
        return reg.factory(params)


def unavailable_boosting_libraries() -> dict[str, bool]:
    """Availability map for optional boosting dependencies."""
    return {
        "XGBoost": HAS_XGBOOST,
        "LightGBM": HAS_LIGHTGBM,
        "CatBoost": HAS_CATBOOST,
    }


def build_default_registry(
    *,
    random_state: int = 42,
    n_clusters: int = 3,
    model_config: ModelConfig | None = None,
    max_n_jobs: int = 2,
) -> ModelRegistry:
    """Build default model registry with optional per-model param overrides."""
    cfg = model_config or ModelConfig()
    registry = ModelRegistry()

    registry.register_model(
        task_type="Classification",
        name="LogisticRegression",
        default_params={"max_iter": 2500, "class_weight": "balanced"},
        factory_func=lambda p: LogisticRegression(**p),
    )
    registry.register_model(
        task_type="Classification",
        name="RandomForest",
        default_params={
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": max_n_jobs,
            "class_weight": "balanced_subsample",
        },
        factory_func=lambda p: RandomForestClassifier(**p),
    )
    registry.register_model(
        task_type="Classification",
        name="ExtraTrees",
        default_params={
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": max_n_jobs,
            "class_weight": "balanced",
        },
        factory_func=lambda p: ExtraTreesClassifier(**p),
    )
    registry.register_model(
        task_type="Classification",
        name="SVM",
        default_params={"kernel": "rbf", "probability": True, "class_weight": "balanced"},
        factory_func=lambda p: SVC(**p),
    )
    registry.register_model(
        task_type="Classification",
        name="KNN",
        default_params={"n_neighbors": 7},
        factory_func=lambda p: KNeighborsClassifier(**p),
    )
    registry.register_model(
        task_type="Classification",
        name="GradientBoosting",
        default_params={"random_state": random_state},
        factory_func=lambda p: GradientBoostingClassifier(**p),
    )

    registry.register_model(
        task_type="Regression",
        name="RandomForest",
        default_params={"n_estimators": 300, "random_state": random_state, "n_jobs": max_n_jobs},
        factory_func=lambda p: RandomForestRegressor(**p),
    )
    registry.register_model(
        task_type="Regression",
        name="ExtraTrees",
        default_params={"n_estimators": 300, "random_state": random_state, "n_jobs": max_n_jobs},
        factory_func=lambda p: ExtraTreesRegressor(**p),
    )
    registry.register_model(
        task_type="Regression",
        name="SVM",
        default_params={"kernel": "rbf"},
        factory_func=lambda p: SVR(**p),
    )
    registry.register_model(
        task_type="Regression",
        name="KNN",
        default_params={"n_neighbors": 7},
        factory_func=lambda p: KNeighborsRegressor(**p),
    )
    registry.register_model(
        task_type="Regression",
        name="GradientBoosting",
        default_params={"random_state": random_state},
        factory_func=lambda p: GradientBoostingRegressor(**p),
    )

    registry.register_model(
        task_type="Clustering",
        name="KMeans",
        default_params={"n_clusters": n_clusters, "n_init": 20, "random_state": random_state},
        factory_func=lambda p: KMeans(**p),
    )

    if HAS_XGBOOST:
        registry.register_model(
            task_type="Classification",
            name="XGBoost",
            default_params={
                "n_estimators": 400,
                "random_state": random_state,
                "eval_metric": "logloss",
                "n_jobs": max_n_jobs,
            },
            factory_func=lambda p: XGBClassifier(**p),
        )
        registry.register_model(
            task_type="Regression",
            name="XGBoost",
            default_params={"n_estimators": 400, "random_state": random_state, "n_jobs": max_n_jobs},
            factory_func=lambda p: XGBRegressor(**p),
        )

    if HAS_LIGHTGBM:
        registry.register_model(
            task_type="Classification",
            name="LightGBM",
            default_params={"n_estimators": 400, "random_state": random_state},
            factory_func=lambda p: LGBMClassifier(**p),
        )
        registry.register_model(
            task_type="Regression",
            name="LightGBM",
            default_params={"n_estimators": 400, "random_state": random_state},
            factory_func=lambda p: LGBMRegressor(**p),
        )

    if HAS_CATBOOST:
        registry.register_model(
            task_type="Classification",
            name="CatBoost",
            default_params={"verbose": 0, "random_state": random_state},
            factory_func=lambda p: CatBoostClassifier(**p),
        )
        registry.register_model(
            task_type="Regression",
            name="CatBoost",
            default_params={"verbose": 0, "random_state": random_state},
            factory_func=lambda p: CatBoostRegressor(**p),
        )

    for task_type in ("Classification", "Regression", "Clustering"):
        config_map = getattr(cfg, task_type.lower(), {})
        for model_name, override in config_map.items():
            if model_name not in registry._registry[task_type]:
                continue
            registry._registry[task_type][model_name].default_params.update(override)

    return registry


def model_param_distributions(task_type: str, model_name: str) -> dict[str, list]:
    """RandomizedSearchCV parameter spaces."""
    spaces: dict[str, dict[str, list]] = {
        "Classification": {
            "LogisticRegression": {"C": [0.01, 0.1, 1.0, 5.0, 10.0]},
            "RandomForest": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "ExtraTrees": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "SVM": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "KNN": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
            "GradientBoosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 5],
            },
            "XGBoost": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.9, 1.0],
            },
            "LightGBM": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
            },
            "CatBoost": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [200, 400, 600],
            },
        },
        "Regression": {
            "RandomForest": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "ExtraTrees": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "SVM": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1]},
            "KNN": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
            "GradientBoosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 5],
            },
            "XGBoost": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.9, 1.0],
            },
            "LightGBM": {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
            },
            "CatBoost": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [200, 400, 600],
            },
        },
    }
    return spaces.get(task_type, {}).get(model_name, {})


def estimate_training_cost(
    *,
    n_rows: int,
    n_features: int,
    n_models: int,
    tuning_enabled: bool,
    tuning_iterations: int,
) -> str:
    """Coarse training-time estimate for UX guidance."""
    complexity = n_rows * max(n_features, 1) * max(n_models, 1)
    if tuning_enabled:
        complexity *= max(tuning_iterations, 1)

    if complexity < 2_000_000:
        return "Estimated cost: low (seconds to <1 minute)."
    if complexity < 20_000_000:
        return "Estimated cost: moderate (1-5 minutes)."
    return "Estimated cost: high (5+ minutes). Consider sampling or fewer models."


if st is not None:
    _cache_resource = st.cache_resource
else:  # pragma: no cover
    def _cache_resource(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@_cache_resource(show_spinner=False)
def get_default_registry_cached(
    *,
    random_state: int = 42,
    n_clusters: int = 3,
    max_n_jobs: int = 2,
) -> ModelRegistry:
    """Cached registry for long-lived app sessions."""
    return build_default_registry(
        random_state=random_state,
        n_clusters=n_clusters,
        model_config=None,
        max_n_jobs=max_n_jobs,
    )
