"""Session-state facade and strict project state serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .serialization import dumps_json, loads_json

SCHEMA_VERSION = 3


class StateValidationError(ValueError):
    """Raised when serialized state payload is invalid."""


DEFAULT_STATE: dict[str, Any] = {
    "df": None,
    "raw_df": None,
    "dataset_name": None,
    "dataset_fingerprint": None,
    "raw_dataset_fingerprint": None,
    "target_col": "<None>",
    "selected_features": [],
    "task_type": "Classification",
    "missing_strategy": "Median",
    "drop_column_threshold": 0.7,
    "constant_fill_value": "missing",
    "encoding_strategy": "OneHotEncoder",
    "one_hot_max_categories": 40,
    "scaling_strategy": "None",
    "remove_duplicates": False,
    "outlier_strategy": "None",
    "winsor_lower_quantile": 0.01,
    "winsor_upper_quantile": 0.99,
    "iqr_multiplier": 1.5,
    "use_polynomial_features": False,
    "interaction_only": False,
    "evaluation_mode": "Train/Test Split",
    "train_size": 0.8,
    "cv_folds": 5,
    "stratify": True,
    "selected_models": [],
    "n_clusters": 3,
    "run_permutation_importance": False,
    "run_shap": False,
    "training_results": {},
    "leaderboard_df": None,
    "best_model_name": None,
    "debug_logging": False,
    "log_to_file": False,
    "sample_large_data": True,
    "sample_size_rows": 50_000,
    "sampled_from_raw": False,
    "advanced_column_mode": False,
    "per_column_missing_strategy": {},
    "enable_hyperparameter_tuning": False,
    "tuning_iterations": 25,
    "enable_early_stopping": True,
    "warn_target_corr_threshold": 0.92,
    "warn_feature_corr_threshold": 0.95,
    "warn_missing_threshold": 0.65,
    "warn_high_cardinality_threshold": 50,
    "profile_html": None,
    "training_job_id": None,
    "training_job_status": None,
    "training_job_error": None,
    "trusted_model_loading": False,
}

PROJECT_KEYS = [
    "dataset_name",
    "dataset_fingerprint",
    "raw_dataset_fingerprint",
    "target_col",
    "selected_features",
    "task_type",
    "missing_strategy",
    "drop_column_threshold",
    "constant_fill_value",
    "encoding_strategy",
    "one_hot_max_categories",
    "scaling_strategy",
    "remove_duplicates",
    "outlier_strategy",
    "winsor_lower_quantile",
    "winsor_upper_quantile",
    "iqr_multiplier",
    "use_polynomial_features",
    "interaction_only",
    "evaluation_mode",
    "train_size",
    "cv_folds",
    "stratify",
    "selected_models",
    "n_clusters",
    "run_permutation_importance",
    "run_shap",
    "sample_large_data",
    "sample_size_rows",
    "advanced_column_mode",
    "per_column_missing_strategy",
    "enable_hyperparameter_tuning",
    "tuning_iterations",
    "enable_early_stopping",
    "warn_target_corr_threshold",
    "warn_feature_corr_threshold",
    "warn_missing_threshold",
    "warn_high_cardinality_threshold",
]

_SCHEMA_TYPES: dict[str, type | tuple[type, ...]] = {
    "dataset_name": (str, type(None)),
    "dataset_fingerprint": (str, type(None)),
    "raw_dataset_fingerprint": (str, type(None)),
    "target_col": str,
    "selected_features": list,
    "task_type": str,
    "missing_strategy": str,
    "drop_column_threshold": (float, int),
    "constant_fill_value": str,
    "encoding_strategy": str,
    "one_hot_max_categories": int,
    "scaling_strategy": str,
    "remove_duplicates": bool,
    "outlier_strategy": str,
    "winsor_lower_quantile": (float, int),
    "winsor_upper_quantile": (float, int),
    "iqr_multiplier": (float, int),
    "use_polynomial_features": bool,
    "interaction_only": bool,
    "evaluation_mode": str,
    "train_size": (float, int),
    "cv_folds": int,
    "stratify": bool,
    "selected_models": list,
    "n_clusters": int,
    "run_permutation_importance": bool,
    "run_shap": bool,
    "sample_large_data": bool,
    "sample_size_rows": int,
    "advanced_column_mode": bool,
    "per_column_missing_strategy": dict,
    "enable_hyperparameter_tuning": bool,
    "tuning_iterations": int,
    "enable_early_stopping": bool,
    "warn_target_corr_threshold": (float, int),
    "warn_feature_corr_threshold": (float, int),
    "warn_missing_threshold": (float, int),
    "warn_high_cardinality_threshold": int,
}


@dataclass(slots=True)
class AppState:
    """Thin typed facade over Streamlit session_state."""

    session: Any

    def get(self, key: str, default: Any = None) -> Any:
        return self.session.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.session[key] = value

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self.session[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return self.session.pop(key, default)

    def clear_all(self, *, preserve: Iterable[str] = ()) -> None:
        keep = set(preserve)
        for key in list(self.session.keys()):
            if key not in keep:
                del self.session[key]

    def clear_training_artifacts(self) -> None:
        self.set("training_results", {})
        self.set("leaderboard_df", None)
        self.set("best_model_name", None)
        self.set("training_job_id", None)
        self.set("training_job_status", None)
        self.set("training_job_error", None)


def init_session_state(session_state) -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in session_state:
            session_state[key] = value


def _validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    unknown = set(settings.keys()) - set(PROJECT_KEYS)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise StateValidationError(f"Unknown config keys: {unknown_list}")

    validated: dict[str, Any] = {}
    for key in PROJECT_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        expected = _SCHEMA_TYPES.get(key)
        if expected is not None and not isinstance(value, expected):
            raise StateValidationError(
                f"Invalid type for '{key}': expected {expected}, got {type(value)}"
            )
        validated[key] = value
    return validated


def export_project_state(session_state, *, fmt: str = "json") -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "settings": {key: session_state.get(key) for key in PROJECT_KEYS},
    }
    if fmt == "json":
        return dumps_json(payload)
    if fmt == "yaml":
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise StateValidationError("YAML export requested but pyyaml is not installed.") from exc
        return yaml.safe_dump(payload).encode("utf-8")
    raise StateValidationError(f"Unsupported format: {fmt}")


def import_project_state(file_bytes: bytes, *, file_name: str = "") -> dict[str, Any]:
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else "json"
    if suffix in {"yaml", "yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise StateValidationError("YAML import requires pyyaml.") from exc
        raw = yaml.safe_load(file_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise StateValidationError("Invalid YAML payload.")
        payload = raw
    else:
        payload = loads_json(file_bytes.decode("utf-8"))

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise StateValidationError(
            f"Unsupported schema version: {version}. Expected: {SCHEMA_VERSION}."
        )
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise StateValidationError("Invalid state payload: missing settings object.")
    return _validate_settings(settings)


def model_trust_warning() -> str:
    return (
        "Only load models from trusted sources. Pickle/joblib deserialization can execute "
        "arbitrary code."
    )
