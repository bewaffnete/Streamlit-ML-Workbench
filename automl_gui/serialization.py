"""Safer serialization helpers for project and model metadata."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn


def _json_safe(value: Any):
    """Convert common scientific Python values to JSON-safe primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return value


def dumps_json(payload: dict[str, Any]) -> bytes:
    """Serialize dict payload to UTF-8 JSON bytes."""
    return json.dumps(payload, default=_json_safe, indent=2).encode("utf-8")


def loads_json(text: str) -> dict[str, Any]:
    """Deserialize JSON text into dict."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")
    return obj


def build_project_payload(settings: dict[str, Any], dataset_fingerprint: str | None) -> dict[str, Any]:
    """Build safe JSON project payload (no pickle/joblib)."""
    return {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "settings": settings,
    }


def build_model_metadata(
    *,
    model_name: str,
    task_type: str,
    metrics: dict[str, Any],
    dataset_fingerprint: str | None,
    feature_cols: list[str],
    preprocessing_summary: dict[str, Any],
    train_seconds: float,
) -> dict[str, Any]:
    """Create model metadata record for auditing/reproducibility."""
    return {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "task_type": task_type,
        "metrics": metrics,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_columns": feature_cols,
        "preprocessing": preprocessing_summary,
        "train_seconds": train_seconds,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }
