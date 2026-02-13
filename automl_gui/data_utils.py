"""Data ingestion, validation, and analysis helpers."""

from __future__ import annotations

import hashlib
from io import BytesIO, StringIO
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DataLoadConfig
from .logging_utils import get_logger

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

LOGGER = get_logger("data")

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
}
ID_TOKEN_PATTERN = re.compile(r"(?:^|[_-])(id|uuid|index|key)(?:$|[_-])")


class UploadValidationError(ValueError):
    """Raised when uploaded file does not pass validation checks."""


class DataLoadError(ValueError):
    """Raised when file content cannot be parsed into a DataFrame."""


def file_extension(file_name: str) -> str:
    """Get lowercase file extension with leading dot."""
    if "." not in file_name:
        return ""
    return "." + file_name.rsplit(".", 1)[-1].lower()


def validate_uploaded_file(
    file_name: str,
    file_size_bytes: int,
    content_type: str | None,
    *,
    max_upload_mb: int,
    file_bytes: bytes | None = None,
) -> None:
    """Validate extension, content type, and file-size limits."""
    extension = file_extension(file_name)
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file type. Allowed types: CSV, Excel (.xlsx/.xls), Parquet."
        )

    max_bytes = max_upload_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise UploadValidationError(
            f"File too large ({file_size_bytes / (1024 ** 2):.1f} MB). Limit is {max_upload_mb} MB."
        )

    if content_type and extension in {".csv", ".xlsx", ".xls"}:
        if content_type not in ALLOWED_MIME_TYPES:
            raise UploadValidationError(
                f"Suspicious content type '{content_type}'. Please verify file integrity."
            )
    if file_bytes is not None:
        _validate_magic_bytes(extension=extension, file_bytes=file_bytes)


def _validate_magic_bytes(*, extension: str, file_bytes: bytes) -> None:
    sample = file_bytes[:16]
    if extension == ".parquet":
        if not (len(file_bytes) >= 4 and file_bytes[:4] == b"PAR1"):
            raise UploadValidationError("Invalid parquet signature.")
        return
    if extension == ".xlsx":
        if not sample.startswith(b"PK"):
            raise UploadValidationError("Invalid XLSX signature.")
        return
    if extension == ".xls":
        if not sample.startswith(b"\xD0\xCF\x11\xE0"):
            raise UploadValidationError("Invalid XLS signature.")
        return
    if extension == ".csv":
        null_ratio = file_bytes[:1024].count(0) / max(min(len(file_bytes), 1024), 1)
        if null_ratio > 0.02:
            raise UploadValidationError("CSV appears to contain binary content.")


def _read_csv_with_fallback(
    file_bytes: bytes,
    *,
    config: DataLoadConfig,
    nrows: int | None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in config.csv_encodings:
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
                low_memory=config.csv_low_memory,
                on_bad_lines=config.csv_on_bad_lines,
                nrows=nrows,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            last_error = exc
            continue
    raise DataLoadError(f"Failed to parse CSV with supported encodings: {last_error}")


def _sample_df(
    df: pd.DataFrame,
    *,
    sample_rows: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_rows is None:
        return df
    size = min(sample_rows, len(df))
    if size <= 0 or size >= len(df):
        return df
    return df.sample(n=size, random_state=random_state)


def load_uploaded_file(
    *,
    file_name: str,
    file_bytes: bytes,
    config: DataLoadConfig,
    nrows: int | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Load CSV/Excel/Parquet bytes into DataFrame with robust parsing."""
    extension = file_extension(file_name)
    try:
        if extension == ".csv":
            df = _read_csv_with_fallback(file_bytes, config=config, nrows=nrows)
        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(BytesIO(file_bytes), nrows=nrows)
        elif extension == ".parquet":
            df = pd.read_parquet(BytesIO(file_bytes))
            if nrows is not None:
                df = df.head(nrows)
        else:
            raise UploadValidationError("Unsupported file format.")
    except (UploadValidationError, DataLoadError):
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        LOGGER.warning("Data parse error for file '%s': %s", file_name, exc, exc_info=True)
        raise DataLoadError(f"Could not parse dataset: {exc}") from exc
    except OSError as exc:
        LOGGER.warning("I/O error for file '%s': %s", file_name, exc, exc_info=True)
        raise DataLoadError(f"Could not read dataset: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected load error for file '%s'", file_name, exc_info=True)
        raise DataLoadError("Unexpected error while loading dataset.") from exc

    return _sample_df(df, sample_rows=sample_rows)


def maybe_sample_large_dataframe(
    df: pd.DataFrame,
    *,
    row_threshold: int = 500_000,
    memory_threshold_mb: int = 1024,
    sample_size: int = 50_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """Sample large datasets to keep interactivity responsive."""
    is_large = len(df) > row_threshold or memory_usage_mb(df) > memory_threshold_mb
    if not is_large:
        return df, False
    safe_size = min(sample_size, len(df))
    return df.sample(safe_size, random_state=random_state), True


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Estimate DataFrame memory footprint in MB."""
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic fingerprint from sample hash + schema descriptors."""
    sampled = df if len(df) <= 20_000 else df.sample(20_000, random_state=42)
    value_hash = pd.util.hash_pandas_object(sampled, index=True).values.tobytes()
    digest = hashlib.sha256(value_hash).hexdigest()
    dtype_sig = "|".join(f"{c}:{str(t)}" for c, t in sorted(df.dtypes.items(), key=lambda x: x[0]))
    col_sig = "|".join(sorted(df.columns.astype(str).tolist()))
    schema_hash = hashlib.sha256(f"{dtype_sig}::{col_sig}".encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{schema_hash}:{df.shape[0]}x{df.shape[1]}"


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level type and null statistics."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "n_unique": df.nunique(dropna=True).values,
        }
    )


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact data type summary."""
    return (
        df.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    """Detect obvious ID-style columns to exclude from default feature list."""
    id_like: list[str] = []
    row_count = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        col_lower = col.lower()
        unique_ratio = series.nunique(dropna=True) / row_count

        # Name-based heuristic (user_id, customer_uuid, index_key, etc.)
        name_match = ID_TOKEN_PATTERN.search(col_lower)
        if name_match and unique_ratio >= 0.75:
            id_like.append(col)
            continue

        # Conservative fallback: very high-cardinality *string-like* columns.
        # Avoid dropping numeric features solely due to uniqueness.
        is_string_like = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if unique_ratio >= 0.995 and is_string_like:
            id_like.append(col)
    return sorted(set(id_like))


def guess_default_features(df: pd.DataFrame, target_col: str | None) -> list[str]:
    """Default feature set excluding target and obvious ID-like columns."""
    id_like = set(detect_id_like_columns(df))
    columns = [c for c in df.columns if c != target_col and c not in id_like]
    if not columns:
        columns = [c for c in df.columns if c != target_col]
    return columns


def numeric_and_categorical_features(
    df: pd.DataFrame,
    features: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Split selected features by dtype."""
    selected = [f for f in features if f in df.columns]
    numeric = df[selected].select_dtypes(include=np.number).columns.tolist()
    categorical = [f for f in selected if f not in numeric]
    return numeric, categorical


def correlation_frame(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation frame for numeric columns."""
    if df.empty:
        return pd.DataFrame()
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method=method)


def correlation_with_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    method: str = "pearson",
) -> pd.Series:
    """Feature-to-target correlation with categorical target fallback."""
    valid_features = [f for f in feature_cols if f in df.columns and f != target_col]
    numeric_features = df[valid_features].select_dtypes(include=np.number).columns.tolist()
    if not numeric_features or target_col not in df.columns:
        return pd.Series(dtype=float)

    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        target = pd.Series(pd.factorize(target)[0], index=df.index, name=target_col)

    corr_df = pd.concat([df[numeric_features], target], axis=1).corr(method=method)
    if target_col not in corr_df.columns:
        return pd.Series(dtype=float)
    return corr_df[target_col].drop(labels=[target_col]).sort_values(ascending=False)


def dataframe_info_text(df: pd.DataFrame) -> str:
    """Capture `df.info()` output as plain text."""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


if st is not None:
    cache_data = st.cache_data
    cache_resource = st.cache_resource
else:  # pragma: no cover
    def cache_data(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator

    def cache_resource(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@cache_resource(show_spinner=False)
def dataset_store() -> dict[str, pd.DataFrame]:
    """Long-lived in-memory dataframe store keyed by fingerprint."""
    return {}


def register_dataset(fingerprint: str, df: pd.DataFrame) -> None:
    dataset_store()[fingerprint] = df


def get_registered_dataset(fingerprint: str) -> pd.DataFrame:
    store = dataset_store()
    if fingerprint not in store:
        raise KeyError(f"Dataset fingerprint '{fingerprint}' not found in cache.")
    return store[fingerprint]


def purge_dataset(fingerprint: str | None = None) -> None:
    store = dataset_store()
    if fingerprint is None:
        store.clear()
    else:
        store.pop(fingerprint, None)


@cache_data(show_spinner=False, max_entries=64)
def cached_head(fingerprint: str, n: int = 10) -> pd.DataFrame:
    """Cached head preview by fingerprint key."""
    return get_registered_dataset(fingerprint).head(n).copy()


@cache_data(show_spinner=False, max_entries=64)
def cached_describe(fingerprint: str) -> pd.DataFrame:
    """Cached describe summary by fingerprint key."""
    return get_registered_dataset(fingerprint).describe(include="all").transpose()


@cache_data(show_spinner=False, max_entries=64)
def cached_info_text(fingerprint: str) -> str:
    """Cached df.info text by fingerprint key."""
    return dataframe_info_text(get_registered_dataset(fingerprint))


@cache_data(show_spinner=False, max_entries=64)
def cached_dataset_overview(fingerprint: str) -> pd.DataFrame:
    """Cached overview by fingerprint key."""
    return dataset_overview(get_registered_dataset(fingerprint))


def dataset_cache_key(fingerprint: str, config: dict[str, Any]) -> str:
    """Build stable cache key from fingerprint + config hash."""
    config_str = str(sorted(config.items()))
    digest = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:10]
    return f"{fingerprint}:{digest}"
