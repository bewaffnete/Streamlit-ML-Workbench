"""Configurable automated data quality warnings and hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import WarningsConfig
from .data_utils import correlation_with_target


@dataclass(slots=True)
class SmartWarning:
    level: str
    title: str
    message: str


def _sample_warning_frame(
    df: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=random_state)


def _top_numeric_features(df: pd.DataFrame, feature_cols: list[str], max_features: int) -> list[str]:
    numeric_features = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    return numeric_features[:max_features]


def _feature_pair_warnings(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    *,
    max_numeric_features: int,
) -> list[SmartWarning]:
    numeric_features = _top_numeric_features(df, feature_cols, max_numeric_features)
    if len(numeric_features) < 2:
        return []

    corr = df[numeric_features].corr().abs()
    warnings: list[SmartWarning] = []
    for i, col_i in enumerate(numeric_features):
        for col_j in numeric_features[i + 1 :]:
            value = corr.loc[col_i, col_j]
            if value >= threshold:
                warnings.append(
                    SmartWarning(
                        level="warning",
                        title="High feature-feature correlation",
                        message=(
                            f"Features '{col_i}' and '{col_j}' are highly correlated "
                            f"(r = {value:.3f}). Consider removing one to reduce multicollinearity."
                        ),
                    )
                )
    return warnings


def _target_correlation_warnings(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    threshold: float,
) -> list[SmartWarning]:
    corr = correlation_with_target(df, target_col=target_col, feature_cols=feature_cols)
    if corr.empty:
        return []
    strong = corr[corr.abs() >= threshold]
    return [
        SmartWarning(
            level="error",
            title="Potential leakage warning",
            message=(
                f"Feature '{feature}' has very strong correlation with target (r = {value:.3f}). "
                "This may cause data leakage or a trivial model. Consider removing or investigating."
            ),
        )
        for feature, value in strong.items()
    ]


def _imbalance_warning(y: pd.Series) -> list[SmartWarning]:
    counts = y.value_counts(dropna=True)
    if counts.empty or len(counts) < 2:
        return []
    ratio = counts.max() / max(counts.min(), 1)
    if ratio <= 3:
        return []
    pct = (counts / counts.sum() * 100).round(2)
    summary = " / ".join([f"{cls}: {value}%" for cls, value in pct.items()])
    level = "error" if ratio > 5 else "warning"
    return [
        SmartWarning(
            level=level,
            title="Class imbalance detected",
            message=(
                f"Strong class imbalance detected (classes: {summary}, ratio {ratio:.2f}:1). "
                "Consider SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, "
                "class_weight='balanced', focal loss, and balanced metrics (F1, PR-AUC)."
            ),
        )
    ]


def _missingness_warnings(df: pd.DataFrame, feature_cols: list[str], threshold: float) -> list[SmartWarning]:
    missing_ratio = df[feature_cols].isna().mean()
    high_missing = missing_ratio[missing_ratio >= threshold]
    return [
        SmartWarning(
            level="warning",
            title="High missingness",
            message=(
                f"Feature '{feature}' has {value * 100:.1f}% missing values. "
                "Consider dropping this column or using robust imputation."
            ),
        )
        for feature, value in high_missing.items()
    ]


def _high_cardinality_warnings(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    threshold: int,
) -> list[SmartWarning]:
    warnings = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique_count = df[col].nunique(dropna=True)
        if unique_count > threshold:
            warnings.append(
                SmartWarning(
                    level="info",
                    title="High cardinality categorical feature",
                    message=(
                        f"Feature '{col}' has {unique_count} unique values. "
                        "Consider target encoding or grouping rare categories."
                    ),
                )
            )
    return warnings


def analyze_smart_warnings(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None,
    task_type: str,
    config: WarningsConfig | None = None,
) -> list[SmartWarning]:
    """Generate prioritized warnings/hints from selected data."""
    cfg = config or WarningsConfig()
    if df.empty or not feature_cols:
        return []

    sampled = _sample_warning_frame(df, max_rows=cfg.max_rows_for_warning_scan)
    warnings: list[SmartWarning] = []
    warnings.extend(_missingness_warnings(sampled, feature_cols, cfg.missing_threshold))
    warnings.extend(_high_cardinality_warnings(sampled, feature_cols, cfg.high_cardinality_threshold))
    warnings.extend(
        _feature_pair_warnings(
            sampled,
            feature_cols,
            cfg.feature_corr_threshold,
            max_numeric_features=cfg.top_numeric_features_for_pairwise,
        )
    )

    if target_col and target_col in sampled.columns and task_type != "Clustering":
        warnings.extend(
            _target_correlation_warnings(
                sampled,
                target_col,
                feature_cols,
                cfg.target_corr_threshold,
            )
        )
        if task_type == "Classification":
            warnings.extend(_imbalance_warning(sampled[target_col]))

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    warnings.sort(key=lambda item: severity_rank.get(item.level, 3))
    return warnings
