"""Configurable preprocessing factory and cleaning helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

MISSING_STRATEGIES = [
    "Mean",
    "Median",
    "Mode",
    "Constant",
    "KNNImputer",
    "IterativeImputer",
    "Drop rows",
    "Drop columns",
]
ENCODING_STRATEGIES = [
    "OneHotEncoder",
    "LabelEncoder",
    "FrequencyEncoding",
    "TargetEncoding",
    "CatBoostEncoding",
]
SCALING_STRATEGIES = ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
OUTLIER_STRATEGIES = ["None", "Clip (Winsorize)", "IQR Removal"]
ADVANCED_PER_COLUMN_STRATEGIES = ["Default", "Mean", "Median", "Mode", "Constant"]


@dataclass(slots=True)
class PreprocessorConfig:
    """Serializable preprocessing configuration DTO."""

    missing_strategy: str = "Median"
    encoding_strategy: str = "OneHotEncoder"
    scaling_strategy: str = "None"
    constant_fill_value: str = "missing"
    one_hot_max_categories: int = 40
    use_polynomial_features: bool = False
    interaction_only: bool = False
    per_column_missing_strategy: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessorConfig":
        return cls(
            missing_strategy=str(payload.get("missing_strategy", "Median")),
            encoding_strategy=str(payload.get("encoding_strategy", "OneHotEncoder")),
            scaling_strategy=str(payload.get("scaling_strategy", "None")),
            constant_fill_value=str(payload.get("constant_fill_value", "missing")),
            one_hot_max_categories=int(payload.get("one_hot_max_categories", 40)),
            use_polynomial_features=bool(payload.get("use_polynomial_features", False)),
            interaction_only=bool(payload.get("interaction_only", False)),
            per_column_missing_strategy=dict(payload.get("per_column_missing_strategy", {})),
        )


@dataclass(slots=True)
class DataCleaningRequest:
    """Input contract for dataset cleaning before split/training."""

    feature_cols: list[str]
    target_col: str | None
    task_type: str
    missing_strategy: str
    drop_column_threshold: float
    remove_duplicates: bool
    outlier_strategy: str
    winsor_lower_quantile: float
    winsor_upper_quantile: float
    iqr_multiplier: float
    per_column_missing_strategy: dict[str, str] = field(default_factory=dict)
    constant_fill_value: str = "missing"

    def to_dict(self) -> dict[str, Any]:
        """Serialize request for background job payloads."""
        return asdict(self)


@dataclass(slots=True)
class DataCleaningResult:
    """Stable cleaning output model."""

    frame: pd.DataFrame
    feature_cols: list[str]
    dropped_columns: list[str]
    dropped_rows: int


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, pd.Series):
        return X.to_frame()
    return pd.DataFrame(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency/count encoder for categorical features."""

    def fit(self, X, y=None):
        X_df = _to_dataframe(X).astype(str)
        self.columns_ = X_df.columns.tolist()
        self.maps_ = {
            col: X_df[col].fillna("__MISSING__").value_counts(normalize=True).to_dict()
            for col in self.columns_
        }
        return self

    def transform(self, X):
        X_df = _to_dataframe(X).astype(str)
        transformed = []
        for col in self.columns_:
            mapper = self.maps_[col]
            transformed.append(
                X_df[col].fillna("__MISSING__").map(mapper).fillna(0.0).astype(float).to_numpy()
            )
        return np.vstack(transformed).T

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_freq" for col in self.columns_], dtype=object)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding for categoricals."""

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoding requires target values.")
        X_df = _to_dataframe(X).astype(str)
        y_series = pd.Series(y).reset_index(drop=True)
        if not pd.api.types.is_numeric_dtype(y_series):
            y_series = pd.Series(pd.factorize(y_series)[0], index=y_series.index)

        self.columns_ = X_df.columns.tolist()
        self.global_mean_ = float(y_series.mean())
        self.maps_ = {}
        frame = X_df.reset_index(drop=True).copy()
        frame["__target__"] = y_series.values
        for col in self.columns_:
            stats = frame.groupby(col, observed=False)["__target__"].agg(["mean", "count"])
            smooth = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X_df = _to_dataframe(X).astype(str)
        transformed = []
        for col in self.columns_:
            transformed.append(
                X_df[col].map(self.maps_[col]).fillna(self.global_mean_).astype(float).to_numpy()
            )
        return np.vstack(transformed).T

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_target_mean" for col in self.columns_], dtype=object)


class CatBoostLikeEncoder(TargetMeanEncoder):
    """Approximate CatBoost encoding (smoothed mean target)."""

    def __init__(self):
        super().__init__(smoothing=20.0)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_catboost_enc" for col in self.columns_], dtype=object)


def _numeric_imputer(strategy: str, constant_fill_value: str):
    if strategy == "Mean":
        return SimpleImputer(strategy="mean")
    if strategy == "Median":
        return SimpleImputer(strategy="median")
    if strategy == "Mode":
        return SimpleImputer(strategy="most_frequent")
    if strategy == "Constant":
        return SimpleImputer(strategy="constant", fill_value=constant_fill_value)
    if strategy == "KNNImputer":
        return KNNImputer(n_neighbors=5)
    if strategy == "IterativeImputer":
        return IterativeImputer(random_state=42, max_iter=10)
    if strategy == "Drop rows":
        return None
    return SimpleImputer(strategy="median")


def _categorical_imputer(strategy: str, constant_fill_value: str):
    if strategy == "Constant":
        return SimpleImputer(strategy="constant", fill_value=constant_fill_value)
    if strategy == "Drop rows":
        return None
    return SimpleImputer(strategy="most_frequent")


def _scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    return None


def _one_hot_encoder(max_categories: int):
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            max_categories=max_categories,
        )
    except TypeError:  # scikit-learn fallback
        return OneHotEncoder(handle_unknown="ignore")


def _categorical_encoder(name: str, max_categories: int):
    if name == "OneHotEncoder":
        return _one_hot_encoder(max_categories)
    if name == "LabelEncoder":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "FrequencyEncoding":
        return FrequencyEncoder()
    if name == "TargetEncoding":
        return TargetMeanEncoder()
    if name == "CatBoostEncoding":
        return CatBoostLikeEncoder()
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _group_columns_by_strategy(
    columns: list[str],
    global_strategy: str,
    overrides: dict[str, str] | None,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        strategy = global_strategy
        if overrides and col in overrides and overrides[col] != "Default":
            strategy = overrides[col]
        grouped[strategy].append(col)
    return dict(grouped)


class PreprocessorFactory:
    """Builds `ColumnTransformer` instances from `PreprocessorConfig`."""

    def build(self, X: pd.DataFrame, config: PreprocessorConfig) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [col for col in X.columns if col not in numeric_features]

        transformers: list[tuple[str, Any, list[str]]] = []

        numeric_groups = _group_columns_by_strategy(
            numeric_features,
            config.missing_strategy,
            config.per_column_missing_strategy,
        )
        for strategy, cols in numeric_groups.items():
            if not cols:
                continue
            steps = []
            imputer = _numeric_imputer(strategy, config.constant_fill_value)
            if imputer is not None:
                steps.append(("imputer", imputer))
            if config.use_polynomial_features:
                steps.append(
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2,
                            include_bias=False,
                            interaction_only=config.interaction_only,
                        ),
                    )
                )
            scaler = _scaler(config.scaling_strategy)
            if scaler is not None:
                steps.append(("scaler", scaler))
            pipeline = Pipeline(steps) if steps else "passthrough"
            transformers.append((f"numeric_{strategy}", pipeline, cols))

        categorical_groups = _group_columns_by_strategy(
            categorical_features,
            config.missing_strategy,
            config.per_column_missing_strategy,
        )
        for strategy, cols in categorical_groups.items():
            if not cols:
                continue
            steps = []
            imputer = _categorical_imputer(strategy, config.constant_fill_value)
            if imputer is not None:
                steps.append(("imputer", imputer))
            steps.append(
                ("encoder", _categorical_encoder(config.encoding_strategy, config.one_hot_max_categories))
            )
            pipeline = Pipeline(steps) if steps else "passthrough"
            transformers.append((f"categorical_{strategy}", pipeline, cols))

        if not transformers:
            raise ValueError("No valid features available after preprocessing setup.")
        return ColumnTransformer(transformers=transformers, remainder="drop")


def clean_dataset_before_split(df: pd.DataFrame, request: DataCleaningRequest) -> DataCleaningResult:
    """Apply row/column-level cleaning before split/training."""
    feature_cols = [c for c in request.feature_cols if c in df.columns]
    cols = list(feature_cols)
    if request.task_type != "Clustering" and request.target_col:
        cols.append(request.target_col)

    work_df = df[cols].copy()
    rows_before = len(work_df)
    if request.remove_duplicates:
        work_df = work_df.drop_duplicates()
    if request.task_type != "Clustering" and request.target_col:
        work_df = work_df.loc[work_df[request.target_col].notna()]

    dropped_columns: list[str] = []
    if request.missing_strategy == "Drop rows":
        work_df = work_df.dropna()
    elif request.missing_strategy == "Drop columns":
        missing_ratio = work_df[feature_cols].isna().mean()
        dropped_columns = (
            missing_ratio[missing_ratio >= request.drop_column_threshold].index.sort_values().tolist()
        )
        work_df = work_df.drop(columns=dropped_columns, errors="ignore")
        feature_cols = [col for col in feature_cols if col not in dropped_columns]

    overrides = request.per_column_missing_strategy or {}
    for col, strategy in overrides.items():
        if col not in work_df.columns or strategy == "Default":
            continue
        if strategy == "Mean" and pd.api.types.is_numeric_dtype(work_df[col]):
            work_df[col] = work_df[col].fillna(work_df[col].mean())
        elif strategy == "Median" and pd.api.types.is_numeric_dtype(work_df[col]):
            work_df[col] = work_df[col].fillna(work_df[col].median())
        elif strategy == "Mode":
            mode = work_df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else request.constant_fill_value
            work_df[col] = work_df[col].fillna(fill)
        elif strategy == "Constant":
            work_df[col] = work_df[col].fillna(request.constant_fill_value)

    numeric_features = work_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    if request.outlier_strategy == "Clip (Winsorize)" and numeric_features:
        for col in numeric_features:
            lower = work_df[col].quantile(request.winsor_lower_quantile)
            upper = work_df[col].quantile(request.winsor_upper_quantile)
            work_df[col] = work_df[col].clip(lower=lower, upper=upper)
    elif request.outlier_strategy == "IQR Removal" and numeric_features:
        mask = pd.Series(True, index=work_df.index)
        for col in numeric_features:
            q1 = work_df[col].quantile(0.25)
            q3 = work_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - request.iqr_multiplier * iqr
            upper = q3 + request.iqr_multiplier * iqr
            col_mask = work_df[col].isna() | ((work_df[col] >= lower) & (work_df[col] <= upper))
            mask = mask & col_mask
        work_df = work_df.loc[mask]

    return DataCleaningResult(
        frame=work_df,
        feature_cols=feature_cols,
        dropped_columns=dropped_columns,
        dropped_rows=max(rows_before - len(work_df), 0),
    )


def sanitize_selected_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> list[str]:
    """Ensure selected features are present and unique."""
    clean: list[str] = []
    for col in feature_cols:
        if col in df.columns and col not in clean:
            clean.append(col)
    return clean


def build_preprocessor(
    X: pd.DataFrame,
    *,
    missing_strategy: str,
    encoding_strategy: str,
    scaling_strategy: str,
    constant_fill_value: str,
    one_hot_max_categories: int,
    use_polynomial_features: bool,
    interaction_only: bool,
    per_column_missing_strategy: dict[str, str] | None = None,
) -> ColumnTransformer:
    """Backward-compatible wrapper for previous API."""
    config = PreprocessorConfig(
        missing_strategy=missing_strategy,
        encoding_strategy=encoding_strategy,
        scaling_strategy=scaling_strategy,
        constant_fill_value=constant_fill_value,
        one_hot_max_categories=one_hot_max_categories,
        use_polynomial_features=use_polynomial_features,
        interaction_only=interaction_only,
        per_column_missing_strategy=per_column_missing_strategy or {},
    )
    return PreprocessorFactory().build(X, config)
