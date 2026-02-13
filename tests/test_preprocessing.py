from __future__ import annotations

import pandas as pd

from automl_gui.preprocessing import (
    DataCleaningRequest,
    PreprocessorConfig,
    PreprocessorFactory,
    build_preprocessor,
    clean_dataset_before_split,
)


def test_clean_dataset_drop_columns_missing():
    df = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": [None, None, None, 4],
            "target": [1, 0, 1, 0],
        }
    )
    request = DataCleaningRequest(
        feature_cols=["a", "b"],
        target_col="target",
        task_type="Classification",
        missing_strategy="Drop columns",
        drop_column_threshold=0.7,
        remove_duplicates=False,
        outlier_strategy="None",
        winsor_lower_quantile=0.01,
        winsor_upper_quantile=0.99,
        iqr_multiplier=1.5,
    )
    result = clean_dataset_before_split(df, request)
    assert "b" in result.dropped_columns
    assert "b" not in result.feature_cols
    assert "a" in result.frame.columns


def test_build_preprocessor_with_target_encoding():
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", "b", "a", None],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    preprocessor = build_preprocessor(
        X,
        missing_strategy="Median",
        encoding_strategy="TargetEncoding",
        scaling_strategy="StandardScaler",
        constant_fill_value="missing",
        one_hot_max_categories=30,
        use_polynomial_features=False,
        interaction_only=False,
    )
    transformed = preprocessor.fit_transform(X, y)
    assert transformed.shape[0] == len(X)


def test_preprocessor_config_roundtrip():
    config = PreprocessorConfig(
        missing_strategy="Median",
        encoding_strategy="FrequencyEncoding",
        scaling_strategy="RobustScaler",
        per_column_missing_strategy={"col_a": "Mode"},
    )
    loaded = PreprocessorConfig.from_dict(config.to_dict())
    assert loaded.missing_strategy == config.missing_strategy
    assert loaded.encoding_strategy == config.encoding_strategy
    factory = PreprocessorFactory()
    assert factory is not None


def test_data_cleaning_request_to_dict_roundtrip():
    request = DataCleaningRequest(
        feature_cols=["f1", "f2"],
        target_col="target",
        task_type="Classification",
        missing_strategy="Median",
        drop_column_threshold=0.7,
        remove_duplicates=True,
        outlier_strategy="None",
        winsor_lower_quantile=0.01,
        winsor_upper_quantile=0.99,
        iqr_multiplier=1.5,
        per_column_missing_strategy={"f1": "Mean"},
        constant_fill_value="missing",
    )
    payload = request.to_dict()
    loaded = DataCleaningRequest(**payload)
    assert loaded.feature_cols == request.feature_cols
    assert loaded.target_col == request.target_col
    assert loaded.per_column_missing_strategy == request.per_column_missing_strategy
