from __future__ import annotations

import pandas as pd
import pytest

from automl_gui.config import DataLoadConfig
from automl_gui.data_utils import (
    UploadValidationError,
    detect_id_like_columns,
    guess_default_features,
    load_uploaded_file,
    maybe_sample_large_dataframe,
    validate_uploaded_file,
)


def test_guess_default_features_excludes_obvious_id_columns():
    df = pd.DataFrame(
        {
            "user_id": [f"id_{i}" for i in range(100)],
            "feature_a": list(range(100)),
            "feature_b": [i * 2 for i in range(100)],
            "target": [0, 1] * 50,
        }
    )
    features = guess_default_features(df, target_col="target")
    assert "target" not in features
    assert "user_id" not in features
    assert set(features) == {"feature_a", "feature_b"}


def test_detect_id_like_columns_finds_high_unique_id_field():
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(200)],
            "value": list(range(200)),
        }
    )
    detected = detect_id_like_columns(df)
    assert "customer_id" in detected


def test_detect_id_like_columns_does_not_drop_unique_numeric_features():
    df = pd.DataFrame(
        {
            "feature_a": list(range(300)),
            "feature_b": [i * 3 for i in range(300)],
            "target": [0, 1] * 150,
        }
    )
    detected = detect_id_like_columns(df)
    assert "feature_a" not in detected
    assert "feature_b" not in detected


def test_validate_uploaded_file_rejects_large_files():
    with pytest.raises(UploadValidationError):
        validate_uploaded_file(
            file_name="large.csv",
            file_size_bytes=300 * 1024 * 1024,
            content_type="text/csv",
            max_upload_mb=200,
        )


def test_load_uploaded_file_csv():
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    df = load_uploaded_file(
        file_name="sample.csv",
        file_bytes=csv_bytes,
        config=DataLoadConfig(),
    )
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]


def test_maybe_sample_large_dataframe():
    df = pd.DataFrame({"x": range(2000)})
    sampled, sampled_flag = maybe_sample_large_dataframe(
        df,
        row_threshold=1000,
        memory_threshold_mb=1,
        sample_size=300,
    )
    assert sampled_flag is True
    assert len(sampled) == 300
