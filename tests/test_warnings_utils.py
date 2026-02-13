from __future__ import annotations

import pandas as pd

from automl_gui.config import WarningsConfig
from automl_gui.warnings_utils import analyze_smart_warnings


def test_detects_target_leakage_warning():
    size = 200
    target = pd.Series([0, 1] * (size // 2))
    df = pd.DataFrame(
        {
            "leaky_feature": target * 100,
            "noise": list(range(size)),
            "target": target,
        }
    )
    warnings = analyze_smart_warnings(
        df=df,
        feature_cols=["leaky_feature", "noise"],
        target_col="target",
        task_type="Classification",
        config=WarningsConfig(target_corr_threshold=0.9),
    )
    messages = " ".join(w.message for w in warnings)
    assert "data leakage" in messages.lower()
    assert any("leaky_feature" in w.message for w in warnings)


def test_detects_class_imbalance_warning():
    df = pd.DataFrame(
        {
            "feat": list(range(100)),
            "target": [0] * 90 + [1] * 10,
        }
    )
    warnings = analyze_smart_warnings(
        df=df,
        feature_cols=["feat"],
        target_col="target",
        task_type="Classification",
    )
    assert any("imbalance" in w.title.lower() for w in warnings)
