from __future__ import annotations

import numpy as np
import pandas as pd

from automl_gui.evaluation import MetricsService


def test_collect_classification_metrics_binary_without_prob():
    service = MetricsService()
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    bundle = service.collect_and_format_metrics(
        task_type="Classification",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
    )
    assert "Accuracy" in bundle.metrics
    assert "ROC-AUC" in bundle.metrics


def test_collect_classification_metrics_multiclass_with_prob():
    service = MetricsService()
    y_true = pd.Series([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    y_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    bundle = service.collect_and_format_metrics(
        task_type="Classification",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    assert "PR-AUC" in bundle.metrics
    assert isinstance(bundle.metrics["PR-AUC"], float)
