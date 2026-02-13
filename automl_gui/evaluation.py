"""Metrics collection and model interpretability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(slots=True)
class MetricsBundle:
    task_type: str
    metrics: dict[str, float]


class MetricsService:
    """Unified metrics collection for classification/regression/clustering."""

    @staticmethod
    def _classification_metrics(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_prob is None:
            metrics["ROC-AUC"] = float("nan")
            metrics["PR-AUC"] = float("nan")
            return metrics

        try:
            labels = np.unique(y_true)
            if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
                metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
                metrics["PR-AUC"] = float(average_precision_score(y_true, y_prob))
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                metrics["PR-AUC"] = float(average_precision_score(y_true, y_prob[:, 1]))
            else:
                y_bin = label_binarize(y_true, classes=labels)
                metrics["ROC-AUC"] = float(
                    roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
                )
                metrics["PR-AUC"] = float(
                    average_precision_score(y_bin, y_prob, average="weighted")
                )
        except (ValueError, TypeError):
            metrics["ROC-AUC"] = float("nan")
            metrics["PR-AUC"] = float("nan")
        return metrics

    @staticmethod
    def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        metrics = {
            "MSE": float(mean_squared_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }
        try:
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_true, y_pred))
        except (ValueError, TypeError, ZeroDivisionError):
            metrics["MAPE"] = float("nan")
        return metrics

    @staticmethod
    def _clustering_metrics(transformed_x, labels: np.ndarray) -> dict[str, float]:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) == len(labels):
            return {"Silhouette Score": float("nan")}
        return {"Silhouette Score": float(silhouette_score(transformed_x, labels))}

    def collect_and_format_metrics(
        self,
        *,
        task_type: str,
        y_true: pd.Series | None,
        y_pred: np.ndarray | None,
        y_prob: np.ndarray | None = None,
        transformed_x=None,
        labels: np.ndarray | None = None,
    ) -> MetricsBundle:
        if task_type == "Classification":
            if y_true is None or y_pred is None:
                raise ValueError("Classification metrics require y_true and y_pred.")
            metrics = self._classification_metrics(y_true, y_pred, y_prob)
            return MetricsBundle(task_type=task_type, metrics=metrics)
        if task_type == "Regression":
            if y_true is None or y_pred is None:
                raise ValueError("Regression metrics require y_true and y_pred.")
            metrics = self._regression_metrics(y_true, y_pred)
            return MetricsBundle(task_type=task_type, metrics=metrics)
        if task_type == "Clustering":
            if labels is None:
                raise ValueError("Clustering metrics require labels.")
            metrics = self._clustering_metrics(transformed_x, labels)
            return MetricsBundle(task_type=task_type, metrics=metrics)
        raise ValueError(f"Unsupported task type: {task_type}")


def confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )


def classification_report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose().reset_index()
    return df.rename(columns={"index": "label"})


def regression_scatter_figure(y_true, y_pred):
    scatter_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        title="Prediction vs Actual",
        template="plotly_white",
    )
    minimum = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
    maximum = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(color="red", dash="dash"),
    )
    return fig


def feature_importance_frame(model_pipeline) -> pd.DataFrame | None:
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    model = model_pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return None
    feature_names = (
        preprocessor.get_feature_names_out()
        if hasattr(preprocessor, "get_feature_names_out")
        else None
    )
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        importances = np.mean(np.abs(coefs), axis=0) if np.ndim(coefs) == 2 else np.abs(coefs)
    elif hasattr(model, "cluster_centers_"):
        importances = np.std(model.cluster_centers_, axis=0)

    if importances is None:
        return None
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]
    if len(feature_names) != len(importances):
        return None
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": np.asarray(importances).ravel()})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_frame(
    model_pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    task_type: str,
    n_repeats: int = 5,
) -> pd.DataFrame | None:
    if X_eval is None or y_eval is None or len(X_eval) < 10:
        return None
    scoring = "f1_weighted" if task_type == "Classification" else "r2"
    result = permutation_importance(
        model_pipeline,
        X_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring,
        n_jobs=-1,
    )
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    names = (
        preprocessor.get_feature_names_out()
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out")
        else [f"feature_{idx}" for idx in range(len(result.importances_mean))]
    )
    if len(names) != len(result.importances_mean):
        return None
    return (
        pd.DataFrame({"Feature": names, "Importance": result.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def feature_importance_figure(importance_df: pd.DataFrame, title: str = "Feature Importance"):
    top_df = importance_df.head(25).sort_values("Importance", ascending=True)
    return px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        template="plotly_white",
    )
