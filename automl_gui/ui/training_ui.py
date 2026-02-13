"""Training/evaluation tab with background job orchestration."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from ..config import TrainConfig
from ..core.jobs import TrainingJobPayload
from ..evaluation import (
    classification_report_frame,
    confusion_matrix_figure,
    feature_importance_figure,
    regression_scatter_figure,
)
from ..logging_utils import get_logger
from ..preprocessing import DataCleaningRequest, PreprocessorConfig
from .context import UIContext

LOGGER = get_logger("ui.training")


def _select_index(options, value, default=0):
    return options.index(value) if value in options else default


def _build_preprocessor_config(ctx: UIContext) -> PreprocessorConfig:
    return PreprocessorConfig(
        missing_strategy=ctx.state.get("missing_strategy"),
        encoding_strategy=ctx.state.get("encoding_strategy"),
        scaling_strategy=ctx.state.get("scaling_strategy"),
        constant_fill_value=ctx.state.get("constant_fill_value"),
        one_hot_max_categories=int(ctx.state.get("one_hot_max_categories")),
        use_polynomial_features=bool(ctx.state.get("use_polynomial_features")),
        interaction_only=bool(ctx.state.get("interaction_only")),
        per_column_missing_strategy=dict(ctx.state.get("per_column_missing_strategy") or {}),
    )


def _build_train_config(ctx: UIContext) -> TrainConfig:
    return TrainConfig(
        task_type=ctx.state.get("task_type"),
        evaluation_mode=ctx.state.get("evaluation_mode"),
        train_size=float(ctx.state.get("train_size")),
        cv_folds=int(ctx.state.get("cv_folds")),
        stratify=bool(ctx.state.get("stratify")),
        enable_hyperparameter_tuning=bool(ctx.state.get("enable_hyperparameter_tuning")),
        tuning_iterations=int(ctx.state.get("tuning_iterations")),
        enable_early_stopping=bool(ctx.state.get("enable_early_stopping")),
        random_state=42,
        n_clusters=int(ctx.state.get("n_clusters")),
        selected_models=list(ctx.state.get("selected_models", [])),
    )


def _build_cleaning_request(ctx: UIContext) -> DataCleaningRequest:
    return DataCleaningRequest(
        feature_cols=list(ctx.state.get("selected_features", [])),
        target_col=ctx.state.get("target_col") if ctx.state.get("target_col") != "<None>" else None,
        task_type=ctx.state.get("task_type"),
        missing_strategy=ctx.state.get("missing_strategy"),
        drop_column_threshold=float(ctx.state.get("drop_column_threshold")),
        remove_duplicates=bool(ctx.state.get("remove_duplicates")),
        outlier_strategy=ctx.state.get("outlier_strategy"),
        winsor_lower_quantile=float(ctx.state.get("winsor_lower_quantile")),
        winsor_upper_quantile=float(ctx.state.get("winsor_upper_quantile")),
        iqr_multiplier=float(ctx.state.get("iqr_multiplier")),
        per_column_missing_strategy=dict(ctx.state.get("per_column_missing_strategy") or {}),
        constant_fill_value=ctx.state.get("constant_fill_value"),
    )


def _submit_training_job(ctx: UIContext) -> None:
    df = ctx.state.get("df")
    if df is None:
        raise ValueError("Upload data before training.")
    pre_cfg = _build_preprocessor_config(ctx)
    train_cfg = _build_train_config(ctx)
    if not train_cfg.selected_models:
        raise ValueError("Select at least one model in 'Model Config'.")
    cleaning = _build_cleaning_request(ctx)

    payload = TrainingJobPayload(
        df=df,
        task_type=train_cfg.task_type,
        target_col=cleaning.target_col,
        selected_features=list(cleaning.feature_cols),
        cleaning_request=cleaning.to_dict(),
        preprocessor_config=pre_cfg.to_dict(),
        train_config=train_cfg.to_dict(),
        dataset_fingerprint=ctx.state.get("dataset_fingerprint"),
        run_permutation_importance=bool(ctx.state.get("run_permutation_importance")),
        runtime_config=ctx.runtime_config.to_dict(),
    )
    job_id = ctx.job_manager.submit_training(payload)
    ctx.state.clear_training_artifacts()
    ctx.state.update(
        {
            "training_job_id": job_id,
            "training_job_status": "queued",
            "training_job_error": None,
        }
    )


def _poll_training_job(ctx: UIContext) -> None:
    job_id = ctx.state.get("training_job_id")
    if not job_id:
        return
    status = ctx.job_manager.status(job_id)
    ctx.state.set("training_job_status", status)
    if status == "completed":
        summary = ctx.job_manager.result(job_id)
        ctx.state.set("training_results", summary.runs)
        ctx.state.set("leaderboard_df", summary.leaderboard)
        ctx.state.set("best_model_name", summary.best_model_name)
        ctx.job_manager.cleanup(job_id)
        ctx.state.set("training_job_id", None)
    elif status == "failed":
        error = ctx.job_manager.error(job_id)
        ctx.state.set("training_job_error", error)
        ctx.job_manager.cleanup(job_id)
        ctx.state.set("training_job_id", None)


def _render_job_status(ctx: UIContext) -> None:
    status = ctx.state.get("training_job_status")
    if not status:
        return
    if status in {"queued", "running"}:
        with st.status(f"Training job {status}...", expanded=True):
            st.write("Models are training in a background process.")
        if st.button("Cancel Training Job", use_container_width=True):
            job_id = ctx.state.get("training_job_id")
            if job_id and ctx.job_manager.cancel(job_id):
                ctx.state.update(
                    {
                        "training_job_status": "cancelled",
                        "training_job_id": None,
                        "training_job_error": None,
                    }
                )
                st.warning("Training job cancelled.")
    elif status == "completed":
        st.success("Training job completed.")
    elif status == "failed":
        st.error(f"Training job failed: {ctx.state.get('training_job_error')}")
    elif status == "cancelled":
        st.warning("Training job cancelled.")


def _render_results(ctx: UIContext) -> None:
    training_results = ctx.state.get("training_results")
    if not training_results:
        st.info("No training results available yet.")
        return

    leaderboard = ctx.state.get("leaderboard_df")
    if leaderboard is not None:
        st.dataframe(leaderboard, use_container_width=True)

    names = list(training_results.keys())
    selected = st.selectbox(
        "Inspect model",
        options=names,
        index=_select_index(names, ctx.state.get("best_model_name")),
    )
    run = training_results[selected]
    result = run.training_result
    metrics = run.metrics
    st.caption(result.mode_note)
    if run.dropped_columns:
        st.caption(f"Dropped columns: {', '.join(run.dropped_columns)}")
    if result.train_seconds is not None:
        st.caption(f"Train time: {result.train_seconds:.2f}s")

    if ctx.state.get("task_type") == "Classification":
        cols = st.columns(6)
        for idx, key in enumerate(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]):
            value = metrics.get(key, float("nan"))
            cols[idx].metric(key, "N/A" if pd.isna(value) else f"{value:.4f}")
        st.plotly_chart(confusion_matrix_figure(result.y_true, result.y_pred), use_container_width=True)
        st.dataframe(classification_report_frame(result.y_true, result.y_pred), use_container_width=True)
    elif ctx.state.get("task_type") == "Regression":
        cols = st.columns(5)
        for idx, key in enumerate(["MSE", "RMSE", "MAE", "R2", "MAPE"]):
            value = metrics.get(key, float("nan"))
            cols[idx].metric(key, "N/A" if pd.isna(value) else f"{value:.4f}")
        st.plotly_chart(regression_scatter_figure(result.y_true, result.y_pred), use_container_width=True)
    else:
        score = metrics.get("Silhouette Score", float("nan"))
        st.metric("Silhouette Score", "N/A" if pd.isna(score) else f"{score:.4f}")

    if run.builtin_importance is not None and not run.builtin_importance.empty:
        st.plotly_chart(feature_importance_figure(run.builtin_importance, "Built-in Importance"), use_container_width=True)
    if run.permutation_importance is not None and not run.permutation_importance.empty:
        st.plotly_chart(
            feature_importance_figure(run.permutation_importance, "Permutation Importance"),
            use_container_width=True,
        )


def render_training_tab(ctx: UIContext) -> None:
    st.subheader("6) Train & Evaluate")
    if ctx.state.get("df") is None:
        st.info("Upload data first.")
        return

    if st.button("Start Training", type="primary", use_container_width=True):
        try:
            _submit_training_job(ctx)
            st.success("Training job submitted.")
        except Exception as exc:
            LOGGER.warning("Training submission failed: %s", exc, exc_info=True)
            st.error(f"Could not start training: {exc}")

    _poll_training_job(ctx)
    _render_job_status(ctx)
    _render_results(ctx)

    # Keep progress responsive without requiring manual clicks.
    if ctx.state.get("training_job_status") in {"queued", "running"}:
        time.sleep(1.0)
        st.rerun()
