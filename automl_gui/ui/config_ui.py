"""Preprocessing and model-configuration tabs."""

from __future__ import annotations

import streamlit as st

from ..models import unavailable_boosting_libraries
from ..preprocessing import (
    ADVANCED_PER_COLUMN_STRATEGIES,
    ENCODING_STRATEGIES,
    MISSING_STRATEGIES,
    OUTLIER_STRATEGIES,
    SCALING_STRATEGIES,
)
from .context import UIContext

TASK_TYPES = ["Classification", "Regression", "Clustering"]
EVALUATION_MODES = ["Train/Test Split", "Cross Validation"]
CV_OPTIONS = [3, 5, 10]


def _select_index(options, value, default=0):
    return options.index(value) if value in options else default


def render_preprocessing_tab(ctx: UIContext) -> None:
    st.subheader("4) Preprocessing")
    if ctx.state.get("df") is None:
        st.info("Upload data first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        ctx.state.set(
            "missing_strategy",
            st.selectbox(
                "Missing values",
                options=MISSING_STRATEGIES,
                index=_select_index(MISSING_STRATEGIES, ctx.state.get("missing_strategy")),
            ),
        )
        if ctx.state.get("missing_strategy") == "Constant":
            ctx.state.set(
                "constant_fill_value",
                st.text_input("Constant fill value", value=ctx.state.get("constant_fill_value")),
            )
        if ctx.state.get("missing_strategy") == "Drop columns":
            ctx.state.set(
                "drop_column_threshold",
                st.slider(
                    "Drop columns threshold",
                    min_value=0.3,
                    max_value=0.95,
                    value=float(ctx.state.get("drop_column_threshold")),
                    step=0.05,
                ),
            )
    with c2:
        ctx.state.set(
            "encoding_strategy",
            st.selectbox(
                "Encoding",
                options=ENCODING_STRATEGIES,
                index=_select_index(ENCODING_STRATEGIES, ctx.state.get("encoding_strategy")),
            ),
        )
        if ctx.state.get("encoding_strategy") == "OneHotEncoder":
            ctx.state.set(
                "one_hot_max_categories",
                st.slider(
                    "OneHot max categories",
                    min_value=5,
                    max_value=200,
                    value=int(ctx.state.get("one_hot_max_categories")),
                    step=5,
                ),
            )
        ctx.state.set(
            "scaling_strategy",
            st.selectbox(
                "Scaling",
                options=SCALING_STRATEGIES,
                index=_select_index(SCALING_STRATEGIES, ctx.state.get("scaling_strategy")),
            ),
        )
    with c3:
        ctx.state.set(
            "outlier_strategy",
            st.selectbox(
                "Outlier handling",
                options=OUTLIER_STRATEGIES,
                index=_select_index(OUTLIER_STRATEGIES, ctx.state.get("outlier_strategy")),
            ),
        )
        if ctx.state.get("outlier_strategy") == "Clip (Winsorize)":
            ctx.state.set(
                "winsor_lower_quantile",
                st.slider(
                    "Lower quantile",
                    min_value=0.0,
                    max_value=0.1,
                    value=float(ctx.state.get("winsor_lower_quantile")),
                    step=0.01,
                ),
            )
            ctx.state.set(
                "winsor_upper_quantile",
                st.slider(
                    "Upper quantile",
                    min_value=0.9,
                    max_value=1.0,
                    value=float(ctx.state.get("winsor_upper_quantile")),
                    step=0.01,
                ),
            )
        if ctx.state.get("outlier_strategy") == "IQR Removal":
            ctx.state.set(
                "iqr_multiplier",
                st.slider(
                    "IQR multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=float(ctx.state.get("iqr_multiplier")),
                    step=0.1,
                ),
            )

    p1, p2, p3 = st.columns(3)
    with p1:
        ctx.state.set(
            "remove_duplicates",
            st.checkbox("Remove duplicates", value=bool(ctx.state.get("remove_duplicates"))),
        )
    with p2:
        ctx.state.set(
            "use_polynomial_features",
            st.checkbox(
                "Polynomial features (degree 2)",
                value=bool(ctx.state.get("use_polynomial_features")),
            ),
        )
        if ctx.state.get("use_polynomial_features"):
            ctx.state.set(
                "interaction_only",
                st.checkbox(
                    "Interaction-only terms",
                    value=bool(ctx.state.get("interaction_only")),
                ),
            )
    with p3:
        ctx.state.set(
            "run_permutation_importance",
            st.checkbox(
                "Permutation importance",
                value=bool(ctx.state.get("run_permutation_importance")),
            ),
        )
        ctx.state.set(
            "run_shap",
            st.checkbox("SHAP summary (manual)", value=bool(ctx.state.get("run_shap"))),
        )

    ctx.state.set(
        "advanced_column_mode",
        st.checkbox(
            "Advanced per-column missing strategy",
            value=bool(ctx.state.get("advanced_column_mode")),
        ),
    )
    if ctx.state.get("advanced_column_mode") and ctx.state.get("selected_features"):
        with st.expander("Per-column missing overrides", expanded=False):
            overrides = dict(ctx.state.get("per_column_missing_strategy") or {})
            for col in ctx.state.get("selected_features")[:40]:
                key = f"override_{col}"
                current = overrides.get(col, "Default")
                selected = st.selectbox(
                    col,
                    options=ADVANCED_PER_COLUMN_STRATEGIES,
                    index=_select_index(ADVANCED_PER_COLUMN_STRATEGIES, current),
                    key=key,
                )
                if selected == "Default":
                    overrides.pop(col, None)
                else:
                    overrides[col] = selected
            ctx.state.set("per_column_missing_strategy", overrides)


def render_model_config_tab(ctx: UIContext) -> None:
    st.subheader("5) Task & Models")
    df = ctx.state.get("df")
    if df is None:
        st.info("Upload data first.")
        return

    ctx.state.set(
        "task_type",
        st.radio(
            "Task",
            options=TASK_TYPES,
            index=_select_index(TASK_TYPES, ctx.state.get("task_type")),
            horizontal=True,
        ),
    )
    if ctx.state.get("task_type") == "Clustering":
        ctx.state.set(
            "n_clusters",
            st.slider("Number of clusters", min_value=2, max_value=12, value=int(ctx.state.get("n_clusters"))),
        )

    available = ctx.services.training_service.available_models(ctx.state.get("task_type"))
    defaults = [m for m in ctx.state.get("selected_models", []) if m in available]
    if not defaults:
        defaults = available[: min(3, len(available))]
    ctx.state.set(
        "selected_models",
        st.multiselect("Models to train", options=available, default=defaults),
    )

    if ctx.state.get("task_type") != "Clustering":
        ctx.state.set(
            "evaluation_mode",
            st.radio(
                "Evaluation",
                options=EVALUATION_MODES,
                index=_select_index(EVALUATION_MODES, ctx.state.get("evaluation_mode")),
                horizontal=True,
            ),
        )
        if ctx.state.get("evaluation_mode") == "Train/Test Split":
            ctx.state.set(
                "train_size",
                st.slider(
                    "Train size",
                    min_value=0.70,
                    max_value=0.95,
                    value=float(ctx.state.get("train_size")),
                    step=0.05,
                ),
            )
        else:
            ctx.state.set(
                "cv_folds",
                st.selectbox(
                    "CV folds",
                    options=CV_OPTIONS,
                    index=_select_index(CV_OPTIONS, ctx.state.get("cv_folds"), default=1),
                ),
            )
        if ctx.state.get("task_type") == "Classification":
            ctx.state.set(
                "stratify",
                st.checkbox("Stratify split", value=bool(ctx.state.get("stratify"))),
            )

    with st.expander("Advanced training options", expanded=False):
        ctx.state.set(
            "enable_hyperparameter_tuning",
            st.checkbox(
                "Enable tuning (RandomizedSearchCV)",
                value=bool(ctx.state.get("enable_hyperparameter_tuning")),
            ),
        )
        if ctx.state.get("enable_hyperparameter_tuning"):
            ctx.state.set(
                "tuning_iterations",
                st.slider(
                    "Tuning iterations",
                    min_value=10,
                    max_value=50,
                    value=int(ctx.state.get("tuning_iterations")),
                    step=5,
                ),
            )
        ctx.state.set(
            "enable_early_stopping",
            st.checkbox(
                "Enable early stopping for boosting",
                value=bool(ctx.state.get("enable_early_stopping")),
            ),
        )

    from ..config import TrainConfig

    train_cfg = TrainConfig(
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
    st.info(
        ctx.services.training_service.estimate_cost(
            n_rows=len(df),
            n_features=max(len(ctx.state.get("selected_features", [])), 1),
            train_config=train_cfg,
        )
    )
    missing = [k for k, v in unavailable_boosting_libraries().items() if not v]
    if missing:
        st.info(f"Optional libs unavailable: {', '.join(missing)}")
