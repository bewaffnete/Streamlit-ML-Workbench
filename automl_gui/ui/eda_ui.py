"""EDA and warning tabs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data_utils import (
    correlation_frame,
    correlation_with_target,
    dtype_summary,
    guess_default_features,
    numeric_and_categorical_features,
)
from ..visualization import (
    make_categorical_count,
    make_corr_heatmap,
    make_missing_values_chart,
    make_numeric_distribution,
)
from .context import UIContext


def _select_index(options, value, default=0):
    return options.index(value) if value in options else default


def render_target_eda_tab(ctx: UIContext) -> None:
    st.subheader("2) Target & Feature Selection + EDA")
    df = ctx.state.get("df")
    if df is None:
        st.info("Upload data first.")
        return

    targets = ["<None>"] + df.columns.tolist()
    ctx.state.set(
        "target_col",
        st.selectbox(
            "Target column",
            options=targets,
            index=_select_index(targets, ctx.state.get("target_col")),
        ),
    )
    target_col = ctx.state.get("target_col")
    feature_options = [c for c in df.columns if c != target_col]
    defaults = [f for f in ctx.state.get("selected_features", []) if f in feature_options]
    if not defaults:
        defaults = guess_default_features(df, target_col)
    ctx.state.set(
        "selected_features",
        st.multiselect("Feature columns", options=feature_options, default=defaults),
    )
    selected = ctx.state.get("selected_features", [])
    if not selected:
        st.warning("Select features to continue.")
        return

    numeric, categorical = numeric_and_categorical_features(df, selected)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(dtype_summary(df[selected]), use_container_width=True)
    with c2:
        missing_fig = make_missing_values_chart(df, selected)
        if missing_fig is not None:
            st.plotly_chart(missing_fig, use_container_width=True)

    st.markdown("#### Numeric")
    if numeric:
        cols = st.columns(2)
        for idx, feat in enumerate(numeric[:10]):
            fig = make_numeric_distribution(df, feat)
            if fig is not None:
                cols[idx % 2].plotly_chart(fig, use_container_width=True)
    st.markdown("#### Categorical")
    if categorical:
        cols = st.columns(2)
        for idx, feat in enumerate(categorical[:8]):
            cols[idx % 2].plotly_chart(make_categorical_count(df, feat), use_container_width=True)

    if len(numeric) >= 2:
        corr_input = df[numeric].copy()
        if target_col != "<None>" and target_col in df.columns:
            target_series = df[target_col]
            corr_input[target_col] = (
                target_series if pd.api.types.is_numeric_dtype(target_series) else pd.factorize(target_series)[0]
            )
        pearson = correlation_frame(corr_input, method="pearson")
        spearman = correlation_frame(corr_input, method="spearman")
        a, b = st.columns(2)
        if not pearson.empty:
            a.plotly_chart(make_corr_heatmap(pearson, "Pearson Correlation"), use_container_width=True)
        if not spearman.empty:
            b.plotly_chart(make_corr_heatmap(spearman, "Spearman Correlation"), use_container_width=True)

        if target_col != "<None>":
            corr_target = correlation_with_target(
                df,
                target_col=target_col,
                feature_cols=numeric,
                method="pearson",
            )
            if not corr_target.empty:
                # `corr_target` is a Series: reset_index then rename the index column to 'feature'
                corr_df = corr_target.rename("pearson_r").reset_index()
                corr_df = corr_df.rename(columns={"index": "feature"})
                st.dataframe(corr_df, use_container_width=True)


def render_warning_tab(ctx: UIContext) -> None:
    st.subheader("3) Smart Data Warnings")
    df = ctx.state.get("df")
    selected_features = ctx.state.get("selected_features", [])
    if df is None or not selected_features:
        st.info("Complete upload and feature selection first.")
        return

    warning_config = ctx.services.warning_service.config
    warning_config.target_corr_threshold = float(ctx.state.get("warn_target_corr_threshold"))
    warning_config.feature_corr_threshold = float(ctx.state.get("warn_feature_corr_threshold"))
    warning_config.missing_threshold = float(ctx.state.get("warn_missing_threshold"))
    warning_config.high_cardinality_threshold = int(ctx.state.get("warn_high_cardinality_threshold"))

    target = ctx.state.get("target_col")
    target_col = target if target != "<None>" else None
    warnings = ctx.services.warning_service.analyze(
        df=df,
        feature_cols=selected_features,
        target_col=target_col,
        task_type=ctx.state.get("task_type"),
    )
    if not warnings:
        st.success("No major warning signals detected.")
        return
    for warning in warnings:
        if warning.level == "error":
            st.error(f"**{warning.title}**\n\n{warning.message}")
        elif warning.level == "warning":
            st.warning(f"**{warning.title}**\n\n{warning.message}")
        else:
            st.info(f"**{warning.title}**\n\n{warning.message}")
