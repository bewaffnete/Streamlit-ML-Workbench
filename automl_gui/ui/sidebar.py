"""Sidebar controls and global settings."""

from __future__ import annotations

import streamlit as st

from ..logging_utils import configure_logging, get_logger
from ..state import PROJECT_KEYS, export_project_state, import_project_state
from .context import UIContext

LOGGER = get_logger("ui.sidebar")


def render_sidebar(ctx: UIContext) -> None:
    st.sidebar.title("Workflow")
    st.sidebar.caption("Operations")

    ctx.state.set(
        "debug_logging",
        st.sidebar.checkbox("Debug logging", value=bool(ctx.state.get("debug_logging"))),
    )
    ctx.state.set(
        "log_to_file",
        st.sidebar.checkbox("Log to rotating file", value=bool(ctx.state.get("log_to_file"))),
    )
    configure_logging(
        debug=bool(ctx.state.get("debug_logging")),
        log_to_file=bool(ctx.state.get("log_to_file")),
    )

    if st.sidebar.button("Reset all (state + cache)", use_container_width=True):
        LOGGER.info("User requested reset-all.")
        st.cache_data.clear()
        st.cache_resource.clear()
        ctx.state.clear_all()
        from ..state import init_session_state

        init_session_state(ctx.state.session)
        st.rerun()
    if st.sidebar.button("Clear cache only", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared.")

    st.sidebar.divider()
    st.sidebar.caption("Warning thresholds")
    ctx.state.set(
        "warn_target_corr_threshold",
        st.sidebar.slider(
            "Target corr threshold",
            min_value=0.80,
            max_value=0.99,
            value=float(ctx.state.get("warn_target_corr_threshold")),
            step=0.01,
        ),
    )
    ctx.state.set(
        "warn_feature_corr_threshold",
        st.sidebar.slider(
            "Feature corr threshold",
            min_value=0.80,
            max_value=0.99,
            value=float(ctx.state.get("warn_feature_corr_threshold")),
            step=0.01,
        ),
    )
    ctx.state.set(
        "warn_missing_threshold",
        st.sidebar.slider(
            "Missing threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(ctx.state.get("warn_missing_threshold")),
            step=0.05,
        ),
    )
    ctx.state.set(
        "warn_high_cardinality_threshold",
        st.sidebar.slider(
            "High-cardinality unique count",
            min_value=20,
            max_value=500,
            value=int(ctx.state.get("warn_high_cardinality_threshold")),
            step=10,
        ),
    )

    st.sidebar.divider()
    st.sidebar.caption("Project State (JSON/YAML)")
    fmt = st.sidebar.selectbox("Export format", options=["json", "yaml"], index=0)
    try:
        blob = export_project_state(ctx.state.session, fmt=fmt)
        st.sidebar.download_button(
            f"Download State ({fmt.upper()})",
            data=blob,
            file_name=f"automl_state.{fmt}",
            mime="application/json" if fmt == "json" else "application/x-yaml",
            use_container_width=True,
        )
    except Exception as exc:
        st.sidebar.warning(f"State export unavailable: {exc}")

    state_file = st.sidebar.file_uploader("Load state", type=["json", "yaml", "yml"])
    if st.sidebar.button("Apply State", use_container_width=True):
        if state_file is None:
            st.sidebar.error("Upload state config first.")
        else:
            try:
                settings = import_project_state(state_file.getvalue(), file_name=state_file.name)
                for key in PROJECT_KEYS:
                    if key in settings:
                        ctx.state.set(key, settings[key])
                ctx.state.clear_training_artifacts()
                st.sidebar.success("State loaded. Re-upload dataset if needed.")
            except Exception as exc:
                LOGGER.warning("State import error: %s", exc, exc_info=True)
                st.sidebar.error(f"Could not import state: {exc}")

    has_data = ctx.state.get("df") is not None
    has_features = bool(ctx.state.get("selected_features"))
    has_target = ctx.state.get("task_type") == "Clustering" or (
        ctx.state.get("target_col") not in {None, "<None>"}
    )
    has_results = bool(ctx.state.get("training_results"))
    st.sidebar.divider()
    st.sidebar.caption("Status")
    st.sidebar.write(f"1. Upload: {'Done' if has_data else 'Pending'}")
    st.sidebar.write(f"2. Feature Selection: {'Done' if has_features else 'Pending'}")
    st.sidebar.write(f"3. Target Ready: {'Done' if has_target else 'Pending'}")
    st.sidebar.write(f"4. Training: {'Done' if has_results else 'Pending'}")
