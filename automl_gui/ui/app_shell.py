"""Top-level UI composition and tab routing."""

from __future__ import annotations

import streamlit as st

from ..config import RuntimeConfig
from ..core.jobs import get_job_manager
from ..state import AppState, init_session_state
from .config_ui import render_model_config_tab, render_preprocessing_tab
from .context import UIContext
from .eda_ui import render_target_eda_tab, render_warning_tab
from .export_ui import render_export_tab
from .sidebar import render_sidebar
from .training_ui import render_training_tab
from .upload_ui import render_upload_tab


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.0rem; padding-bottom: 1.6rem;}
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 8px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app(services, runtime_config: RuntimeConfig) -> None:
    init_session_state(st.session_state)
    _apply_style()
    context = UIContext(
        services=services,
        state=AppState(st.session_state),
        runtime_config=runtime_config,
        job_manager=get_job_manager(max_workers=runtime_config.background_workers),
    )
    render_sidebar(context)

    st.title("AutoML Workbench")
    # brief app description removed per request
    st.markdown("GitHub: [bewaffnete](https://github.com/bewaffnete)")

    tabs = st.tabs(
        [
            "1. Data Upload",
            "2. Target & EDA",
            "3. Warnings",
            "4. Preprocessing",
            "5. Model Config",
            "6. Train & Evaluate",
            "7. Predict & Export",
        ]
    )

    with tabs[0]:
        render_upload_tab(context)
    with tabs[1]:
        render_target_eda_tab(context)
    with tabs[2]:
        render_warning_tab(context)
    with tabs[3]:
        render_preprocessing_tab(context)
    with tabs[4]:
        render_model_config_tab(context)
    with tabs[5]:
        render_training_tab(context)
    with tabs[6]:
        render_export_tab(context)
