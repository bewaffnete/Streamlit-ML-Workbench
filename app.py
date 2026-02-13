from __future__ import annotations

import os

import streamlit as st

from automl_gui.config import DataLoadConfig, RuntimeConfig, WarningsConfig
from automl_gui.logging_utils import configure_logging
from automl_gui.services import build_service_container
from automl_gui.ui import render_app


def main() -> None:
    st.set_page_config(page_title="AutoML Workbench", page_icon="🧠", layout="wide")
    configure_logging(debug=False, log_to_file=False)
    runtime_config = RuntimeConfig()

    services = build_service_container(
        data_config=DataLoadConfig(max_upload_mb=int(os.getenv("AUTOML_MAX_UPLOAD_MB", "200"))),
        warning_config=WarningsConfig(),
        runtime_config=runtime_config,
        n_clusters=3,
    )
    render_app(services, runtime_config=runtime_config)


if __name__ == "__main__":
    main()
