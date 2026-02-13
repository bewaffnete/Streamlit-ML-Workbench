"""Model export/inference tab with trusted-loading guardrails."""

from __future__ import annotations

from io import BytesIO
import os

import joblib
import streamlit as st

from ..config import DataLoadConfig
from ..data_utils import load_uploaded_file
from ..logging_utils import get_logger
from ..serialization import dumps_json
from ..state import model_trust_warning
from .context import UIContext

LOGGER = get_logger("ui.export")


def _select_index(options, value, default=0):
    return options.index(value) if value in options else default


def _render_trusted_model_loader(ctx: UIContext) -> None:
    st.markdown("#### Trusted Model Loading (Optional)")
    trusted = st.checkbox(
        "I trust this model file and understand deserialization risks.",
        value=bool(ctx.state.get("trusted_model_loading")),
    )
    ctx.state.set("trusted_model_loading", trusted)
    model_file = st.file_uploader("Load model (.joblib/.pkl)", type=["joblib", "pkl"], key="trusted_model_uploader")
    if st.button("Load Trusted Model", use_container_width=True):
        if model_file is None:
            st.error("Upload a model file first.")
            return
        if not trusted:
            st.error(model_trust_warning())
            return
        try:
            payload = joblib.load(model_file)
            st.success("Trusted model loaded.")
            ctx.state.set("trusted_loaded_model", payload)
        except Exception as exc:
            LOGGER.warning("Trusted model load failed: %s", exc, exc_info=True)
            st.error(f"Could not load model: {exc}")


def render_export_tab(ctx: UIContext) -> None:
    st.subheader("7) Predict & Export")
    training_results = ctx.state.get("training_results")
    if not training_results:
        st.info("Train at least one model first.")
        _render_trusted_model_loader(ctx)
        return

    names = list(training_results.keys())
    selected = st.selectbox(
        "Model for export/inference",
        options=names,
        index=_select_index(names, ctx.state.get("best_model_name")),
    )
    run = training_results[selected]
    result = run.training_result
    pipeline = result.pipeline

    st.warning(model_trust_warning())
    st.download_button(
        "Download model metadata (JSON)",
        data=dumps_json(run.metadata),
        file_name=f"{selected}_metadata.json",
        mime="application/json",
        use_container_width=True,
    )

    blob = BytesIO()
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_cols": run.feature_cols,
            "task_type": ctx.state.get("task_type"),
            "metadata": run.metadata,
        },
        blob,
    )
    st.download_button(
        "Download model (.joblib, trusted use only)",
        data=blob.getvalue(),
        file_name=f"{selected}_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.download_button(
        "Download eval predictions",
        data=run.predictions_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected}_eval_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Predict on New Data")
    new_file = st.file_uploader(
        "Upload CSV/XLSX/Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
        key="predict_uploader",
    )
    if new_file is not None:
        try:
            new_df = load_uploaded_file(
                file_name=new_file.name,
                file_bytes=new_file.getvalue(),
                config=DataLoadConfig(max_upload_mb=int(os.getenv("AUTOML_MAX_UPLOAD_MB", "200"))),
            )
            missing = [col for col in run.feature_cols if col not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return
            X_new = new_df[run.feature_cols].copy()
            preds = pipeline.predict(X_new)
            out_df = new_df.copy()
            out_df["prediction"] = preds
            if ctx.state.get("task_type") == "Classification" and hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_new)
                model_obj = pipeline.named_steps["model"]
                classes = (
                    model_obj.classes_
                    if hasattr(model_obj, "classes_")
                    else [f"class_{i}" for i in range(probs.shape[1])]
                )
                for idx, cls in enumerate(classes):
                    out_df[f"proba_{cls}"] = probs[:, idx]
            st.dataframe(out_df.head(20), use_container_width=True)
            st.download_button(
                "Download predictions",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected}_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            LOGGER.warning("Prediction failed: %s", exc, exc_info=True)
            st.error(f"Prediction failed: {exc}")

    _render_trusted_model_loader(ctx)
