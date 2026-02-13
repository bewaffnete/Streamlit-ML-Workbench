"""Upload and dataset overview tab."""

from __future__ import annotations

import streamlit as st
from streamlit.components.v1 import html as st_html

from ..config import DataLoadConfig
from ..data_utils import (
    DataLoadError,
    UploadValidationError,
    cached_dataset_overview,
    cached_describe,
    cached_head,
    cached_info_text,
    memory_usage_mb,
)
from ..logging_utils import get_logger
from .context import UIContext

LOGGER = get_logger("ui.upload")


@st.cache_data(show_spinner=False, max_entries=4)
def _generate_profile_html(fingerprint: str) -> str:
    from ydata_profiling import ProfileReport
    from ..data_utils import get_registered_dataset

    df = get_registered_dataset(fingerprint)
    report = ProfileReport(df, minimal=True, title="AutoML Dataset Profile")
    return report.to_html()


def render_upload_tab(ctx: UIContext) -> None:
    st.subheader("1) Data Upload")
    uploader = st.file_uploader(
        "Upload CSV / Excel / Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
    )

    if uploader is not None:
        file_sig = f"{uploader.name}:{uploader.size}"
        if ctx.state.get("dataset_name") != file_sig:
            try:
                loaded = ctx.services.data_service.load_dataset(
                    file_name=uploader.name,
                    file_size=uploader.size,
                    content_type=uploader.type,
                    file_bytes=uploader.getvalue(),
                    use_sampling=bool(ctx.state.get("sample_large_data")),
                    sample_rows=int(ctx.state.get("sample_size_rows")),
                )
                ctx.state.update(
                    {
                        "raw_df": loaded.raw_df,
                        "df": loaded.working_df,
                        "sampled_from_raw": loaded.was_sampled,
                        "dataset_name": file_sig,
                        "raw_dataset_fingerprint": loaded.raw_fingerprint,
                        "dataset_fingerprint": loaded.working_fingerprint,
                        "target_col": "<None>",
                        "selected_features": [],
                        "profile_html": None,
                    }
                )
                from ..data_utils import guess_default_features

                ctx.state.set("selected_features", guess_default_features(loaded.working_df, None))
                ctx.state.clear_training_artifacts()
            except UploadValidationError as exc:
                st.error(str(exc))
                return
            except DataLoadError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                LOGGER.exception("Unexpected dataset load error.", exc_info=True)
                st.error(f"Unexpected load error: {exc}")
                return

    raw_df = ctx.state.get("raw_df")
    if raw_df is None:
        st.info("Upload a dataset to continue.")
        return

    if len(raw_df) > 500_000 or memory_usage_mb(raw_df) > 1024:
        st.warning("Large dataset detected. Sampling is recommended.")
        c1, c2 = st.columns(2)
        with c1:
            ctx.state.set(
                "sample_large_data",
                st.checkbox("Use sampled dataset", value=bool(ctx.state.get("sample_large_data"))),
            )
        with c2:
            ctx.state.set(
                "sample_size_rows",
                st.slider(
                    "Sample rows",
                    min_value=10_000,
                    max_value=min(200_000, len(raw_df)),
                    value=min(int(ctx.state.get("sample_size_rows")), len(raw_df)),
                    step=5_000,
                ),
            )

    df = ctx.state.get("df")
    if df is None:
        st.info("No working dataset available.")
        return

    if ctx.state.get("sampled_from_raw"):
        st.caption(f"Using sample: {len(df):,} rows of {len(raw_df):,}.")

    fp = ctx.state.get("dataset_fingerprint")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Memory (MB)", f"{memory_usage_mb(df):.2f}")
    c4.metric("Fingerprint", str(fp)[:12] if fp else "n/a")

    t1, t2, t3 = st.tabs(["Head", "Info", "Describe"])
    with t1:
        st.dataframe(cached_head(str(fp), 10), use_container_width=True)
        st.dataframe(cached_dataset_overview(str(fp)), use_container_width=True, height=280)
    with t2:
        st.code(cached_info_text(str(fp)), language="text")
    with t3:
        st.dataframe(cached_describe(str(fp)), use_container_width=True, height=420)

    if st.button("Generate ydata-profiling report"):
        try:
            with st.status("Profiling...", expanded=True) as status:
                status.write("Computing report (can be slow)...")
                html = _generate_profile_html(str(fp))
                ctx.state.set("profile_html", html)
                status.update(label="Done", state="complete")
        except ModuleNotFoundError:
            st.info("Install ydata-profiling to use this feature.")
        except Exception as exc:
            LOGGER.warning("Profiling failed: %s", exc, exc_info=True)
            st.error(f"Profiling failed: {exc}")

    profile_html = ctx.state.get("profile_html")
    if profile_html:
        st.download_button(
            "Download Profile HTML",
            data=profile_html.encode("utf-8"),
            file_name="dataset_profile.html",
            mime="text/html",
        )
        with st.expander("Profile Preview"):
            st_html(profile_html, height=700, scrolling=True)
