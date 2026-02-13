"""Plot builders for Streamlit UI."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def make_numeric_distribution(df: pd.DataFrame, feature: str):
    """Histogram with KDE overlay for numeric feature."""
    series = pd.to_numeric(df[feature], errors="coerce").dropna()
    if series.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            name="Histogram",
            histnorm="probability density",
            marker_color="#3B82F6",
            opacity=0.7,
        )
    )

    if series.nunique() > 1:
        x_axis = np.linspace(series.min(), series.max(), 200)
        kde = gaussian_kde(series)(x_axis)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=kde,
                mode="lines",
                name="KDE",
                line=dict(color="#EF4444", width=2),
            )
        )

    fig.update_layout(
        title=f"Numeric Distribution: {feature}",
        template="plotly_white",
        bargap=0.05,
        legend_orientation="h",
    )
    return fig


def make_categorical_count(df: pd.DataFrame, feature: str, top_n: int = 20):
    """Count plot for a categorical feature."""
    counts = df[feature].astype(str).fillna("NaN").value_counts().head(top_n).reset_index()
    counts.columns = [feature, "count"]
    return px.bar(
        counts,
        x=feature,
        y="count",
        title=f"Categorical Count: {feature} (Top {top_n})",
        template="plotly_white",
    )


def make_missing_values_chart(df: pd.DataFrame, columns: Iterable[str]):
    """Missing-value percentage chart."""
    selected = [c for c in columns if c in df.columns]
    if not selected:
        return None
    missing = (df[selected].isna().mean() * 100).sort_values(ascending=False).reset_index()
    missing.columns = ["feature", "missing_pct"]
    return px.bar(
        missing,
        x="feature",
        y="missing_pct",
        title="Missing Values (%)",
        template="plotly_white",
    )


def make_corr_heatmap(corr_df: pd.DataFrame, title: str):
    """Correlation matrix heatmap."""
    return px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
