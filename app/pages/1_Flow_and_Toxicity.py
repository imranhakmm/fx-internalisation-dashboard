from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.charts import build_horizon_curve, build_markout_box, build_tag_pair_heatmap
from app.data import config_hash, load_dashboard_data
from src.pricing import get_active_config
from src.toxicity import (
    compute_tag_horizon_curve,
    compute_tag_pair_toxicity,
    get_markout_distribution,
    per_tag_horizon_table,
)

HORIZON_OPTIONS = {
    "1s": 1.0,
    "5s": 5.0,
    "30s": 30.0,
    "60s": 60.0,
    "300s": 300.0,
}


def style_detail_table(detail_table: pl.DataFrame) -> pd.io.formats.style.Styler:
    renamed = detail_table.rename(
        {
            "tag": "Tag",
            "trade_count": "Trades",
            "notional_usd_mm": "Notional ($mm)",
            "mean_bp_1s": "Mean @1s",
            "mean_bp_5s": "Mean @5s",
            "mean_bp_30s": "Mean @30s",
            "mean_bp_60s": "Mean @60s",
            "mean_bp_300s": "Mean @300s",
            "reporting_score_30s": "Reporting score (30s)",
            "routing_score_60s": "Routing score (60s)",
        }
    ).to_pandas()

    markout_columns = ["Mean @1s", "Mean @5s", "Mean @30s", "Mean @60s", "Mean @300s"]
    color_map = -renamed[markout_columns]

    return (
        renamed.style.background_gradient(
            subset=markout_columns,
            cmap="Reds",
            vmin=-0.1,
            vmax=1.0,
            gmap=color_map,
            axis=None,
        )
        .format(
            {
                "Trades": "{:,.0f}",
                "Notional ($mm)": "${:,.1f} mm",
                "Mean @1s": "{:+.2f} bp",
                "Mean @5s": "{:+.2f} bp",
                "Mean @30s": "{:+.2f} bp",
                "Mean @60s": "{:+.2f} bp",
                "Mean @300s": "{:+.2f} bp",
                "Reporting score (30s)": "{:+.2f}",
                "Routing score (60s)": "{:+.2f}",
            }
        )
        .hide(axis="index")
    )


def render_page() -> None:
    st.set_page_config(page_title="Flow & Toxicity", layout="wide")
    st.title("Flow & Toxicity")
    st.caption("Per-tag markout profiles, toxicity by tag and pair, and distribution analysis.")

    config = get_active_config()
    data = load_dashboard_data(config_hash(config))
    markouts = data["markouts"]
    available_tags = markouts["tag"].unique().sort().to_list()

    control_columns = st.columns([2, 1], gap="large")
    selected_tags = control_columns[0].multiselect("Tags", options=available_tags, default=available_tags)
    selected_horizon_label = control_columns[1].selectbox(
        "Horizon",
        options=list(HORIZON_OPTIONS.keys()),
        index=3,
    )

    if not selected_tags:
        st.info("Select at least one tag to view toxicity analytics.")
        st.stop()

    selected_horizon = HORIZON_OPTIONS[selected_horizon_label]
    filtered_markouts = markouts.filter(pl.col("tag").is_in(selected_tags))

    horizon_curve = compute_tag_horizon_curve(filtered_markouts, tags=selected_tags)
    distribution = get_markout_distribution(filtered_markouts, horizon_seconds=selected_horizon, tags=selected_tags)
    tag_pair_toxicity = compute_tag_pair_toxicity(filtered_markouts, horizon_seconds=selected_horizon)
    detail_table = per_tag_horizon_table(filtered_markouts)

    st.plotly_chart(build_horizon_curve(horizon_curve, selected_tags), use_container_width=True)

    chart_columns = st.columns(2, gap="large")
    chart_columns[0].plotly_chart(
        build_markout_box(distribution, selected_horizon, selected_tags),
        use_container_width=True,
    )
    chart_columns[1].plotly_chart(
        build_tag_pair_heatmap(tag_pair_toxicity, selected_horizon),
        use_container_width=True,
    )

    st.subheader("Per-tag markout detail")
    st.dataframe(style_detail_table(detail_table), use_container_width=True, hide_index=True, height=300)

    st.caption(
        "Toxicity score = −mean(markout_bp) at horizon. Higher = more toxic. "
        "Reporting score (30s) is the standard reference; routing score (60s) is the "
        "horizon over which AS is realised on internalised flow."
    )


if __name__ == "__main__":
    render_page()
