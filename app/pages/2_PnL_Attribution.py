from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.charts import (
    build_pnl_bars_by_category,
    build_pnl_by_hour,
    build_pnl_composition_stacked,
    build_pnl_waterfall,
)
from app.data import config_hash, load_dashboard_data
from app._styles import apply_global_styles
from src.pricing import get_active_config


def build_detail_table(per_tag: pl.DataFrame) -> pl.DataFrame:
    return (
        per_tag.with_columns(
            [
                (pl.col("notional_usd") / 1_000_000.0).alias("notional_usd_mm"),
                (pl.col("pnl_usd") / (pl.col("notional_usd") / 1_000_000.0)).alias("pnl_per_mm"),
            ]
        )
        .sort("pnl_usd", descending=True)
        .select(
            [
                pl.col("tag").alias("Tag"),
                pl.col("trade_count").alias("Trades"),
                pl.col("notional_usd_mm").alias("Notional ($mm)"),
                pl.col("spread_capture_usd").alias("Spread capture"),
                pl.col("as_usd").alias("AS drag"),
                pl.col("hedge_cost_usd").alias("Hedge cost"),
                pl.col("pnl_usd").alias("Net PnL"),
                pl.col("pnl_per_mm").alias("PnL / mm"),
            ]
        )
    )


def style_detail_table(detail_table: pl.DataFrame) -> pd.io.formats.style.Styler:
    detail_pd = detail_table.to_pandas()

    def row_background(row: pd.Series) -> list[str]:
        background = "background-color: rgba(123, 182, 143, 0.15)" if row["Net PnL"] >= 0 else "background-color: rgba(216, 138, 138, 0.15)"
        return [background] * len(row)

    return (
        detail_pd.style.apply(row_background, axis=1)
        .format(
            {
                "Trades": "{:,.0f}",
                "Notional ($mm)": "${:,.1f} mm",
                "Spread capture": "${:+,.0f}",
                "AS drag": "${:+,.0f}",
                "Hedge cost": "${:+,.0f}",
                "Net PnL": "${:+,.0f}",
                "PnL / mm": "${:+,.2f}",
            }
        )
        .hide(axis="index")
    )


def render_page() -> None:
    st.set_page_config(page_title="PnL Attribution", layout="wide")
    apply_global_styles()
    st.title("PnL Attribution")
    st.caption("Where the book made and lost money — by source, by tag, by pair, by hour.")

    config = get_active_config()
    data = load_dashboard_data(config_hash(config))
    attribution_totals = data["attribution_totals"]
    attribution_by_hour = data["attribution_by_hour"]
    per_tag = data["per_tag"]
    per_pair = data["per_pair"]

    st.plotly_chart(build_pnl_waterfall(attribution_totals), width="stretch")

    bar_columns = st.columns(2, gap="large")
    bar_columns[0].plotly_chart(
        build_pnl_bars_by_category(per_tag, "tag", "PnL by tag"),
        width="stretch",
    )
    bar_columns[1].plotly_chart(
        build_pnl_bars_by_category(per_pair, "pair", "PnL by pair"),
        width="stretch",
    )

    st.plotly_chart(build_pnl_composition_stacked(per_tag), width="stretch")
    st.plotly_chart(build_pnl_by_hour(attribution_by_hour), width="stretch")

    st.subheader("Per-tag PnL detail")
    st.dataframe(
        style_detail_table(build_detail_table(per_tag)),
        width="stretch",
        hide_index=True,
        height=310,
    )

    st.caption(
        "AS drag and hedge cost are deductions from spread capture: AS drag applies only "
        "to internalised flow; hedge cost applies only to hedged flow. A negative-PnL tag "
        "means the chosen route was wrong for that tag at the current config."
    )


if __name__ == "__main__":
    render_page()
