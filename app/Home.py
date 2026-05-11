from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import config_hash, load_dashboard_data
from app.charts import build_cumulative_pnl_chart, build_routing_breakdown_chart
from app.formatting import fmt_count, fmt_pct, fmt_usd_compact, fmt_usd_per_million, fmt_usd_signed
from src.pricing import get_active_config


def build_pair_route_breakdown(trades_with_pnl: pl.DataFrame) -> pl.DataFrame:
    route_table = (
        trades_with_pnl.group_by(["pair", "route"])
        .agg(pl.sum("notional_usd").alias("notional_usd"))
        .pivot(on="route", index="pair", values="notional_usd")
        .fill_null(0.0)
    )
    if "internalise" not in route_table.columns:
        route_table = route_table.with_columns(pl.lit(0.0).alias("internalise"))
    if "hedge" not in route_table.columns:
        route_table = route_table.with_columns(pl.lit(0.0).alias("hedge"))

    return (
        route_table.with_columns(
            [
                (pl.col("internalise") + pl.col("hedge")).alias("total_notional"),
            ]
        )
        .with_columns(
            [
                (100.0 * pl.col("internalise") / pl.col("total_notional")).fill_nan(0.0).alias("internalise_pct"),
                (100.0 * pl.col("hedge") / pl.col("total_notional")).fill_nan(0.0).alias("hedge_pct"),
            ]
        )
        .sort("total_notional", descending=True)
        .select(["pair", "internalise_pct", "hedge_pct"])
    )


def build_tag_route_breakdown(trades_with_pnl: pl.DataFrame) -> pl.DataFrame:
    route_table = (
        trades_with_pnl.group_by(["tag", "route"])
        .agg(pl.sum("notional_usd").alias("notional_usd"))
        .pivot(on="route", index="tag", values="notional_usd")
        .fill_null(0.0)
    )
    if "internalise" not in route_table.columns:
        route_table = route_table.with_columns(pl.lit(0.0).alias("internalise"))
    if "hedge" not in route_table.columns:
        route_table = route_table.with_columns(pl.lit(0.0).alias("hedge"))

    return (
        route_table.with_columns(
            [
                (pl.col("internalise") + pl.col("hedge")).alias("total_notional"),
            ]
        )
        .with_columns(
            [
                (100.0 * pl.col("internalise") / pl.col("total_notional")).fill_nan(0.0).alias("internalise_pct"),
                (100.0 * pl.col("hedge") / pl.col("total_notional")).fill_nan(0.0).alias("hedge_pct"),
            ]
        )
        .sort("total_notional", descending=True)
        .select(["tag", "internalise_pct", "hedge_pct"])
    )


def build_toxicity_table(
    per_tag: pl.DataFrame,
    reporting_scores: dict[str, float],
    routing_scores: dict[str, float],
    trades_with_pnl: pl.DataFrame,
) -> pl.DataFrame:
    route_by_tag = (
        trades_with_pnl.group_by("tag")
        .agg(
            [
                pl.col("route").n_unique().alias("route_count"),
                pl.col("route").first().alias("first_route"),
            ]
        )
        .with_columns(
            pl.when(pl.col("route_count") == 1)
            .then(pl.col("first_route"))
            .otherwise(pl.lit("mixed"))
            .alias("route")
        )
        .select(["tag", "route"])
    )

    toxicity_base = pl.DataFrame(
        {
            "tag": list(reporting_scores.keys()),
            "reporting_score_30s": list(reporting_scores.values()),
            "routing_score_60s": [routing_scores[tag] for tag in reporting_scores],
        }
    )
    return (
        toxicity_base.join(
            per_tag.select(["tag", "notional_usd", "pnl_usd"]).with_columns(
                (pl.col("notional_usd") / 1_000_000.0).alias("notional_usd_mm")
            ),
            on="tag",
            how="left",
        )
        .join(route_by_tag, on="tag", how="left")
        .select(
            [
                "tag",
                "reporting_score_30s",
                "routing_score_60s",
                "route",
                "notional_usd_mm",
                "pnl_usd",
            ]
        )
        .sort("notional_usd_mm", descending=True)
    )


def style_toxicity_table(toxicity_table: pl.DataFrame) -> pd.io.formats.style.Styler:
    def route_badge(route: str) -> str:
        if route == "hedge":
            return "background-color: #DDE7F0; color: #1F3A56; font-weight: 600"
        if route == "internalise":
            return "background-color: #DCEAD9; color: #2A4A2A; font-weight: 600"
        return "background-color: #ECEFF3; color: #44505C; font-weight: 600"

    toxicity_pd = toxicity_table.rename(
        {
            "tag": "Tag",
            "reporting_score_30s": "Reporting score (30s)",
            "routing_score_60s": "Routing score (60s)",
            "route": "Route",
            "notional_usd_mm": "Notional ($mm)",
            "pnl_usd": "PnL ($)",
        }
    ).to_pandas()

    return (
        toxicity_pd.style.background_gradient(
            subset=["Reporting score (30s)", "Routing score (60s)"],
            cmap="Reds",
            vmin=-0.1,
            vmax=1.0,
        )
        .format(
            {
                "Reporting score (30s)": "{:+.2f}",
                "Routing score (60s)": "{:+.2f}",
                "Notional ($mm)": "{:,.1f}",
                "PnL ($)": "${:+,.0f}",
            }
        )
        .map(route_badge, subset=["Route"])
        .hide(axis="index")
    )


def build_per_pair_display(per_pair: pl.DataFrame, trades_with_pnl: pl.DataFrame) -> pl.DataFrame:
    route_breakdown = (
        trades_with_pnl.group_by("pair")
        .agg(
            [
                pl.sum("notional_usd").alias("total_notional"),
                pl.when(pl.col("route") == "internalise")
                .then(pl.col("notional_usd"))
                .otherwise(0.0)
                .sum()
                .alias("internalised_notional"),
            ]
        )
        .with_columns((100.0 * pl.col("internalised_notional") / pl.col("total_notional")).alias("internalisation_pct"))
        .select(["pair", "internalisation_pct"])
    )

    return (
        per_pair.join(route_breakdown, on="pair", how="left")
        .with_columns(
            [
                (pl.col("notional_usd") / 1_000_000_000.0).alias("notional_bn"),
                (pl.col("pnl_usd") / (pl.col("notional_usd") / 1_000_000.0)).alias("pnl_per_mm"),
            ]
        )
        .sort("notional_usd", descending=True)
        .select(
            [
                pl.col("pair").alias("Pair"),
                pl.col("trade_count").alias("# Trades"),
                pl.col("notional_bn").alias("Notional"),
                pl.col("pnl_usd").alias("PnL"),
                pl.col("pnl_per_mm").alias("PnL / mm"),
                pl.col("mean_spread_captured_bp").alias("Mean spread captured"),
                pl.col("internalisation_pct").alias("Internalisation %"),
            ]
        )
    )


def render_dashboard() -> None:
    st.set_page_config(
        page_title="eFX Internalisation Dashboard",
        page_icon="📈",
        layout="wide",
    )

    config = get_active_config()
    data = load_dashboard_data(config_hash(config))
    trades_with_pnl = data["trades_with_pnl"]
    reporting_scores = data["reporting_scores"]
    routing_scores = data["routing_scores"]
    per_pair = data["per_pair"]
    per_tag = data["per_tag"]
    cumulative_ts = data["cumulative_ts"]

    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #F7F9FB;
            border: 1px solid #E6ECF2;
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] p {
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #1F2937;
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    total_notional = float(trades_with_pnl["notional_usd"].sum())
    trade_count = trades_with_pnl.height
    total_pnl = float(trades_with_pnl["pnl_usd"].sum())
    pnl_per_mm = total_pnl / (total_notional / 1_000_000.0)
    internalised_notional = float(
        trades_with_pnl.filter(pl.col("route") == "internalise")["notional_usd"].sum() or 0.0
    )
    hedged_notional = float(trades_with_pnl.filter(pl.col("route") == "hedge")["notional_usd"].sum() or 0.0)
    internalisation_ratio = 100.0 * internalised_notional / total_notional
    hedge_ratio = 100.0 * hedged_notional / total_notional

    per_pair_display = build_per_pair_display(per_pair, trades_with_pnl)
    pair_route_breakdown = build_pair_route_breakdown(trades_with_pnl)
    tag_route_breakdown = build_tag_route_breakdown(trades_with_pnl)
    toxicity_table = build_toxicity_table(per_tag, reporting_scores, routing_scores, trades_with_pnl)

    st.title("eFX Internalisation Dashboard")
    st.caption("KPI overview — synthetic G3 FX flow, default routing config")
    st.markdown(
        "_Pricing engine, flow toxicity, and PnL attribution on a simulated trading day._"
    )
    st.caption(
        "Default config: half_spread {EURUSD: 0.2 / USDJPY: 0.3 / GBPUSD: 0.4} bp · "
        "hedge_cost {0.06 / 0.09 / 0.12} bp · T_int 60s · toxicity threshold 0.30 · "
        "size threshold $5M"
    )

    metric_columns = st.columns(6, gap="small")
    metric_columns[0].metric("Total Notional", fmt_usd_compact(total_notional), delta=None)
    metric_columns[1].metric(r"\# Trades", fmt_count(trade_count), delta=None)
    metric_columns[2].metric("Gross PnL", fmt_usd_signed(total_pnl), delta=None)
    metric_columns[3].metric("PnL / mm", fmt_usd_per_million(pnl_per_mm), delta=None)
    metric_columns[4].metric("Internalisation Ratio", fmt_pct(internalisation_ratio), delta=None)
    metric_columns[5].metric("Hedge Ratio", fmt_pct(hedge_ratio), delta=None)

    st.subheader("Per-pair performance")
    st.dataframe(
        per_pair_display,
        hide_index=True,
        width="stretch",
        height=180,
        column_config={
            "Pair": st.column_config.TextColumn("Pair", width="small"),
            "# Trades": st.column_config.NumberColumn("# Trades", format="%d", width="small"),
            "Notional": st.column_config.NumberColumn("Notional", format="$%.2fbn", width="small"),
            "PnL": st.column_config.NumberColumn("PnL", format="$%+.0f", width="small"),
            "PnL / mm": st.column_config.NumberColumn("PnL / mm", format="$%+.2f", width="small"),
            "Mean spread captured": st.column_config.NumberColumn("Mean spread captured", format="%.2f bp", width="small"),
            "Internalisation %": st.column_config.NumberColumn("Internalisation %", format="%.1f%%", width="small"),
        },
    )

    st.subheader("Cumulative PnL attribution")
    st.plotly_chart(build_cumulative_pnl_chart(cumulative_ts), use_container_width=True)

    st.subheader("Routing breakdown")
    route_cols = st.columns(2)
    route_cols[0].plotly_chart(
        build_routing_breakdown_chart(pair_route_breakdown, "pair", "Notional share by pair"),
        use_container_width=True,
    )
    route_cols[1].plotly_chart(
        build_routing_breakdown_chart(tag_route_breakdown, "tag", "Notional share by tag"),
        use_container_width=True,
    )

    st.subheader("Flow toxicity by tag")
    st.dataframe(style_toxicity_table(toxicity_table), hide_index=True, use_container_width=True, height=260)


def measure_loader_cache() -> tuple[float, float]:
    config = get_active_config()
    active_hash = config_hash(config)
    start = perf_counter()
    load_dashboard_data(active_hash)
    first_call = perf_counter() - start
    start = perf_counter()
    load_dashboard_data(active_hash)
    second_call = perf_counter() - start
    return first_call, second_call


if __name__ == "__main__":
    render_dashboard()
