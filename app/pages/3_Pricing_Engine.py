from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.charts import build_pnl_delta_by_tag
from app.data import config_hash, load_dashboard_data, load_projected_snapshot
from app.formatting import fmt_pct, fmt_usd_per_million, fmt_usd_signed
from src.pricing import DEFAULT_CONFIG, PricingConfig, config_to_payload, get_active_config

PAIR_ORDER = ["EURUSD", "USDJPY", "GBPUSD"]
TAG_ORDER = ["HFT_A", "HFT_B", "PB_C", "RetailAgg_D", "Corp_E", "Bank_F"]


def _route_badge(route: str) -> str:
    if route == "hedge":
        return "background-color: #DDE7F0; color: #1F3A56; font-weight: 600"
    if route == "internalise":
        return "background-color: #DCEAD9; color: #2A4A2A; font-weight: 600"
    return "background-color: #ECEFF3; color: #44505C; font-weight: 600"


def _delta_color(value: float) -> str:
    if value > 0:
        return "color: #2A7B4F; font-weight: 600"
    if value < 0:
        return "color: #B74D4D; font-weight: 600"
    return "color: #6B7280; font-weight: 600"


def _collapse_routes_by_tag(trades_with_pnl: pl.DataFrame) -> pl.DataFrame:
    return (
        trades_with_pnl.group_by("tag")
        .agg(
            [
                pl.len().alias("trade_count"),
                (pl.sum("notional_usd") / 1_000_000.0).alias("notional_usd_mm"),
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
        .select(["tag", "route", "trade_count", "notional_usd_mm"])
    )


def _build_pending_config(current_config: PricingConfig) -> PricingConfig:
    st.subheader("Per-pair parameters")
    pair_columns = st.columns(3, gap="large")
    half_spread_bp: dict[str, float] = {}
    hedge_cost_bp: dict[str, float] = {}

    for pair, column in zip(PAIR_ORDER, pair_columns):
        column.markdown(f"**`{pair}`**")
        half_spread_bp[pair] = float(
            column.slider(
                "half_spread_bp",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(current_config.half_spread_bp[pair]),
                key=f"pe_{pair.lower()}_hs",
            )
        )
        hedge_cost_bp[pair] = float(
            column.slider(
                "hedge_cost_bp",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                value=float(current_config.hedge_cost_bp[pair]),
                key=f"pe_{pair.lower()}_hc",
            )
        )

    st.subheader("Routing thresholds")
    threshold_columns = st.columns(3, gap="large")
    size_threshold_mm = float(
        threshold_columns[0].slider(
            "internalise_size_threshold_usd",
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            value=float(current_config.internalise_size_threshold_usd) / 1_000_000.0,
            format="$%.1fM",
            key="pe_size_threshold",
        )
    )
    toxicity_threshold = float(
        threshold_columns[1].slider(
            "internalise_toxicity_threshold",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=float(current_config.internalise_toxicity_threshold),
            key="pe_toxicity_threshold",
        )
    )
    inventory_decay_seconds = float(
        threshold_columns[2].slider(
            "inventory_decay_seconds",
            min_value=5,
            max_value=300,
            step=5,
            value=int(current_config.inventory_decay_seconds),
            key="pe_inventory_decay",
        )
    )

    tag_widens: dict[str, float] = {}
    tag_routes: dict[str, str] = {}
    with st.expander("Per-tag overrides — widen and route", expanded=False):
        for tag in TAG_ORDER:
            row = st.columns([2, 3, 3], gap="medium")
            row[0].markdown(f"**{tag}**")
            tag_widens[tag] = float(
                row[1].slider(
                    f"{tag} widen_bp",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(current_config.tag_widen_bp.get(tag, 0.0)),
                    key=f"pe_tag_widen_{tag}",
                    label_visibility="collapsed",
                )
            )
            route_options = ["auto", "internalise", "hedge", "reject"]
            current_route = current_config.tag_route_override.get(tag, "auto")
            tag_routes[tag] = row[2].selectbox(
                f"{tag} route override",
                options=route_options,
                index=route_options.index(current_route),
                key=f"pe_tag_route_{tag}",
                label_visibility="collapsed",
            )

    return PricingConfig(
        half_spread_bp=half_spread_bp,
        hedge_cost_bp=hedge_cost_bp,
        internalise_size_threshold_usd=size_threshold_mm * 1_000_000.0,
        internalise_toxicity_threshold=toxicity_threshold,
        inventory_decay_seconds=inventory_decay_seconds,
        tag_widen_bp={tag: widen for tag, widen in tag_widens.items() if widen > 0.0},
        tag_route_override={tag: route for tag, route in tag_routes.items() if route != "auto"},
    )


def _build_delta_frame(current_per_tag: pl.DataFrame, projected_per_tag: pl.DataFrame) -> pl.DataFrame:
    return (
        current_per_tag.select(["tag", pl.col("pnl_usd").alias("current_pnl_usd")])
        .join(
            projected_per_tag.select(["tag", pl.col("pnl_usd").alias("projected_pnl_usd")]),
            on="tag",
            how="outer",
            coalesce=True,
        )
        .with_columns(
            [
                pl.col("current_pnl_usd").fill_null(0.0),
                pl.col("projected_pnl_usd").fill_null(0.0),
            ]
        )
        .with_columns((pl.col("projected_pnl_usd") - pl.col("current_pnl_usd")).alias("pnl_delta_usd"))
        .select(["tag", "current_pnl_usd", "projected_pnl_usd", "pnl_delta_usd"])
    )


def _build_route_flip_table(
    current_trades_with_pnl: pl.DataFrame,
    projected_trades_with_pnl: pl.DataFrame,
    pnl_delta_by_tag: pl.DataFrame,
) -> pl.DataFrame:
    current_routes = _collapse_routes_by_tag(current_trades_with_pnl).rename({"route": "current_route"})
    projected_routes = _collapse_routes_by_tag(projected_trades_with_pnl).rename({"route": "pending_route"})

    return (
        current_routes.join(projected_routes, on="tag", how="outer", coalesce=True)
        .join(pnl_delta_by_tag.select(["tag", "pnl_delta_usd"]), on="tag", how="left")
        .with_columns(
            [
                pl.coalesce([pl.col("trade_count"), pl.lit(0)]).alias("trade_count"),
                pl.coalesce([pl.col("notional_usd_mm"), pl.lit(0.0)]).alias("notional_usd_mm"),
                pl.col("pnl_delta_usd").fill_null(0.0),
            ]
        )
        .filter(pl.col("current_route") != pl.col("pending_route"))
        .sort(pl.col("pnl_delta_usd").abs(), descending=True)
        .select(
            [
                pl.col("tag").alias("Tag"),
                pl.col("current_route").alias("Current route"),
                pl.col("pending_route").alias("Pending route"),
                pl.col("trade_count").alias("Trades"),
                pl.col("notional_usd_mm").alias("Notional ($mm)"),
                pl.col("pnl_delta_usd").alias("PnL delta ($)"),
            ]
        )
    )


def _style_route_flips(route_flips: pl.DataFrame) -> pd.io.formats.style.Styler:
    route_pd = route_flips.to_pandas()
    return (
        route_pd.style.format(
            {
                "Trades": "{:,.0f}",
                "Notional ($mm)": "${:,.1f}",
                "PnL delta ($)": "${:+,.0f}",
            }
        )
        .map(_route_badge, subset=["Current route", "Pending route"])
        .map(_delta_color, subset=["PnL delta ($)"])
        .set_properties(subset=["Tag"], **{"font-weight": "600"})
        .hide(axis="index")
    )


def _internalisation_ratio(trades_with_pnl: pl.DataFrame) -> float:
    total_notional = float(trades_with_pnl["notional_usd"].sum() or 0.0)
    if total_notional == 0.0:
        return 0.0
    internalised_notional = float(
        trades_with_pnl.filter(pl.col("route") == "internalise")["notional_usd"].sum() or 0.0
    )
    return 100.0 * internalised_notional / total_notional


def render_page() -> None:
    st.set_page_config(page_title="Pricing Engine", layout="wide")
    st.title("Pricing Engine")
    st.caption("Adjust pricing and routing parameters; see projected PnL impact in real time.")

    current_config = get_active_config()
    current_data = load_dashboard_data(config_hash(current_config))
    pending_config = _build_pending_config(current_config)

    current_hash = config_hash(current_config)
    pending_hash = config_hash(pending_config)

    preview_start = perf_counter()
    if pending_hash == current_hash:
        projected_data = {
            "trades_with_pnl": current_data["trades_with_pnl"],
            "per_pair": current_data["per_pair"],
            "per_tag": current_data["per_tag"],
            "attribution_totals": current_data["attribution_totals"],
            "routing_scores": current_data["routing_scores"],
        }
    else:
        projected_data = load_projected_snapshot(config_to_payload(pending_config))
    preview_ms = (perf_counter() - preview_start) * 1000.0

    current_trades = current_data["trades_with_pnl"]
    projected_trades = projected_data["trades_with_pnl"]
    current_totals = current_data["attribution_totals"]
    projected_totals = projected_data["attribution_totals"]

    total_notional = float(projected_trades["notional_usd"].sum() or 0.0)
    projected_pnl_per_mm = projected_totals["net_pnl_usd"] / (total_notional / 1_000_000.0)
    current_pnl_per_mm = current_totals["net_pnl_usd"] / (total_notional / 1_000_000.0)
    projected_internalisation = _internalisation_ratio(projected_trades)
    current_internalisation = _internalisation_ratio(current_trades)

    st.subheader("Projected impact vs current config")
    metric_row_1 = st.columns(3, gap="small")
    metric_row_2 = st.columns(3, gap="small")
    metric_row_1[0].metric(
        "Total PnL",
        fmt_usd_signed(projected_totals["net_pnl_usd"]),
        delta=fmt_usd_signed(projected_totals["net_pnl_usd"] - current_totals["net_pnl_usd"]),
        delta_color="normal",
    )
    metric_row_1[1].metric(
        "PnL / mm",
        fmt_usd_per_million(projected_pnl_per_mm),
        delta=fmt_usd_per_million(projected_pnl_per_mm - current_pnl_per_mm),
        delta_color="normal",
    )
    metric_row_1[2].metric(
        "Internalisation ratio",
        fmt_pct(projected_internalisation),
        delta=f"{projected_internalisation - current_internalisation:+.1f} pp",
        delta_color="off",
    )
    metric_row_2[0].metric(
        "Spread capture",
        f"${projected_totals['spread_capture_usd']:,.0f}",
        delta=fmt_usd_signed(projected_totals["spread_capture_usd"] - current_totals["spread_capture_usd"]),
        delta_color="normal",
    )
    metric_row_2[1].metric(
        "AS drag",
        f"${projected_totals['adverse_selection_usd']:,.0f}",
        delta=fmt_usd_signed(projected_totals["adverse_selection_usd"] - current_totals["adverse_selection_usd"]),
        delta_color="inverse",
    )
    metric_row_2[2].metric(
        "Hedge cost",
        f"${projected_totals['hedge_cost_usd']:,.0f}",
        delta=fmt_usd_signed(projected_totals["hedge_cost_usd"] - current_totals["hedge_cost_usd"]),
        delta_color="inverse",
    )

    pnl_delta_by_tag = _build_delta_frame(current_data["per_tag"], projected_data["per_tag"])
    route_flips = _build_route_flip_table(current_trades, projected_trades, pnl_delta_by_tag)

    chart_columns = st.columns([3, 2], gap="large")
    chart_columns[0].plotly_chart(build_pnl_delta_by_tag(pnl_delta_by_tag), width="stretch")
    chart_columns[1].subheader("Route flips")
    if route_flips.is_empty():
        chart_columns[1].caption("No route changes from current config.")
    else:
        chart_columns[1].dataframe(
            _style_route_flips(route_flips),
            width="stretch",
            hide_index=True,
            height=280,
        )

    st.divider()
    action_columns = st.columns([1, 1, 6], gap="small")
    if action_columns[0].button("Apply to dashboard", type="primary"):
        st.session_state["pricing_config"] = pending_config
        st.toast("✓ Config applied — Home and PnL Attribution pages updated", icon="✅")
    if action_columns[1].button("Reset to defaults"):
        st.session_state.pop("pricing_config", None)
        for key in list(st.session_state.keys()):
            if key.startswith("pe_"):
                del st.session_state[key]
        st.rerun()

    st.caption(
        "Current config = sliders' last Apply; pending config = current slider state. "
        "KPI tiles show projected vs current. Apply propagates the pending config to "
        "all other pages of the dashboard."
    )


if __name__ == "__main__":
    render_page()
