from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen import generate_and_write_dataset
from src.markouts import DEFAULT_HORIZONS_SECONDS, compute_markouts
from src.actions import Recommendation, apply_recommendation_to_config, generate_recommendations
from src.pnl import (
    compute_pnl,
    cumulative_pnl_timeseries,
    pnl_attribution_by_hour,
    pnl_attribution_totals,
    summarise_pnl_by_pair,
    summarise_pnl_by_tag,
)
from src.pricing import (
    PricingConfig,
    assign_routes,
    config_from_payload,
    config_to_payload,
    compute_routing_toxicity_scores,
    compute_tag_toxicity_scores,
    get_active_config,
    reprice_trades,
)


def config_hash(config: PricingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def load_market_data() -> dict[str, pl.DataFrame]:
    trades_path = DATA_DIR / "trades.parquet"
    ticks_path = DATA_DIR / "ticks.parquet"

    if not (trades_path.exists() and ticks_path.exists()):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_and_write_dataset(DATA_DIR)

    try:
        trades = pl.read_parquet(trades_path)
        ticks = pl.read_parquet(ticks_path)
    except Exception:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_and_write_dataset(DATA_DIR)
        trades = pl.read_parquet(trades_path)
        ticks = pl.read_parquet(ticks_path)

    return {
        "trades": trades,
        "ticks": ticks,
    }


@st.cache_data(show_spinner=False)
def load_historical_markouts(horizons_seconds: tuple[float, ...]) -> pl.DataFrame:
    market_data = load_market_data()
    ordered_horizons = tuple(dict.fromkeys(float(horizon) for horizon in horizons_seconds))
    return compute_markouts(
        trades=market_data["trades"],
        ticks=market_data["ticks"],
        horizons_seconds=ordered_horizons,
    )


@st.cache_data(show_spinner=False)
def load_projected_snapshot(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = config_from_payload(config_payload)
    market_data = load_market_data()
    raw_trades = market_data["trades"]
    ticks = market_data["ticks"]

    routing_markouts = load_historical_markouts((float(config.inventory_decay_seconds),))
    routing_scores = compute_routing_toxicity_scores(routing_markouts, config)
    projected_trades = reprice_trades(raw_trades, config)
    projected_routes = assign_routes(projected_trades, config, routing_scores)
    projected_pnl = compute_pnl(projected_routes, ticks, config)
    per_pair = summarise_pnl_by_pair(projected_pnl)
    per_tag = summarise_pnl_by_tag(projected_pnl)
    attribution_totals = pnl_attribution_totals(projected_pnl)
    attribution_by_hour = pnl_attribution_by_hour(projected_pnl)

    return {
        "config": config,
        "routing_scores": routing_scores,
        "trades_with_pnl": projected_pnl,
        "per_pair": per_pair,
        "per_tag": per_tag,
        "attribution_totals": attribution_totals,
        "attribution_by_hour": attribution_by_hour,
    }


@st.cache_data(show_spinner=False)
def load_dashboard_data(config_hash: str) -> dict[str, Any]:
    """
    Loads trades + ticks, computes markouts, scores, routes, and PnL.
    """
    del config_hash
    config = get_active_config()
    market_data = load_market_data()
    raw_trades = market_data["trades"]
    ticks = market_data["ticks"]
    required_horizons = tuple(dict.fromkeys((*DEFAULT_HORIZONS_SECONDS, float(config.inventory_decay_seconds))))
    markouts = load_historical_markouts(required_horizons)
    reporting_scores = compute_tag_toxicity_scores(markouts, horizon_seconds=30.0)
    routing_scores = compute_routing_toxicity_scores(markouts, config)
    projected_data = load_projected_snapshot(config_to_payload(config))
    trades_with_pnl = projected_data["trades_with_pnl"]
    per_pair = projected_data["per_pair"]
    per_tag = projected_data["per_tag"]
    attribution_totals = projected_data["attribution_totals"]
    attribution_by_hour = projected_data["attribution_by_hour"]
    cumulative_ts = cumulative_pnl_timeseries(trades_with_pnl, bucket="1h")

    return {
        "markouts": markouts,
        "raw_trades": raw_trades,
        "ticks": ticks,
        "trades_with_pnl": trades_with_pnl,
        "reporting_scores": reporting_scores,
        "routing_scores": routing_scores,
        "config": config,
        "per_pair": per_pair,
        "per_tag": per_tag,
        "cumulative_ts": cumulative_ts,
        "attribution_totals": attribution_totals,
        "attribution_by_hour": attribution_by_hour,
    }


@st.cache_data(show_spinner=False)
def load_action_recommendations(config_payload: dict[str, Any]) -> list[Recommendation]:
    config = config_from_payload(config_payload)
    market_data = load_market_data()
    required_horizons = tuple(dict.fromkeys((*DEFAULT_HORIZONS_SECONDS, float(config.inventory_decay_seconds))))
    markouts = load_historical_markouts(required_horizons)
    return generate_recommendations(
        trades=market_data["trades"],
        ticks=market_data["ticks"],
        markouts=markouts,
        current_config=config,
    )


@st.cache_data(show_spinner=False)
def load_actions_bundle(config_payload: dict[str, Any]) -> dict[str, Any]:
    config = config_from_payload(config_payload)
    recommendations = load_action_recommendations(config_payload)
    combined_config = config
    for recommendation in recommendations:
        if recommendation.action_type == "KEEP":
            continue
        combined_config = apply_recommendation_to_config(combined_config, recommendation)

    combined_projection = load_projected_snapshot(config_to_payload(combined_config))
    return {
        "recommendations": recommendations,
        "combined_config": combined_config,
        "combined_projection": combined_projection,
    }
