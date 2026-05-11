from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from src.actions import MIN_IMPROVEMENT_USD, apply_recommendation_to_config, generate_recommendations
from src.data_gen import generate_synthetic_fx_data
from src.markouts import DEFAULT_HORIZONS_SECONDS, compute_markouts
from src.pnl import compute_pnl
from src.pricing import DEFAULT_CONFIG, PricingConfig, assign_routes, compute_routing_toxicity_scores, reprice_trades


@lru_cache(maxsize=1)
def _load_market_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    project_root = Path(__file__).resolve().parents[1]
    trades_path = project_root / "data" / "trades.parquet"
    ticks_path = project_root / "data" / "ticks.parquet"

    if trades_path.exists() and ticks_path.exists():
        return pl.read_parquet(trades_path), pl.read_parquet(ticks_path)

    ticks, trades = generate_synthetic_fx_data()
    return trades, ticks


def test_generate_recommendations_finds_pb_c_hedge_win() -> None:
    trades, ticks = _load_market_data()
    markouts = compute_markouts(trades, ticks, horizons_seconds=DEFAULT_HORIZONS_SECONDS)

    recommendations = generate_recommendations(trades, ticks, markouts, DEFAULT_CONFIG)
    pb_c = next(rec for rec in recommendations if rec.tag == "PB_C")

    assert pb_c.action_type == "HEDGE"
    assert pb_c.pnl_delta > 250_000.0


def test_apply_recommendation_round_trip_matches_projected_total() -> None:
    trades, ticks = _load_market_data()
    markouts = compute_markouts(trades, ticks, horizons_seconds=DEFAULT_HORIZONS_SECONDS)
    recommendation = generate_recommendations(trades, ticks, markouts, DEFAULT_CONFIG)[0]

    new_config = apply_recommendation_to_config(DEFAULT_CONFIG, recommendation)
    routing_scores = compute_routing_toxicity_scores(markouts, new_config)
    repriced_trades = reprice_trades(trades, new_config)
    routed_trades = assign_routes(repriced_trades, new_config, routing_scores)
    pnl_df = compute_pnl(routed_trades, ticks, new_config)
    realised_total = float(pnl_df["pnl_usd"].sum() or 0.0)

    assert abs(realised_total - recommendation.projected_pnl_total) < 1.0


def test_generate_recommendations_returns_keep_below_minimum_improvement() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    ticks = pl.DataFrame(
        {
            "ts": [base_ts, base_ts + timedelta(seconds=60)],
            "pair": ["EURUSD", "EURUSD"],
            "tick_index": [0, 1],
            "mid": [100.0, 100.0],
        }
    )
    trades = pl.DataFrame(
        {
            "ts": [base_ts] * 4,
            "pair": ["EURUSD"] * 4,
            "tag": ["TestTag"] * 4,
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "notional_usd": [100_000.0] * 4,
            "mid_at_trade": [100.0] * 4,
            "executed_price": [100.0020, 99.9980, 100.0020, 99.9980],
        }
    )
    markouts = compute_markouts(trades, ticks, horizons_seconds=(30.0, 60.0))
    config = replace(
        DEFAULT_CONFIG,
        half_spread_bp={"EURUSD": 0.2},
        hedge_cost_bp={"EURUSD": 0.06},
        inventory_decay_seconds=60.0,
    )

    recommendations = generate_recommendations(trades, ticks, markouts, config)

    assert len(recommendations) == 1
    assert recommendations[0].action_type == "KEEP"
    assert recommendations[0].pnl_delta == 0.0
    assert MIN_IMPROVEMENT_USD == 5_000.0
