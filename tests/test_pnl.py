from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl

from src.pnl import compute_pnl, pnl_attribution_by_hour, pnl_attribution_totals
from src.pricing import DEFAULT_CONFIG, assign_routes, compute_routing_toxicity_scores


def _ticks_for_mid_path(mid_values: list[float], step_seconds: int = 30) -> pl.DataFrame:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    return pl.DataFrame(
        {
            "ts": [base_ts + timedelta(seconds=step_seconds * idx) for idx in range(len(mid_values))],
            "pair": ["EURUSD"] * len(mid_values),
            "tick_index": list(range(len(mid_values))),
            "mid": mid_values,
        }
    )


def test_spread_capture_invariant_matches_absolute_distance() -> None:
    config = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0)
    ticks = _ticks_for_mid_path([100.0, 100.0])
    trades = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 15, 7, 0, 0), datetime(2024, 1, 15, 7, 0, 0)],
            "pair": ["EURUSD", "EURUSD"],
            "tag": ["TagA", "TagB"],
            "side": ["BUY", "SELL"],
            "notional_usd": [1_000_000.0, 1_000_000.0],
            "mid_at_trade": [100.0, 100.0],
            "executed_price": [100.125, 99.875],
            "route": ["internalise", "hedge"],
        }
    )

    pnl_df = compute_pnl(trades, ticks, config)
    expected = [12.5, 12.5]

    assert pnl_df["spread_capture_bp"].to_list() == expected


def test_internalised_pnl_decomposition_identity() -> None:
    config = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0)
    ticks = _ticks_for_mid_path([100.0, 100.25])
    trades = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 15, 7, 0, 0)],
            "pair": ["EURUSD"],
            "tag": ["TagA"],
            "side": ["BUY"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [100.0],
            "executed_price": [100.125],
            "route": ["internalise"],
        }
    )

    pnl_row = compute_pnl(trades, ticks, config).row(0, named=True)

    assert pnl_row["pnl_usd"] == pnl_row["spread_capture_usd"] - pnl_row["adverse_selection_usd"]


def test_hedged_pnl_decomposition_and_zero_as() -> None:
    config = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0, hedge_cost_bp={"EURUSD": 0.4})
    ticks = _ticks_for_mid_path([100.0, 120.0])
    trades = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 15, 7, 0, 0)],
            "pair": ["EURUSD"],
            "tag": ["TagA"],
            "side": ["BUY"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [100.0],
            "executed_price": [100.125],
            "route": ["hedge"],
        }
    )

    pnl_row = compute_pnl(trades, ticks, config).row(0, named=True)

    assert pnl_row["pnl_usd"] == pnl_row["spread_capture_usd"] - pnl_row["hedge_cost_usd"]
    assert pnl_row["adverse_selection_usd"] == 0.0


def test_hedged_trades_have_zero_as_regardless_of_mid_move() -> None:
    config = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0, hedge_cost_bp={"EURUSD": 0.4})
    ticks = _ticks_for_mid_path([100.0, 80.0])
    trades = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 15, 7, 0, 0)],
            "pair": ["EURUSD"],
            "tag": ["TagA"],
            "side": ["SELL"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [100.0],
            "executed_price": [99.875],
            "route": ["hedge"],
        }
    )

    pnl_row = compute_pnl(trades, ticks, config).row(0, named=True)

    assert pnl_row["adverse_selection_usd"] == 0.0


def test_internalised_sign_response_to_mid_move() -> None:
    config = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0)
    base_trade = {
        "ts": [datetime(2024, 1, 15, 7, 0, 0)],
        "pair": ["EURUSD"],
        "tag": ["TagA"],
        "side": ["BUY"],
        "notional_usd": [1_000_000.0],
        "mid_at_trade": [100.0],
        "executed_price": [100.125],
        "route": ["internalise"],
    }

    pnl_against = compute_pnl(pl.DataFrame(base_trade), _ticks_for_mid_path([100.0, 100.40]), config).row(0, named=True)
    pnl_favour = compute_pnl(pl.DataFrame(base_trade), _ticks_for_mid_path([100.0, 99.80]), config).row(0, named=True)

    assert pnl_against["pnl_usd"] < pnl_against["spread_capture_usd"]
    assert pnl_favour["pnl_usd"] > pnl_favour["spread_capture_usd"]


def test_default_config_end_to_end_total_pnl_is_positive() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    ticks = pl.DataFrame(
        {
            "ts": [base_ts, base_ts + timedelta(seconds=60)],
            "pair": ["EURUSD", "EURUSD"],
            "tick_index": [0, 1],
            "mid": [100.0, 99.95],
        }
    ).vstack(
        pl.DataFrame(
            {
                "ts": [base_ts, base_ts + timedelta(seconds=60)],
                "pair": ["USDJPY", "USDJPY"],
                "tick_index": [0, 1],
                "mid": [150.0, 150.30],
            }
        )
    )
    trades = pl.DataFrame(
        {
            "ts": [base_ts, base_ts, base_ts],
            "pair": ["EURUSD", "USDJPY", "EURUSD"],
            "tag": ["RetailAgg_D", "Bank_F", "Corp_E"],
            "side": ["BUY", "BUY", "SELL"],
            "notional_usd": [1_000_000.0, 10_000_000.0, 10_000_000.0],
            "mid_at_trade": [100.0, 150.0, 100.0],
            "executed_price": [100.0020, 150.0045, 99.9960],
        }
    )
    markouts = pl.DataFrame(
        {
            "tag": ["RetailAgg_D", "Bank_F", "Corp_E"],
            "markout_bp_30s": [0.00, -0.10, -0.05],
            "markout_bp_60s": [0.00, -0.50, -0.40],
        }
    )

    routing_toxicity_scores = compute_routing_toxicity_scores(markouts, DEFAULT_CONFIG)
    routed = assign_routes(trades, DEFAULT_CONFIG, routing_toxicity_scores)
    pnl_df = compute_pnl(routed, ticks, DEFAULT_CONFIG)

    assert float(pnl_df["pnl_usd"].sum()) > 0.0


def test_pnl_attribution_totals_match_column_sums() -> None:
    pnl_df = pl.DataFrame(
        {
            "spread_capture_usd": [100.0, 200.0],
            "adverse_selection_usd": [20.0, None],
            "hedge_cost_usd": [10.0, 15.0],
            "pnl_usd": [70.0, 185.0],
        }
    )

    totals = pnl_attribution_totals(pnl_df)

    assert totals == {
        "spread_capture_usd": 300.0,
        "adverse_selection_usd": 20.0,
        "hedge_cost_usd": 25.0,
        "net_pnl_usd": 255.0,
    }


def test_pnl_attribution_by_hour_rolls_up_to_totals() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    pnl_df = pl.DataFrame(
        {
            "ts": [
                base_ts,
                base_ts + timedelta(minutes=30),
                base_ts + timedelta(hours=1),
            ],
            "spread_capture_usd": [100.0, 50.0, 150.0],
            "adverse_selection_usd": [20.0, None, 10.0],
            "hedge_cost_usd": [5.0, 10.0, 15.0],
            "pnl_usd": [75.0, 40.0, 125.0],
        }
    )

    hourly = pnl_attribution_by_hour(pnl_df)
    totals = pnl_attribution_totals(pnl_df)

    assert float(hourly["spread_capture_usd"].sum()) == totals["spread_capture_usd"]
    assert float(hourly["adverse_selection_usd"].sum()) == totals["adverse_selection_usd"]
    assert float(hourly["hedge_cost_usd"].sum()) == totals["hedge_cost_usd"]
    assert float(hourly["net_pnl_usd"].sum()) == totals["net_pnl_usd"]
