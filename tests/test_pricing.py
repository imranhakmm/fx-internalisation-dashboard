from __future__ import annotations

from dataclasses import replace

import polars as pl

from src.pricing import (
    DEFAULT_CONFIG,
    assign_routes,
    compute_routing_toxicity_scores,
    compute_tag_toxicity_scores,
    reprice_trades,
)


def test_assign_routes_respects_size_and_toxicity_thresholds() -> None:
    trades = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "pair": ["EURUSD", "EURUSD", "EURUSD"],
            "tag": ["Bank_F", "HFT_A", "RetailAgg_D"],
            "side": ["BUY", "BUY", "SELL"],
            "notional_usd": [6_000_000.0, 1_000_000.0, 100_000.0],
            "mid_at_trade": [1.0, 1.0, 1.0],
            "executed_price": [1.0, 1.0, 1.0],
        }
    )
    routing_toxicity_scores = {"Bank_F": 0.1, "HFT_A": 0.8, "RetailAgg_D": 0.02}

    routed = assign_routes(trades, DEFAULT_CONFIG, routing_toxicity_scores)

    assert routed["route"].to_list() == ["hedge", "hedge", "internalise"]


def test_assign_routes_tag_override_beats_size_rule() -> None:
    trades = pl.DataFrame(
        {
            "ts": [1],
            "pair": ["USDJPY"],
            "tag": ["Bank_F"],
            "side": ["BUY"],
            "notional_usd": [6_000_000.0],
            "mid_at_trade": [150.0],
            "executed_price": [150.1],
        }
    )
    config = replace(DEFAULT_CONFIG, tag_route_override={"Bank_F": "internalise"})

    routed = assign_routes(trades, config, {"Bank_F": 0.9})

    assert routed["route"].to_list() == ["internalise"]


def test_compute_tag_toxicity_scores_returns_negative_mean_markout() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["HFT_A", "HFT_A", "RetailAgg_D", "RetailAgg_D"],
            "markout_bp_30s": [-0.6, -0.4, 0.1, -0.1],
        }
    )

    scores = compute_tag_toxicity_scores(markouts, horizon_seconds=30.0)

    assert scores == {"HFT_A": 0.5, "RetailAgg_D": -0.0}


def test_routing_uses_hold_horizon() -> None:
    trades = pl.DataFrame(
        {
            "ts": [1],
            "pair": ["EURUSD"],
            "tag": ["Bank_F"],
            "side": ["BUY"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [1.0],
            "executed_price": [1.0],
        }
    )
    markouts = pl.DataFrame(
        {
            "tag": ["Bank_F", "Bank_F"],
            "markout_bp_30s": [-0.1, -0.1],
            "markout_bp_60s": [-0.5, -0.5],
        }
    )

    config_60 = replace(DEFAULT_CONFIG, inventory_decay_seconds=60.0)
    config_30 = replace(DEFAULT_CONFIG, inventory_decay_seconds=30.0)

    routed_60 = assign_routes(
        trades,
        config_60,
        compute_routing_toxicity_scores(markouts, config_60),
    )
    routed_30 = assign_routes(
        trades,
        config_30,
        compute_routing_toxicity_scores(markouts, config_30),
    )

    assert routed_60["route"].to_list() == ["hedge"]
    assert routed_30["route"].to_list() == ["internalise"]


def test_reprice_trades_uses_pair_spread_and_tag_widen() -> None:
    trades = pl.DataFrame(
        {
            "ts": [1, 2],
            "pair": ["EURUSD", "USDJPY"],
            "tag": ["HFT_B", "RetailAgg_D"],
            "side": ["BUY", "SELL"],
            "notional_usd": [100_000.0, 200_000.0],
            "mid_at_trade": [1.1000, 150.00],
            "executed_price": [1.100022, 149.99505],
        }
    )
    config = replace(
        DEFAULT_CONFIG,
        half_spread_bp={"EURUSD": 0.20, "USDJPY": 0.30, "GBPUSD": 0.40},
        tag_widen_bp={"HFT_B": 0.10},
    )

    repriced = reprice_trades(trades, config)

    assert repriced["executed_price"].to_list() == [1.100033, 149.99505]
