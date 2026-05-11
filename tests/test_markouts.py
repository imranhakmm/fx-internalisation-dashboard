from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from src.markouts import compute_markouts


def test_compute_markouts_sign_convention() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    ticks = pl.DataFrame(
        {
            "ts": [
                base_ts,
                base_ts + timedelta(seconds=10),
                base_ts + timedelta(seconds=20),
                base_ts + timedelta(seconds=30),
                base_ts + timedelta(seconds=40),
            ],
            "pair": ["EURUSD"] * 5,
            "tick_index": [0, 1, 2, 3, 4],
            "mid": [100.00, 100.05, 100.10, 100.20, 99.80],
        }
    )
    trades = pl.DataFrame(
        {
            "ts": [base_ts, base_ts + timedelta(seconds=10)],
            "pair": ["EURUSD", "EURUSD"],
            "tag": ["TagA", "TagB"],
            "side": ["BUY", "SELL"],
            "notional_usd": [1_000_000.0, 1_000_000.0],
            "mid_at_trade": [100.00, 100.00],
            "executed_price": [100.10, 99.90],
        }
    )

    markouts = compute_markouts(trades=trades, ticks=ticks, horizons_seconds=(30.0,), keep_forward_mids=True)

    assert markouts["mid_at_t_plus_30s"].to_list() == [100.20, 99.80]
    assert markouts["markout_bp_30s"].to_list() == [-10.0, -10.0]


def test_compute_markouts_returns_null_past_session_end() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    ticks = pl.DataFrame(
        {
            "ts": [base_ts, base_ts + timedelta(seconds=10), base_ts + timedelta(seconds=20)],
            "pair": ["EURUSD", "EURUSD", "EURUSD"],
            "tick_index": [0, 1, 2],
            "mid": [100.00, 100.10, 100.20],
        }
    )
    trades = pl.DataFrame(
        {
            "ts": [base_ts + timedelta(seconds=10)],
            "pair": ["EURUSD"],
            "tag": ["TagA"],
            "side": ["BUY"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [100.10],
            "executed_price": [100.20],
        }
    )

    markouts = compute_markouts(trades=trades, ticks=ticks, horizons_seconds=(30.0,), keep_forward_mids=True)

    assert markouts["mid_at_t_plus_30s"].to_list() == [None]
    assert markouts["markout_bp_30s"].to_list() == [None]


def test_compute_markouts_zero_when_mid_unchanged_and_no_spread() -> None:
    base_ts = datetime(2024, 1, 15, 7, 0, 0)
    ticks = pl.DataFrame(
        {
            "ts": [base_ts, base_ts + timedelta(seconds=30)],
            "pair": ["EURUSD", "EURUSD"],
            "tick_index": [0, 1],
            "mid": [100.00, 100.00],
        }
    )
    trades = pl.DataFrame(
        {
            "ts": [base_ts],
            "pair": ["EURUSD"],
            "tag": ["TagA"],
            "side": ["BUY"],
            "notional_usd": [1_000_000.0],
            "mid_at_trade": [100.00],
            "executed_price": [100.00],
        }
    )

    markouts = compute_markouts(trades=trades, ticks=ticks, horizons_seconds=(30.0,))

    assert markouts["markout_bp_30s"].to_list() == [0.0]
