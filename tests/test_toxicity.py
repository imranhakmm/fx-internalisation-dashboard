from __future__ import annotations

from math import isnan

import polars as pl
import pytest

from src.toxicity import (
    compute_tag_horizon_curve,
    compute_tag_pair_toxicity,
    get_markout_distribution,
    per_tag_horizon_table,
)


def test_compute_tag_horizon_curve_aggregates_known_means_and_single_trade_ci() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["TagA", "TagA", "TagB"],
            "pair": ["EURUSD", "EURUSD", "USDJPY"],
            "notional_usd": [1_000_000.0, 2_000_000.0, 500_000.0],
            "markout_bp_1s": [1.0, 3.0, 5.0],
            "markout_bp_5s": [2.0, 4.0, 7.0],
        }
    )

    curve = compute_tag_horizon_curve(markouts, horizons_seconds=(1.0, 5.0))

    assert curve.select(["tag", "horizon_s", "mean_bp", "trade_count"]).to_dicts() == [
        {"tag": "TagA", "horizon_s": 1.0, "mean_bp": 2.0, "trade_count": 2},
        {"tag": "TagA", "horizon_s": 5.0, "mean_bp": 3.0, "trade_count": 2},
        {"tag": "TagB", "horizon_s": 1.0, "mean_bp": 5.0, "trade_count": 1},
        {"tag": "TagB", "horizon_s": 5.0, "mean_bp": 7.0, "trade_count": 1},
    ]

    tag_b_rows = curve.filter(pl.col("tag") == "TagB").sort("horizon_s").to_dicts()
    assert isnan(tag_b_rows[0]["ci_low_bp"])
    assert isnan(tag_b_rows[0]["ci_high_bp"])
    assert isnan(tag_b_rows[1]["ci_low_bp"])
    assert isnan(tag_b_rows[1]["ci_high_bp"])


def test_compute_tag_horizon_curve_ci_matches_two_trade_example() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["TagA", "TagA"],
            "pair": ["EURUSD", "EURUSD"],
            "notional_usd": [1_000_000.0, 1_000_000.0],
            "markout_bp_30s": [1.0, -1.0],
        }
    )

    curve_row = compute_tag_horizon_curve(markouts, horizons_seconds=(30.0,)).row(0, named=True)

    assert curve_row["mean_bp"] == pytest.approx(0.0, abs=1e-12)
    assert curve_row["ci_low_bp"] == pytest.approx(-1.96, abs=1e-6)
    assert curve_row["ci_high_bp"] == pytest.approx(1.96, abs=1e-6)


def test_compute_tag_pair_toxicity_flips_sign_of_mean_markout() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["TagA", "TagA"],
            "pair": ["EURUSD", "EURUSD"],
            "notional_usd": [1_000_000.0, 1_000_000.0],
            "markout_bp_30s": [-0.4, -0.6],
        }
    )

    toxicity = compute_tag_pair_toxicity(markouts, horizon_seconds=30.0)

    assert toxicity.to_dicts() == [
        {"tag": "TagA", "pair": "EURUSD", "trade_count": 2, "toxicity_score": 0.5}
    ]


def test_null_markouts_are_excluded_only_from_affected_horizon_counts() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["TagA", "TagA", "TagA"],
            "pair": ["EURUSD", "EURUSD", "EURUSD"],
            "notional_usd": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "markout_bp_1s": [0.1, -0.1, 0.0],
            "markout_bp_300s": [0.5, None, -0.5],
        }
    )

    curve = compute_tag_horizon_curve(markouts, horizons_seconds=(1.0, 300.0))
    counts = {
        row["horizon_s"]: row["trade_count"]
        for row in curve.select(["horizon_s", "trade_count"]).to_dicts()
    }

    assert counts == {1.0: 3, 300.0: 2}


def test_distribution_and_wide_table_helpers_return_expected_shapes() -> None:
    markouts = pl.DataFrame(
        {
            "tag": ["TagA", "TagA", "TagB"],
            "pair": ["EURUSD", "GBPUSD", "USDJPY"],
            "notional_usd": [2_000_000.0, 1_000_000.0, 3_000_000.0],
            "markout_bp_1s": [-0.2, 0.2, 0.0],
            "markout_bp_5s": [-0.1, 0.1, 0.0],
            "markout_bp_30s": [-0.4, -0.2, 0.1],
            "markout_bp_60s": [-0.6, -0.3, 0.2],
            "markout_bp_300s": [-0.8, -0.4, 0.3],
        }
    )

    distribution = get_markout_distribution(markouts, horizon_seconds=60.0, tags=["TagA"])
    wide_table = per_tag_horizon_table(markouts)

    assert distribution.to_dicts() == [
        {"tag": "TagA", "markout_bp": -0.6},
        {"tag": "TagA", "markout_bp": -0.3},
    ]
    assert wide_table.select(["tag", "trade_count", "notional_usd_mm", "reporting_score_30s", "routing_score_60s"]).to_dicts() == [
        {
            "tag": "TagA",
            "trade_count": 2,
            "notional_usd_mm": 3.0,
            "reporting_score_30s": 0.30000000000000004,
            "routing_score_60s": 0.44999999999999996,
        },
        {
            "tag": "TagB",
            "trade_count": 1,
            "notional_usd_mm": 3.0,
            "reporting_score_30s": -0.1,
            "routing_score_60s": -0.2,
        },
    ]
