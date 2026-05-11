"""Forward mid lookup and LP-perspective markout calculations."""

from __future__ import annotations

from typing import Sequence

import polars as pl

DEFAULT_HORIZONS_SECONDS: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0, 300.0)
MARKOUT_ROUND_DECIMALS = 10


def _horizon_label(horizon_seconds: float) -> str:
    if float(horizon_seconds).is_integer():
        return str(int(horizon_seconds))
    return str(horizon_seconds).replace(".", "p")


def _duration_from_seconds(horizon_seconds: float) -> pl.Expr:
    horizon_ms = int(round(horizon_seconds * 1000))
    return pl.duration(milliseconds=horizon_ms)


def _side_sign_expr() -> pl.Expr:
    return pl.when(pl.col("side") == "BUY").then(1.0).otherwise(-1.0)


def compute_markouts(
    trades: pl.DataFrame,
    ticks: pl.DataFrame,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
    keep_forward_mids: bool = False,
) -> pl.DataFrame:
    """Extend trades with forward mids and LP markouts at the requested horizons."""
    working = trades.with_row_index("__row_id").sort(["pair", "ts", "__row_id"])
    ticks_sorted = ticks.sort(["pair", "ts"])

    for horizon_seconds in horizons_seconds:
        label = _horizon_label(horizon_seconds)
        lookup_ts_col = f"__lookup_ts_{label}s"
        mid_col = f"mid_at_t_plus_{label}s"
        markout_col = f"markout_bp_{label}s"

        working = working.with_columns(
            (pl.col("ts") + _duration_from_seconds(horizon_seconds)).alias(lookup_ts_col)
        )

        forward_ticks = ticks_sorted.select(
            [
                "pair",
                pl.col("ts").alias(lookup_ts_col),
                pl.col("mid").alias(mid_col),
            ]
        ).sort(["pair", lookup_ts_col])

        working = working.join_asof(
            forward_ticks,
            on=lookup_ts_col,
            by="pair",
            strategy="forward",
            check_sortedness=False,
        ).with_columns(
            (
                1e4
                * _side_sign_expr()
                * (pl.col("executed_price") - pl.col(mid_col))
                / pl.col("mid_at_trade")
            )
            .round(MARKOUT_ROUND_DECIMALS)
            .alias(markout_col)
        )

    columns_to_drop = ["__row_id"] + [
        f"__lookup_ts_{_horizon_label(horizon_seconds)}s"
        for horizon_seconds in horizons_seconds
    ]
    if not keep_forward_mids:
        columns_to_drop.extend(
            [f"mid_at_t_plus_{_horizon_label(horizon_seconds)}s" for horizon_seconds in horizons_seconds]
        )

    return working.sort("__row_id").drop(columns_to_drop)


def aggregate_markouts_by_tag(
    markouts: pl.DataFrame,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """Per-tag mean, median, P10, P90, and % negative markout at each horizon."""
    aggregations: list[pl.Expr] = [
        pl.len().alias("trade_count"),
        pl.sum("notional_usd").alias("notional_usd"),
    ]
    for horizon_seconds in horizons_seconds:
        label = _horizon_label(horizon_seconds)
        markout_col = f"markout_bp_{label}s"
        aggregations.extend(
            [
                pl.col(markout_col).mean().round(MARKOUT_ROUND_DECIMALS).alias(f"mean_markout_bp_{label}s"),
                pl.col(markout_col).median().round(MARKOUT_ROUND_DECIMALS).alias(f"median_markout_bp_{label}s"),
                pl.col(markout_col).quantile(0.10).round(MARKOUT_ROUND_DECIMALS).alias(f"p10_markout_bp_{label}s"),
                pl.col(markout_col).quantile(0.90).round(MARKOUT_ROUND_DECIMALS).alias(f"p90_markout_bp_{label}s"),
                (100.0 * pl.col(markout_col).lt(0).mean())
                .round(4)
                .alias(f"pct_negative_markout_{label}s"),
            ]
        )

    return markouts.group_by("tag").agg(aggregations).sort("trade_count", descending=True)


def aggregate_markouts_by_tag_pair(
    markouts: pl.DataFrame,
    horizon_seconds: float = 30.0,
) -> pl.DataFrame:
    """Per-(tag, pair) toxicity table at a single horizon for dashboard heatmaps."""
    label = _horizon_label(horizon_seconds)
    markout_col = f"markout_bp_{label}s"
    mean_col = f"mean_markout_bp_{label}s"
    return (
        markouts.group_by(["tag", "pair"])
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.sum("notional_usd").alias("notional_usd"),
                pl.col(markout_col).mean().round(MARKOUT_ROUND_DECIMALS).alias(mean_col),
                (-pl.col(markout_col).mean()).round(MARKOUT_ROUND_DECIMALS).alias(f"toxicity_score_{label}s"),
                (100.0 * pl.col(markout_col).lt(0).mean()).round(4).alias(f"pct_negative_markout_{label}s"),
            ]
        )
        .sort(["tag", "pair"])
    )
