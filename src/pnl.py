"""Per-trade PnL decomposition for internalised and hedged flow."""

from __future__ import annotations

from typing import Dict, Mapping

import polars as pl

from src.markouts import _horizon_label, compute_markouts
from src.pricing import PricingConfig


def _side_sign_expr() -> pl.Expr:
    return pl.when(pl.col("side") == "BUY").then(1.0).otherwise(-1.0)


def _pair_lookup_frame(mapping: Mapping[str, float], value_name: str) -> pl.DataFrame:
    return pl.DataFrame({"pair": list(mapping.keys()), value_name: list(mapping.values())})


def compute_pnl(
    trades_with_routes: pl.DataFrame,
    ticks: pl.DataFrame,
    config: PricingConfig,
) -> pl.DataFrame:
    """
    Add per-trade spread capture, adverse selection, hedge cost, and net PnL.
    """
    horizon_label = _horizon_label(config.inventory_decay_seconds)
    forward_mid_col = f"mid_at_t_plus_{horizon_label}s"

    markout_df = compute_markouts(
        trades=trades_with_routes,
        ticks=ticks,
        horizons_seconds=(config.inventory_decay_seconds,),
        keep_forward_mids=True,
    )
    hedge_cost_lookup = _pair_lookup_frame(config.hedge_cost_bp, "hedge_cost_bp")

    return (
        markout_df.join(hedge_cost_lookup, on="pair", how="left")
        .with_columns(
            [
                (
                    _side_sign_expr()
                    * 1e4
                    * (pl.col("executed_price") - pl.col("mid_at_trade"))
                    / pl.col("mid_at_trade")
                ).alias("_raw_spread_capture_bp"),
                (
                    _side_sign_expr()
                    * 1e4
                    * (pl.col(forward_mid_col) - pl.col("mid_at_trade"))
                    / pl.col("mid_at_trade")
                ).alias("_raw_adverse_selection_bp"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("route") == "reject")
                .then(0.0)
                .otherwise(pl.col("_raw_spread_capture_bp"))
                .alias("spread_capture_bp"),
                pl.when(pl.col("route") == "internalise")
                .then(pl.col("_raw_adverse_selection_bp"))
                .when(pl.col("route") == "hedge")
                .then(0.0)
                .otherwise(0.0)
                .alias("adverse_selection_bp"),
            ]
        )
        .with_columns(
            [
                (pl.col("spread_capture_bp") * 1e-4 * pl.col("notional_usd")).alias("spread_capture_usd"),
                pl.when(pl.col("route") == "internalise")
                .then(pl.col("adverse_selection_bp") * 1e-4 * pl.col("notional_usd"))
                .when(pl.col("route") == "hedge")
                .then(0.0)
                .otherwise(0.0)
                .alias("adverse_selection_usd"),
                pl.when(pl.col("route") == "hedge")
                .then(pl.col("hedge_cost_bp") * 1e-4 * pl.col("notional_usd"))
                .otherwise(0.0)
                .alias("hedge_cost_usd"),
            ]
        )
        .with_columns(
            pl.when(pl.col("route") == "internalise")
            .then(
                pl.when(pl.col(forward_mid_col).is_null())
                .then(pl.col("spread_capture_usd"))
                .otherwise(pl.col("spread_capture_usd") - pl.col("adverse_selection_usd"))
            )
            .when(pl.col("route") == "hedge")
            .then(pl.col("spread_capture_usd") - pl.col("hedge_cost_usd"))
            .otherwise(0.0)
            .alias("pnl_usd")
        )
        .with_columns(
            pl.when((pl.col("route") == "internalise") & pl.col(forward_mid_col).is_null())
            .then(None)
            .otherwise(pl.col("adverse_selection_usd"))
            .alias("adverse_selection_usd")
        )
        .drop(["_raw_spread_capture_bp", "_raw_adverse_selection_bp"])
    )


def _summary_aggregations() -> list[pl.Expr]:
    return [
        pl.len().alias("trade_count"),
        pl.sum("notional_usd").alias("notional_usd"),
        pl.sum("pnl_usd").alias("pnl_usd"),
        pl.sum("spread_capture_usd").alias("spread_capture_usd"),
        pl.col("adverse_selection_usd").fill_null(0.0).sum().alias("as_usd"),
        pl.sum("hedge_cost_usd").alias("hedge_cost_usd"),
        pl.mean("spread_capture_bp").alias("mean_spread_captured_bp"),
    ]


def summarise_pnl_by_pair(pnl_df: pl.DataFrame) -> pl.DataFrame:
    """Per-pair PnL and spread summary."""
    return pnl_df.group_by("pair").agg(_summary_aggregations()).sort("notional_usd", descending=True)


def summarise_pnl_by_tag(pnl_df: pl.DataFrame) -> pl.DataFrame:
    """Per-tag PnL and spread summary."""
    return pnl_df.group_by("tag").agg(_summary_aggregations()).sort("notional_usd", descending=True)


def cumulative_pnl_timeseries(
    pnl_df: pl.DataFrame,
    bucket: str = "1h",
) -> pl.DataFrame:
    """
    Return cumulative book PnL components over time buckets.
    """
    bucketed = (
        pnl_df.with_columns(pl.col("ts").dt.truncate(bucket).alias("bucket_start"))
        .group_by("bucket_start")
        .agg(
            [
                pl.sum("spread_capture_usd").alias("spread_capture_usd"),
                pl.col("adverse_selection_usd").fill_null(0.0).sum().alias("as_drag_usd"),
                pl.sum("hedge_cost_usd").alias("hedge_cost_usd"),
                pl.sum("pnl_usd").alias("net_pnl_usd"),
            ]
        )
        .sort("bucket_start")
    )

    return bucketed.with_columns(
        [
            pl.col("spread_capture_usd").cum_sum().alias("spread_capture_cum"),
            pl.col("as_drag_usd").cum_sum().alias("as_drag_cum"),
            pl.col("hedge_cost_usd").cum_sum().alias("hedge_cost_cum"),
            pl.col("net_pnl_usd").cum_sum().alias("net_pnl_cum"),
        ]
    ).select(
        [
            "bucket_start",
            "spread_capture_cum",
            "as_drag_cum",
            "hedge_cost_cum",
            "net_pnl_cum",
        ]
    )


def pnl_attribution_totals(pnl_df: pl.DataFrame) -> Dict[str, float]:
    """
    Return total spread capture, AS drag, hedge cost, and net PnL in USD.
    """
    totals = pnl_df.select(
        [
            pl.sum("spread_capture_usd").alias("spread_capture_usd"),
            pl.col("adverse_selection_usd").fill_null(0.0).sum().alias("adverse_selection_usd"),
            pl.sum("hedge_cost_usd").alias("hedge_cost_usd"),
            pl.sum("pnl_usd").alias("net_pnl_usd"),
        ]
    ).row(0, named=True)
    return {key: float(value or 0.0) for key, value in totals.items()}


def pnl_attribution_by_hour(pnl_df: pl.DataFrame) -> pl.DataFrame:
    """
    Return per-hour, non-cumulative PnL attribution components.
    """
    return (
        pnl_df.with_columns(pl.col("ts").dt.truncate("1h").alias("hour_start"))
        .group_by("hour_start")
        .agg(
            [
                pl.sum("spread_capture_usd").alias("spread_capture_usd"),
                pl.col("adverse_selection_usd").fill_null(0.0).sum().alias("adverse_selection_usd"),
                pl.sum("hedge_cost_usd").alias("hedge_cost_usd"),
                pl.sum("pnl_usd").alias("net_pnl_usd"),
            ]
        )
        .sort("hour_start")
    )
