"""Toxicity aggregation helpers for tag and tag-pair flow analysis."""

from __future__ import annotations

from math import nan
from typing import Optional, Sequence

import polars as pl

from src.markouts import DEFAULT_HORIZONS_SECONDS


def _horizon_label(horizon_seconds: float) -> str:
    if float(horizon_seconds).is_integer():
        return str(int(horizon_seconds))
    return str(horizon_seconds).replace(".", "p")


def _markout_column(horizon_seconds: float) -> str:
    return f"markout_bp_{_horizon_label(horizon_seconds)}s"


def _filtered_markouts(markouts: pl.DataFrame, tags: Optional[Sequence[str]]) -> pl.DataFrame:
    if not tags:
        return markouts
    return markouts.filter(pl.col("tag").is_in(tags))


def compute_tag_horizon_curve(
    markouts: pl.DataFrame,
    tags: Optional[Sequence[str]] = None,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """
    Long-format markout summary for mean-vs-horizon charting with 95% CIs.
    """
    filtered = _filtered_markouts(markouts, tags)
    curve_frames: list[pl.DataFrame] = []

    for horizon_seconds in horizons_seconds:
        markout_col = _markout_column(horizon_seconds)
        per_horizon = (
            filtered.filter(pl.col(markout_col).is_not_null())
            .group_by("tag")
            .agg(
                [
                    pl.len().alias("trade_count"),
                    pl.col(markout_col).mean().alias("mean_bp"),
                    pl.col(markout_col).std().alias("std_bp"),
                ]
            )
            .with_columns(
                [
                    pl.lit(float(horizon_seconds)).alias("horizon_s"),
                    pl.when(pl.col("trade_count") > 1)
                    .then(1.96 * pl.col("std_bp") / pl.col("trade_count").sqrt())
                    .otherwise(pl.lit(nan))
                    .alias("ci_half_width_bp"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("trade_count") > 1)
                    .then(pl.col("mean_bp") - pl.col("ci_half_width_bp"))
                    .otherwise(pl.lit(nan))
                    .alias("ci_low_bp"),
                    pl.when(pl.col("trade_count") > 1)
                    .then(pl.col("mean_bp") + pl.col("ci_half_width_bp"))
                    .otherwise(pl.lit(nan))
                    .alias("ci_high_bp"),
                ]
            )
            .select(["tag", "horizon_s", "mean_bp", "ci_low_bp", "ci_high_bp", "trade_count"])
        )
        curve_frames.append(per_horizon)

    if not curve_frames:
        return pl.DataFrame(
            schema={
                "tag": pl.String,
                "horizon_s": pl.Float64,
                "mean_bp": pl.Float64,
                "ci_low_bp": pl.Float64,
                "ci_high_bp": pl.Float64,
                "trade_count": pl.UInt32,
            }
        )

    return pl.concat(curve_frames).sort(["tag", "horizon_s"])


def compute_tag_pair_toxicity(
    markouts: pl.DataFrame,
    horizon_seconds: float = 30.0,
) -> pl.DataFrame:
    """
    Per-(tag, pair) toxicity score at a single horizon for heatmap rendering.
    """
    markout_col = _markout_column(horizon_seconds)
    return (
        markouts.filter(pl.col(markout_col).is_not_null())
        .group_by(["tag", "pair"])
        .agg(
            [
                pl.len().alias("trade_count"),
                (-pl.col(markout_col).mean()).alias("toxicity_score"),
            ]
        )
        .sort(["tag", "pair"])
    )


def get_markout_distribution(
    markouts: pl.DataFrame,
    horizon_seconds: float = 60.0,
    tags: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Per-trade markouts at a single horizon for distribution charts.
    """
    markout_col = _markout_column(horizon_seconds)
    return (
        _filtered_markouts(markouts, tags)
        .select(["tag", pl.col(markout_col).alias("markout_bp")])
        .filter(pl.col("markout_bp").is_not_null())
        .sort(["tag", "markout_bp"])
    )


def per_tag_horizon_table(
    markouts: pl.DataFrame,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """
    Wide-format per-tag markout summary used by the detailed toxicity table.
    """
    aggregations: list[pl.Expr] = [
        pl.len().alias("trade_count"),
        (pl.sum("notional_usd") / 1_000_000.0).alias("notional_usd_mm"),
    ]

    for horizon_seconds in horizons_seconds:
        label = _horizon_label(horizon_seconds)
        markout_col = _markout_column(horizon_seconds)
        aggregations.append(pl.col(markout_col).mean().alias(f"mean_bp_{label}s"))

    summary = markouts.group_by("tag").agg(aggregations)

    reporting_label = _horizon_label(30.0)
    routing_label = _horizon_label(60.0)

    return (
        summary.with_columns(
            [
                (-pl.col(f"mean_bp_{reporting_label}s")).alias("reporting_score_30s"),
                (-pl.col(f"mean_bp_{routing_label}s")).alias("routing_score_60s"),
            ]
        )
        .sort("routing_score_60s", descending=True)
    )
