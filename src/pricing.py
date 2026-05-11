"""Pricing configuration, trade repricing, and routing rules."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl

from src.markouts import _horizon_label


@dataclass(frozen=True)
class PricingConfig:
    """Runtime pricing and routing controls for the internalisation book."""

    half_spread_bp: Mapping[str, float]
    internalise_size_threshold_usd: float
    internalise_toxicity_threshold: float
    hedge_cost_bp: Mapping[str, float]
    inventory_decay_seconds: float
    tag_widen_bp: Mapping[str, float]
    tag_route_override: Mapping[str, str]


DEFAULT_CONFIG = PricingConfig(
    half_spread_bp={"EURUSD": 0.2, "USDJPY": 0.3, "GBPUSD": 0.4},
    internalise_size_threshold_usd=5_000_000,
    internalise_toxicity_threshold=0.3,
    # LPs earn the client spread and pay a tighter inter-dealer spread when hedging.
    # A realistic default is roughly one-third of the client half-spread.
    hedge_cost_bp={"EURUSD": 0.06, "USDJPY": 0.09, "GBPUSD": 0.12},
    inventory_decay_seconds=60.0,
    tag_widen_bp={},
    tag_route_override={},
)


def _tag_toxicity_lookup_frame(tag_toxicity_scores: Mapping[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tag": pl.Series("tag", list(tag_toxicity_scores.keys()), dtype=pl.String),
            "tag_toxicity_score": pl.Series(
                "tag_toxicity_score",
                list(tag_toxicity_scores.values()),
                dtype=pl.Float64,
            ),
        }
    )


def _tag_override_lookup_frame(tag_route_override: Mapping[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tag": pl.Series("tag", list(tag_route_override.keys()), dtype=pl.String),
            "tag_route_override": pl.Series(
                "tag_route_override",
                list(tag_route_override.values()),
                dtype=pl.String,
            ),
        }
    )


def _pair_value_lookup_frame(mapping: Mapping[str, float], value_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pair": pl.Series("pair", list(mapping.keys()), dtype=pl.String),
            value_name: pl.Series(value_name, list(mapping.values()), dtype=pl.Float64),
        }
    )


def _tag_widen_lookup_frame(tag_widen_bp: Mapping[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tag": pl.Series("tag", list(tag_widen_bp.keys()), dtype=pl.String),
            "tag_widen_bp": pl.Series("tag_widen_bp", list(tag_widen_bp.values()), dtype=pl.Float64),
        }
    )


def _side_sign_expr() -> pl.Expr:
    return pl.when(pl.col("side") == "BUY").then(1.0).otherwise(-1.0)


def config_to_payload(config: PricingConfig) -> dict[str, Any]:
    return asdict(config)


def config_from_payload(payload: Mapping[str, Any]) -> PricingConfig:
    return PricingConfig(
        half_spread_bp=dict(payload["half_spread_bp"]),
        internalise_size_threshold_usd=float(payload["internalise_size_threshold_usd"]),
        internalise_toxicity_threshold=float(payload["internalise_toxicity_threshold"]),
        hedge_cost_bp=dict(payload["hedge_cost_bp"]),
        inventory_decay_seconds=float(payload["inventory_decay_seconds"]),
        tag_widen_bp=dict(payload["tag_widen_bp"]),
        tag_route_override=dict(payload["tag_route_override"]),
    )


def reprice_trades(
    trades: pl.DataFrame,
    config: PricingConfig,
) -> pl.DataFrame:
    """
    Rebuild executed prices from the current config's pair spreads and tag widens.

    The historical dataset already bakes in baseline per-tag widens. This helper
    preserves that baseline and applies only the delta from the active config:
    pair spread changes relative to DEFAULT_CONFIG plus any explicit extra widen.
    """
    pair_spreads = _pair_value_lookup_frame(config.half_spread_bp, "configured_half_spread_bp")
    default_pair_spreads = _pair_value_lookup_frame(DEFAULT_CONFIG.half_spread_bp, "default_half_spread_bp")
    tag_widens = _tag_widen_lookup_frame(config.tag_widen_bp)

    return (
        trades.join(pair_spreads, on="pair", how="left")
        .join(default_pair_spreads, on="pair", how="left")
        .join(tag_widens, on="tag", how="left")
        .with_columns(
            [
                pl.col("configured_half_spread_bp").fill_null(0.0),
                pl.col("default_half_spread_bp").fill_null(0.0),
                pl.col("tag_widen_bp").fill_null(0.0),
            ]
        )
        .with_columns(
            [
                (
                    _side_sign_expr()
                    * 1e4
                    * (pl.col("executed_price") - pl.col("mid_at_trade"))
                    / pl.col("mid_at_trade")
                ).alias("historical_half_spread_bp"),
                (pl.col("configured_half_spread_bp") - pl.col("default_half_spread_bp")).alias("pair_spread_delta_bp"),
            ]
        )
        .with_columns(
            (pl.col("historical_half_spread_bp") + pl.col("pair_spread_delta_bp") + pl.col("tag_widen_bp")).alias(
                "effective_half_spread_bp"
            )
        )
        .with_columns(
            (
                pl.col("mid_at_trade")
                * (1.0 + _side_sign_expr() * pl.col("effective_half_spread_bp") * 1e-4)
            ).alias("executed_price")
        )
        .drop(
            [
                "configured_half_spread_bp",
                "default_half_spread_bp",
                "tag_widen_bp",
                "historical_half_spread_bp",
                "pair_spread_delta_bp",
                "effective_half_spread_bp",
            ]
        )
    )


def compute_tag_toxicity_scores(
    markouts: pl.DataFrame,
    horizon_seconds: float = 30.0,
) -> dict[str, float]:
    """
    Per-tag toxicity score = -mean(markout_bp_{horizon}s).
    Higher = more toxic from the LP perspective.
    Default 30s aligns with the standard reporting horizon used in dashboards.
    """
    label = _horizon_label(horizon_seconds)
    markout_col = f"markout_bp_{label}s"
    if markout_col not in markouts.columns:
        raise ValueError(f"Missing required markout column: {markout_col}")

    scores = (
        markouts.group_by("tag")
        .agg((-pl.col(markout_col).mean()).alias("toxicity_score"))
        .sort("tag")
        .iter_rows(named=True)
    )
    return {str(row["tag"]): float(row["toxicity_score"]) for row in scores}


def compute_routing_toxicity_scores(
    markouts: pl.DataFrame,
    config: PricingConfig,
) -> dict[str, float]:
    """
    Compute routing toxicity at the inventory decay horizon.

    Routing should be evaluated over the same horizon as inventory is held,
    while reporting may continue to use a standard 30s convention.
    """
    return compute_tag_toxicity_scores(markouts, horizon_seconds=config.inventory_decay_seconds)


def assign_routes(
    trades: pl.DataFrame,
    config: PricingConfig,
    routing_toxicity_scores: dict[str, float],
) -> pl.DataFrame:
    """
    Add vectorised routing decisions to the trade frame.

    Decision tree:
      1. Explicit tag override (internalise / hedge / reject)
      2. Size threshold
      3. Toxicity threshold
      4. Else internalise
    """
    toxicity_lookup = _tag_toxicity_lookup_frame(routing_toxicity_scores)
    override_lookup = _tag_override_lookup_frame(config.tag_route_override)

    return (
        trades.join(toxicity_lookup, on="tag", how="left")
        .join(override_lookup, on="tag", how="left")
        .with_columns(
            [
                pl.col("tag_toxicity_score").fill_null(0.0),
                pl.col("tag_route_override").fill_null("auto"),
            ]
        )
        .with_columns(
            pl.when(pl.col("tag_route_override").is_in(["internalise", "hedge", "reject"]))
            .then(pl.col("tag_route_override"))
            .when(pl.col("notional_usd") > config.internalise_size_threshold_usd)
            .then(pl.lit("hedge"))
            .when(pl.col("tag_toxicity_score") > config.internalise_toxicity_threshold)
            .then(pl.lit("hedge"))
            .otherwise(pl.lit("internalise"))
            .alias("route")
        )
        .drop("tag_route_override")
    )


def get_active_config() -> PricingConfig:
    """
    Return the active config from Streamlit session state when available.
    """
    import streamlit as st

    return st.session_state.get("pricing_config", DEFAULT_CONFIG)
