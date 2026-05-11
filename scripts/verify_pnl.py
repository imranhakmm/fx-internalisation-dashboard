from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.markouts import compute_markouts
from src.pnl import compute_pnl, summarise_pnl_by_pair, summarise_pnl_by_tag
from src.pricing import (
    DEFAULT_CONFIG,
    assign_routes,
    compute_routing_toxicity_scores,
    compute_tag_toxicity_scores,
)


def _format_summary(df: pl.DataFrame, round_digits: int = 2) -> pl.DataFrame:
    float_columns = [name for name, dtype in df.schema.items() if dtype.is_float()]
    if not float_columns:
        return df
    return df.with_columns([pl.col(column).round(round_digits) for column in float_columns])


def _route_breakdown(pnl_df: pl.DataFrame) -> pl.DataFrame:
    return (
        pnl_df.group_by(["tag", "route"])
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.sum("notional_usd").alias("notional_usd"),
                pl.sum("pnl_usd").alias("pnl_usd"),
            ]
        )
        .sort(["tag", "route"])
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _toxicity_score_table(
    reporting_toxicity_scores: dict[str, float],
    routing_toxicity_scores: dict[str, float],
    routing_horizon_seconds: float,
) -> pl.DataFrame:
    tags = sorted(set(reporting_toxicity_scores) | set(routing_toxicity_scores))
    return pl.DataFrame(
        {
            "tag": tags,
            "toxicity_score_30s": [reporting_toxicity_scores.get(tag) for tag in tags],
            f"toxicity_score_{int(routing_horizon_seconds)}s": [
                routing_toxicity_scores.get(tag) for tag in tags
            ],
        }
    )


def main() -> None:
    trades = pl.read_parquet(PROJECT_ROOT / "data" / "trades.parquet")
    ticks = pl.read_parquet(PROJECT_ROOT / "data" / "ticks.parquet")

    required_horizons = tuple(dict.fromkeys((30.0, DEFAULT_CONFIG.inventory_decay_seconds)))
    markouts = compute_markouts(trades=trades, ticks=ticks, horizons_seconds=required_horizons)
    reporting_toxicity_scores = compute_tag_toxicity_scores(markouts, horizon_seconds=30.0)
    routing_toxicity_scores = compute_routing_toxicity_scores(markouts, DEFAULT_CONFIG)
    toxicity_table = _toxicity_score_table(
        reporting_toxicity_scores,
        routing_toxicity_scores,
        DEFAULT_CONFIG.inventory_decay_seconds,
    )
    routed = assign_routes(trades, DEFAULT_CONFIG, routing_toxicity_scores)
    pnl_df = compute_pnl(routed, ticks, DEFAULT_CONFIG)

    pair_summary = summarise_pnl_by_pair(pnl_df)
    tag_summary = summarise_pnl_by_tag(pnl_df)
    route_breakdown = _route_breakdown(pnl_df)

    total_notional = float(pnl_df["notional_usd"].sum())
    total_pnl = float(pnl_df["pnl_usd"].sum())
    total_spread_capture = float(pnl_df["spread_capture_usd"].sum())
    internalised_notional = float(
        pnl_df.filter(pl.col("route") == "internalise")["notional_usd"].sum() or 0.0
    )
    hedged_notional = float(pnl_df.filter(pl.col("route") == "hedge")["notional_usd"].sum() or 0.0)
    pnl_per_million = total_pnl / (total_notional / 1_000_000.0)

    print("Toxicity scores (reporting 30s vs routing hold horizon):")
    print(_format_summary(toxicity_table, round_digits=4))
    print()

    print("Per-pair PnL summary:")
    print(_format_summary(pair_summary))
    print()

    print("Per-tag PnL summary:")
    print(_format_summary(tag_summary))
    print()

    print("Routing breakdown by tag:")
    print(_format_summary(route_breakdown))
    print()

    print("Overall:")
    print(
        f"total_notional_usd={total_notional:,.2f}, "
        f"total_pnl_usd={total_pnl:,.2f}, "
        f"pnl_per_million={pnl_per_million:,.2f}, "
        f"internalisation_ratio={100.0 * _ratio(internalised_notional, total_notional):.2f}%, "
        f"hedge_ratio={100.0 * _ratio(hedged_notional, total_notional):.2f}%"
    )
    print(
        f"total_spread_capture_usd={total_spread_capture:,.2f}, "
        f"internalised_notional_usd={internalised_notional:,.2f}, "
        f"hedged_notional_usd={hedged_notional:,.2f}"
    )
    print()

    hft_a_hedge_ratio = _ratio(
        float(
            pnl_df.filter((pl.col("tag") == "HFT_A") & (pl.col("route") == "hedge"))["notional_usd"].sum() or 0.0
        ),
        float(pnl_df.filter(pl.col("tag") == "HFT_A")["notional_usd"].sum() or 0.0),
    )
    retail_hedge_ratio = _ratio(
        float(
            pnl_df.filter((pl.col("tag") == "RetailAgg_D") & (pl.col("route") == "hedge"))["notional_usd"].sum()
            or 0.0
        ),
        float(pnl_df.filter(pl.col("tag") == "RetailAgg_D")["notional_usd"].sum() or 0.0),
    )

    checks = [
        (
            "Total PnL is non-zero",
            total_pnl != 0.0,
            f"measured total_pnl_usd={total_pnl:,.2f}",
        ),
        (
            "Internalisation ratio is between 10% and 40%",
            0.10 <= _ratio(internalised_notional, total_notional) <= 0.40,
            f"measured internalisation_ratio={100.0 * _ratio(internalised_notional, total_notional):.2f}%",
        ),
        (
            "Spread capture USD is non-negative for every trade",
            pnl_df.filter(pl.col("spread_capture_usd") < 0.0).is_empty(),
            f"violations={pnl_df.filter(pl.col('spread_capture_usd') < 0.0).height}",
        ),
        (
            "Hedged trades have zero adverse selection USD",
            pnl_df.filter((pl.col("route") == "hedge") & (pl.col("adverse_selection_usd") != 0.0)).is_empty(),
            f"violations={pnl_df.filter((pl.col('route') == 'hedge') & (pl.col('adverse_selection_usd') != 0.0)).height}",
        ),
        (
            "HFT_A is hedged more aggressively than RetailAgg_D",
            hft_a_hedge_ratio > retail_hedge_ratio,
            f"measured hedge_ratio_notional: HFT_A={100.0 * hft_a_hedge_ratio:.2f}% vs RetailAgg_D={100.0 * retail_hedge_ratio:.2f}%",
        ),
    ]

    print("Soft checks:")
    for description, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"- {status}: {description}")
        print(f"  {detail}")


if __name__ == "__main__":
    main()
