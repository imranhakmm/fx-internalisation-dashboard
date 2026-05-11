"""Recommendation engine for per-tag internalisation actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional

import polars as pl

from src.markouts import _horizon_label, compute_markouts
from src.pnl import compute_pnl, summarise_pnl_by_tag
from src.pricing import (
    PricingConfig,
    assign_routes,
    compute_routing_toxicity_scores,
    config_to_payload,
    reprice_trades,
)
from src.toxicity import compute_tag_pair_toxicity

MIN_IMPROVEMENT_USD = 5_000.0
CANDIDATE_WIDENS_BP: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
ActionType = Literal["KEEP", "WIDEN", "HEDGE", "INTERNALISE", "EXCLUDE"]


@dataclass(frozen=True)
class Recommendation:
    tag: str
    action_type: ActionType
    widen_amount_bp: Optional[float]
    current_route: str
    current_pnl_total: float
    projected_pnl_total: float
    pnl_delta: float
    reasoning: str
    secondary_notes: List[str]
    current_pnl_tag: float = 0.0
    projected_pnl_tag: float = 0.0


def _snapshot_key(config: PricingConfig) -> str:
    return json.dumps(config_to_payload(config), sort_keys=True)


def _route_by_tag(trades_with_pnl: pl.DataFrame) -> dict[str, str]:
    route_table = (
        trades_with_pnl.group_by("tag")
        .agg(
            [
                pl.col("route").n_unique().alias("route_count"),
                pl.col("route").first().alias("first_route"),
            ]
        )
        .with_columns(
            pl.when(pl.col("route_count") == 1)
            .then(pl.col("first_route"))
            .otherwise(pl.lit("mixed"))
            .alias("route")
        )
        .select(["tag", "route"])
    )
    return {str(row["tag"]): str(row["route"]) for row in route_table.iter_rows(named=True)}


def _per_tag_map(per_tag: pl.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row["tag"]): {key: float(value or 0.0) if key != "tag" else str(value) for key, value in row.items()}
        for row in per_tag.iter_rows(named=True)
    }


def _evaluate_config(
    trades: pl.DataFrame,
    ticks: pl.DataFrame,
    routing_scores: dict[str, float],
    config: PricingConfig,
) -> dict[str, object]:
    repriced_trades = reprice_trades(trades, config)
    routed_trades = assign_routes(repriced_trades, config, routing_scores)
    pnl_df = compute_pnl(routed_trades, ticks, config)
    per_tag = summarise_pnl_by_tag(pnl_df)
    return {
        "config": config,
        "trades_with_pnl": pnl_df,
        "per_tag": per_tag,
        "per_tag_map": _per_tag_map(per_tag),
        "route_by_tag": _route_by_tag(pnl_df),
        "total_pnl": float(pnl_df["pnl_usd"].sum() or 0.0),
    }


def _build_secondary_notes(
    trades: pl.DataFrame,
    tag_pair_toxicity: pl.DataFrame,
    tag: str,
) -> List[str]:
    notes: list[str] = []
    tag_pair = tag_pair_toxicity.filter(pl.col("tag") == tag).sort("toxicity_score", descending=True)

    if tag_pair.height >= 2:
        high_row = tag_pair.row(0, named=True)
        low_row = tag_pair.row(tag_pair.height - 1, named=True)
        spread = float(high_row["toxicity_score"]) - float(low_row["toxicity_score"])
        if spread > 0.5:
            notes.append(
                "Toxicity asymmetric across pairs: "
                f"{high_row['pair']} score {float(high_row['toxicity_score']):.2f} vs "
                f"{low_row['pair']} score {float(low_row['toxicity_score']):.2f}. "
                "Per-pair widening or routing would extract more value than the per-tag treatment used here."
            )

    avg_notional_mm = (
        trades.filter(pl.col("tag") == tag).select((pl.mean("notional_usd") / 1_000_000.0).alias("avg")).item()
    )
    if float(avg_notional_mm or 0.0) > 5.0:
        notes.append(
            f"Large average ticket size ({float(avg_notional_mm):.1f} mm). "
            "Score-based threshold underweights notional; consider notional-aware routing in production."
        )

    anti_toxic_pairs = tag_pair.filter(pl.col("toxicity_score") < -0.1).sort("toxicity_score")
    if anti_toxic_pairs.height > 0:
        row = anti_toxic_pairs.row(0, named=True)
        notes.append(
            f"{row['pair']} flow is anti-toxic (LP earns markout). "
            "Could increase internalisation share for this tag-pair specifically."
        )

    return notes


def _reasoning_for_action(
    tag: str,
    action_type: ActionType,
    widen_amount_bp: Optional[float],
    current_route: str,
    current_metrics: dict[str, float],
    projected_metrics: dict[str, float],
    current_total_pnl: float,
    projected_total_pnl: float,
    score: float,
) -> str:
    notional_mm = current_metrics["notional_usd"] / 1_000_000.0
    current_tag_pnl = current_metrics["pnl_usd"]
    projected_tag_pnl = projected_metrics["pnl_usd"]
    total_delta = projected_total_pnl - current_total_pnl

    if action_type == "HEDGE":
        return (
            f"Internalising {tag} costs ${current_metrics['as_usd']:,.0f} in AS drag against "
            f"${current_metrics['spread_capture_usd']:,.0f} captured spread. Hedging the same flow "
            f"captures the spread and pays ~${projected_metrics['hedge_cost_usd']:,.0f} in hedge costs, "
            f"netting {projected_tag_pnl:+,.0f} for the tag and lifting total book PnL to "
            f"{projected_total_pnl:+,.0f} from {current_total_pnl:+,.0f}. "
            f"Score-based threshold misses the economics here because score is only {score:.2f} while "
            f"notional is ${notional_mm:,.0f} mm."
        )

    if action_type == "WIDEN" and widen_amount_bp is not None:
        extra_capture = projected_metrics["spread_capture_usd"] - current_metrics["spread_capture_usd"]
        route_phrase = "without changing the routing decision" if current_route != "mixed" else "while keeping routing broadly unchanged"
        return (
            f"{tag} is near the routing boundary at score {score:.2f}. Widening by {widen_amount_bp:.2f} bp "
            f"adds approximately ${extra_capture:,.0f} of spread capture {route_phrase}, lifting tag PnL "
            f"from {current_tag_pnl:+,.0f} to {projected_tag_pnl:+,.0f}. "
            f"That moves total book PnL to {projected_total_pnl:+,.0f}, a {total_delta:+,.0f} change."
        )

    if action_type == "INTERNALISE":
        return (
            f"{tag} hedges ${notional_mm:,.0f} mm of notional at score {score:.2f} — low enough that "
            f"internalising would only expose ~${projected_metrics['as_usd']:,.0f} of AS while saving "
            f"${current_metrics['hedge_cost_usd']:,.0f} of hedge cost. "
            f"Tag PnL moves from {current_tag_pnl:+,.0f} to {projected_tag_pnl:+,.0f}, for a total-book delta "
            f"of {total_delta:+,.0f}."
        )

    if action_type == "EXCLUDE":
        return (
            f"{tag} currently contributes {current_tag_pnl:+,.0f}, but the flow remains uneconomic after routing "
            f"and widening alternatives. Rejecting it removes the drag and lifts total book PnL to "
            f"{projected_total_pnl:+,.0f}, a {total_delta:+,.0f} improvement."
        )

    return (
        f"{tag} contributes {current_tag_pnl:+,.0f} at current config. "
        f"No alternative action improves total book PnL by more than ${MIN_IMPROVEMENT_USD:,.0f}."
    )


def _mutate_config_for_action(
    config: PricingConfig,
    tag: str,
    action_type: Literal["WIDEN", "HEDGE", "INTERNALISE", "EXCLUDE"],
    widen_amount_bp: Optional[float] = None,
) -> PricingConfig:
    tag_widen_bp = dict(config.tag_widen_bp)
    tag_route_override = dict(config.tag_route_override)

    if action_type == "WIDEN" and widen_amount_bp is not None:
        tag_widen_bp[tag] = widen_amount_bp
        tag_route_override.pop(tag, None)
    elif action_type == "HEDGE":
        tag_route_override[tag] = "hedge"
    elif action_type == "INTERNALISE":
        tag_route_override[tag] = "internalise"
    elif action_type == "EXCLUDE":
        tag_route_override[tag] = "reject"

    return replace(config, tag_widen_bp=tag_widen_bp, tag_route_override=tag_route_override)


def _ensure_markouts_have_horizon(
    trades: pl.DataFrame,
    ticks: pl.DataFrame,
    markouts: pl.DataFrame,
    horizon_seconds: float,
) -> pl.DataFrame:
    markout_col = f"markout_bp_{_horizon_label(horizon_seconds)}s"
    if markout_col in markouts.columns:
        return markouts

    extra_markouts = compute_markouts(
        trades=trades,
        ticks=ticks,
        horizons_seconds=(horizon_seconds,),
    ).select(["ts", "pair", "tag", "side", "notional_usd", markout_col])

    return markouts.join(
        extra_markouts,
        on=["ts", "pair", "tag", "side", "notional_usd"],
        how="left",
    )


def generate_recommendations(
    trades: pl.DataFrame,
    ticks: pl.DataFrame,
    markouts: pl.DataFrame,
    current_config: PricingConfig,
) -> List[Recommendation]:
    """
    Generate one ranked recommendation per tag under the current config.
    """
    markouts = _ensure_markouts_have_horizon(
        trades=trades,
        ticks=ticks,
        markouts=markouts,
        horizon_seconds=current_config.inventory_decay_seconds,
    )
    routing_scores = compute_routing_toxicity_scores(markouts, current_config)
    tag_pair_toxicity = compute_tag_pair_toxicity(markouts, horizon_seconds=current_config.inventory_decay_seconds)
    current_snapshot = _evaluate_config(trades, ticks, routing_scores, current_config)
    current_total_pnl = float(current_snapshot["total_pnl"])
    current_routes: dict[str, str] = current_snapshot["route_by_tag"]  # type: ignore[assignment]
    tags = trades["tag"].unique().sort().to_list()
    snapshot_cache: dict[str, dict[str, object]] = {_snapshot_key(current_config): current_snapshot}
    recommendations: list[Recommendation] = []

    for tag in tags:
        current_route = current_routes[tag]
        current_metrics = dict(current_snapshot["per_tag_map"][tag])  # type: ignore[index]
        secondary_notes = _build_secondary_notes(trades, tag_pair_toxicity, tag)

        best_action_type: ActionType = "KEEP"
        best_widen: Optional[float] = None
        best_snapshot = current_snapshot

        candidate_specs: list[tuple[Literal["WIDEN", "HEDGE", "INTERNALISE", "EXCLUDE"], Optional[float]]] = []
        if current_route != "hedge":
            candidate_specs.extend([("WIDEN", widen_bp) for widen_bp in CANDIDATE_WIDENS_BP])
            candidate_specs.append(("HEDGE", None))
        if current_route == "hedge":
            candidate_specs.append(("INTERNALISE", None))
        if current_route != "reject":
            candidate_specs.append(("EXCLUDE", None))

        for action_type, widen_bp in candidate_specs:
            candidate_config = _mutate_config_for_action(current_config, tag, action_type, widen_bp)
            cache_key = _snapshot_key(candidate_config)
            if cache_key not in snapshot_cache:
                snapshot_cache[cache_key] = _evaluate_config(trades, ticks, routing_scores, candidate_config)
            candidate_snapshot = snapshot_cache[cache_key]

            if float(candidate_snapshot["total_pnl"]) > float(best_snapshot["total_pnl"]):
                best_action_type = action_type
                best_widen = widen_bp
                best_snapshot = candidate_snapshot

        projected_total_pnl = float(best_snapshot["total_pnl"])
        projected_metrics = dict(best_snapshot["per_tag_map"][tag])  # type: ignore[index]
        pnl_delta = projected_total_pnl - current_total_pnl

        if pnl_delta < MIN_IMPROVEMENT_USD:
            best_action_type = "KEEP"
            best_widen = None
            best_snapshot = current_snapshot
            projected_total_pnl = current_total_pnl
            projected_metrics = current_metrics
            pnl_delta = 0.0

        reasoning = _reasoning_for_action(
            tag=tag,
            action_type=best_action_type,
            widen_amount_bp=best_widen,
            current_route=current_route,
            current_metrics=current_metrics,
            projected_metrics=projected_metrics,
            current_total_pnl=current_total_pnl,
            projected_total_pnl=projected_total_pnl,
            score=routing_scores[tag],
        )
        recommendations.append(
            Recommendation(
                tag=tag,
                action_type=best_action_type,
                widen_amount_bp=best_widen,
                current_route=current_route,
                current_pnl_total=current_total_pnl,
                projected_pnl_total=projected_total_pnl,
                pnl_delta=pnl_delta,
                reasoning=reasoning,
                secondary_notes=secondary_notes,
                current_pnl_tag=current_metrics["pnl_usd"],
                projected_pnl_tag=projected_metrics["pnl_usd"],
            )
        )

    return sorted(recommendations, key=lambda rec: rec.pnl_delta, reverse=True)


def apply_recommendation_to_config(
    config: PricingConfig,
    rec: Recommendation,
) -> PricingConfig:
    """
    Apply a single recommendation mutation to a config.
    """
    if rec.action_type == "KEEP":
        return config
    if rec.action_type == "WIDEN":
        return _mutate_config_for_action(config, rec.tag, "WIDEN", rec.widen_amount_bp)
    if rec.action_type == "HEDGE":
        return _mutate_config_for_action(config, rec.tag, "HEDGE")
    if rec.action_type == "INTERNALISE":
        return _mutate_config_for_action(config, rec.tag, "INTERNALISE")
    return _mutate_config_for_action(config, rec.tag, "EXCLUDE")
