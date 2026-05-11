from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import config_hash, load_actions_bundle, load_dashboard_data
from app.formatting import fmt_count, fmt_usd_signed
from src.actions import Recommendation, apply_recommendation_to_config
from src.pricing import config_to_payload, get_active_config

ACTION_STYLE = {
    "KEEP": "background-color: #DCEAD9; color: #2A4A2A;",
    "WIDEN": "background-color: #F7E2BF; color: #7A4B00;",
    "HEDGE": "background-color: #F3D4D4; color: #8C2F39;",
    "INTERNALISE": "background-color: #DDE7F0; color: #1F3A56;",
    "EXCLUDE": "background-color: #ECEFF3; color: #44505C;",
}

ROUTE_STYLE = {
    "internalise": "background-color: #DCEAD9; color: #2A4A2A;",
    "hedge": "background-color: #DDE7F0; color: #1F3A56;",
    "reject": "background-color: #ECEFF3; color: #44505C;",
    "mixed": "background-color: #F3EBD1; color: #6E5A22;",
}

ACTION_TEXT = {
    "KEEP": "Keep",
    "WIDEN": "Widen",
    "HEDGE": "Hedge",
    "INTERNALISE": "Internalise",
    "EXCLUDE": "Exclude",
}


def _pill(label: str, style: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.22rem 0.55rem;border-radius:999px;"
        f"font-size:0.82rem;font-weight:700;{style}'>{label}</span>"
    )


def _route_badge(route: str) -> str:
    label = {
        "hedge": "Hedged",
        "internalise": "Internalised",
        "reject": "Rejected",
        "mixed": "Mixed",
    }.get(route, route.title())
    return _pill(label, ROUTE_STYLE.get(route, ROUTE_STYLE["mixed"]))


def _action_label(rec: Recommendation) -> str:
    if rec.action_type == "WIDEN" and rec.widen_amount_bp is not None:
        return f"WIDEN +{rec.widen_amount_bp:.2f} bp"
    return rec.action_type


def _action_badge(rec: Recommendation) -> str:
    return _pill(_action_label(rec), ACTION_STYLE[rec.action_type])


def _combined_improvement(current_total: float, combined_total: float) -> float:
    return combined_total - current_total


def _escape_markdown_currency(text: str) -> str:
    return text.replace("$", "\\$")


def _apply_all_recommendations(config, recommendations: list[Recommendation]):
    updated_config = config
    for recommendation in recommendations:
        if recommendation.action_type == "KEEP":
            continue
        updated_config = apply_recommendation_to_config(updated_config, recommendation)
    return updated_config


def render_page() -> None:
    st.set_page_config(page_title="Actions", layout="wide")
    st.title("Actions")
    st.caption("Auto-generated recommendations from flow behaviour and current PnL. One-click apply propagates to all pages.")

    active_config = get_active_config()
    dashboard_data = load_dashboard_data(config_hash(active_config))
    current_total_pnl = float(dashboard_data["attribution_totals"]["net_pnl_usd"])

    actions_bundle = load_actions_bundle(config_to_payload(active_config))
    recommendations: list[Recommendation] = actions_bundle["recommendations"]
    combined_projection = actions_bundle["combined_projection"]
    combined_total_pnl = float(combined_projection["attribution_totals"]["net_pnl_usd"])
    non_keep_count = sum(1 for rec in recommendations if rec.action_type != "KEEP")

    with st.container(border=True):
        summary_cols = st.columns(3, gap="large")
        summary_cols[0].metric("Total recommended actions", fmt_count(non_keep_count))
        summary_cols[1].metric("If all applied, projected PnL", fmt_usd_signed(combined_total_pnl))
        summary_cols[2].metric(
            "Projected improvement",
            fmt_usd_signed(_combined_improvement(current_total_pnl, combined_total_pnl)),
        )

    for recommendation in recommendations:
        with st.container(border=True):
            cols = st.columns([2, 4, 3], gap="large")

            with cols[0]:
                st.markdown(f"### {recommendation.tag}")
                st.markdown("**Currently**", unsafe_allow_html=False)
                st.markdown(_route_badge(recommendation.current_route), unsafe_allow_html=True)
                st.markdown(f"**Current PnL:** {fmt_usd_signed(recommendation.current_pnl_tag)}")

            with cols[1]:
                st.markdown("**Recommended action**", unsafe_allow_html=False)
                st.markdown(_action_badge(recommendation), unsafe_allow_html=True)
                st.markdown(_escape_markdown_currency(recommendation.reasoning))
                for note in recommendation.secondary_notes:
                    st.caption(f"ℹ️ {note}")

            with cols[2]:
                st.metric("Projected impact", fmt_usd_signed(recommendation.pnl_delta))
                if recommendation.action_type != "KEEP":
                    if st.button(
                        "Apply",
                        key=f"apply_{recommendation.tag}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["pricing_config"] = apply_recommendation_to_config(active_config, recommendation)
                        st.toast(f"Applied {_action_label(recommendation)} for {recommendation.tag}", icon="✅")
                        st.rerun()
                else:
                    st.caption("No action needed")

    st.divider()
    action_cols = st.columns([2, 6], gap="large")
    if action_cols[0].button("Apply all non-KEEP recommendations", type="primary", use_container_width=True):
        st.session_state["pricing_config"] = _apply_all_recommendations(active_config, recommendations)
        st.toast("Applied all current recommendations", icon="✅")
        st.rerun()

    st.caption(
        "Each recommendation's projected impact is computed independently against the current config. "
        "Combined effects when applying multiple actions may differ slightly due to interactions."
    )


if __name__ == "__main__":
    render_page()
