from __future__ import annotations

from typing import Mapping, Sequence

import plotly.graph_objects as go
import polars as pl

from app.formatting import fmt_usd_signed

CAPTURE_GREEN = "#A8D5BA"
ADVERSE_RED = "#F4B6B6"
HEDGE_BLUE = "#A6C8E0"
NET_DARK = "#1F3B5B"
INTERNALISE_GREEN = "#7FB685"
HEDGE_BLUE_GREY = "#6E8CA0"
POSITIVE_GREEN = "#7BB68F"
NEGATIVE_RED = "#D88A8A"
TAG_COLORS = {
    "HFT_A": "#A11D33",
    "HFT_B": "#D46A6A",
    "PB_C": "#B88A1C",
    "RetailAgg_D": "#4C956C",
    "Corp_E": "#6C757D",
    "Bank_F": "#2E86AB",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    stripped = hex_color.lstrip("#")
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def build_cumulative_pnl_chart(cumulative_ts: pl.DataFrame) -> go.Figure:
    x_values = cumulative_ts["bucket_start"].to_list()
    spread_capture = cumulative_ts["spread_capture_cum"].to_list()
    as_drag = [-value for value in cumulative_ts["as_drag_cum"].to_list()]
    hedge_cost = [-value for value in cumulative_ts["hedge_cost_cum"].to_list()]
    net_pnl = cumulative_ts["net_pnl_cum"].to_list()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=spread_capture,
            mode="lines",
            name="Spread capture",
            line={"color": CAPTURE_GREEN, "width": 0},
            fill="tozeroy",
            stackgroup="capture",
            hovertemplate="%{x|%H:%M}<br>Spread capture: $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=as_drag,
            mode="lines",
            name="AS drag",
            line={"color": ADVERSE_RED, "width": 0},
            fill="tozeroy",
            stackgroup="drag",
            hovertemplate="%{x|%H:%M}<br>AS drag: $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=hedge_cost,
            mode="lines",
            name="Hedge cost",
            line={"color": HEDGE_BLUE, "width": 0},
            fill="tonexty",
            stackgroup="drag",
            hovertemplate="%{x|%H:%M}<br>Hedge cost: $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=net_pnl,
            mode="lines",
            name="Net PnL",
            line={"color": NET_DARK, "width": 3},
            hovertemplate="%{x|%H:%M}<br>Net PnL: $%{y:,.0f}<extra></extra>",
        )
    )

    figure.update_layout(
        template="plotly_white",
        title={"text": "Cumulative PnL by source (USD)", "font": {"size": 16}},
        height=440,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    figure.update_yaxes(
        tickformat="$,.0f",
        gridcolor="#E0E0E0",
        zeroline=True,
        zerolinecolor="#888888",
        zerolinewidth=1,
    )
    figure.update_xaxes(showgrid=False, tickformat="%H:%M")
    return figure


def build_routing_breakdown_chart(
    route_share_df: pl.DataFrame,
    category_col: str,
    title: str,
) -> go.Figure:
    categories = route_share_df[category_col].to_list()
    internalise_pct = route_share_df["internalise_pct"].to_list()
    hedge_pct = route_share_df["hedge_pct"].to_list()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=categories,
            x=internalise_pct,
            orientation="h",
            name="Internalise",
            marker_color=INTERNALISE_GREEN,
            hovertemplate="%{y}<br>Internalise: %{x:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=categories,
            x=hedge_pct,
            orientation="h",
            name="Hedge",
            marker_color=HEDGE_BLUE_GREY,
            hovertemplate="%{y}<br>Hedge: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 14}},
        height=320,
        barmode="stack",
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    figure.update_xaxes(
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E8E8E8",
        title=None,
    )
    figure.update_yaxes(title=None, categoryorder="array", categoryarray=categories[::-1])
    return figure


def build_horizon_curve(curve_df: pl.DataFrame, selected_tags: Sequence[str]) -> go.Figure:
    figure = go.Figure()

    for tag in selected_tags:
        tag_curve = curve_df.filter(pl.col("tag") == tag).sort("horizon_s")
        if tag_curve.is_empty():
            continue

        color = TAG_COLORS.get(tag, NET_DARK)
        x_values = tag_curve["horizon_s"].to_list()
        ci_low = tag_curve["ci_low_bp"].to_list()
        ci_high = tag_curve["ci_high_bp"].to_list()
        mean_values = tag_curve["mean_bp"].to_list()
        customdata = list(
            zip(
                ci_low,
                ci_high,
                tag_curve["trade_count"].to_list(),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=ci_low,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=tag,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=ci_high,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.14),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=tag,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=mean_values,
                mode="lines+markers",
                name=tag,
                legendgroup=tag,
                line={"color": color, "width": 3},
                marker={"size": 7, "color": color},
                customdata=customdata,
                hovertemplate=(
                    "%{fullData.name}<br>Horizon: %{x:.0f}s"
                    "<br>Mean: %{y:.2f} bp"
                    "<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                    "<br>Trades: %{customdata[2]:,}<extra></extra>"
                ),
            )
        )

    figure.add_hline(y=0.0, line_dash="dash", line_color="#888888", line_width=1)
    figure.update_layout(
        template="plotly_white",
        title={"text": "Mean markout (bp) vs horizon — LP perspective", "font": {"size": 16}},
        height=480,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 56, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    figure.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=[1, 5, 30, 60, 300],
        ticktext=["1s", "5s", "30s", "60s", "300s"],
        showgrid=False,
        title="Horizon (log scale)",
    )
    figure.update_yaxes(title="Markout (bp)", gridcolor="#E0E0E0", zeroline=False)
    return figure


def build_markout_box(
    distribution_df: pl.DataFrame,
    horizon_seconds: float,
    selected_tags: Sequence[str],
) -> go.Figure:
    stats = (
        distribution_df.filter(pl.col("tag").is_in(selected_tags))
        .group_by("tag")
        .agg(
            [
                pl.col("markout_bp").median().alias("median_bp"),
                pl.col("markout_bp").quantile(0.25).alias("q1_bp"),
                pl.col("markout_bp").quantile(0.75).alias("q3_bp"),
                pl.col("markout_bp").quantile(0.05).alias("p5_bp"),
                pl.col("markout_bp").quantile(0.95).alias("p95_bp"),
            ]
        )
        .sort("median_bp")
    )
    ordered_tags = stats["tag"].to_list()
    figure = go.Figure()

    for tag in ordered_tags:
        tag_distribution = distribution_df.filter(pl.col("tag") == tag)
        if tag_distribution.is_empty():
            continue

        color = TAG_COLORS.get(tag, NET_DARK)
        stat_row = stats.filter(pl.col("tag") == tag).row(0, named=True)
        outliers = tag_distribution.filter(
            (pl.col("markout_bp") < stat_row["p5_bp"]) | (pl.col("markout_bp") > stat_row["p95_bp"])
        )

        figure.add_trace(
            go.Box(
                q1=[stat_row["q1_bp"]],
                median=[stat_row["median_bp"]],
                q3=[stat_row["q3_bp"]],
                lowerfence=[stat_row["p5_bp"]],
                upperfence=[stat_row["p95_bp"]],
                y=[tag],
                orientation="h",
                name=tag,
                legendgroup=tag,
                line={"color": color, "width": 1.6},
                fillcolor=_hex_to_rgba(color, 0.38),
                whiskerwidth=0.6,
                hovertemplate=(
                    "%{y}<br>Q1: %{q1:.2f} bp"
                    "<br>Median: %{median:.2f} bp"
                    "<br>Q3: %{q3:.2f} bp"
                    "<br>Whiskers: [%{lowerfence:.2f}, %{upperfence:.2f}]<extra></extra>"
                ),
                showlegend=False,
            )
        )
        if not outliers.is_empty():
            figure.add_trace(
                go.Scatter(
                    x=outliers["markout_bp"].to_list(),
                    y=[tag] * outliers.height,
                    mode="markers",
                    marker={"color": color, "opacity": 0.3, "size": 3},
                    hovertemplate="%{y}<br>Outlier: %{x:.2f} bp<extra></extra>",
                    showlegend=False,
                    legendgroup=tag,
                )
            )

    figure.add_vline(x=0.0, line_dash="dash", line_color="#888888", line_width=1)
    figure.update_layout(
        template="plotly_white",
        title={"text": f"Markout distribution at {int(horizon_seconds)}s — bp, LP perspective", "font": {"size": 16}},
        height=280,
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        hovermode="closest",
    )
    figure.update_xaxes(title="Markout (bp)", range=[-2, 2], gridcolor="#E0E0E0", zeroline=False)
    figure.update_yaxes(
        title=None,
        showgrid=False,
        categoryorder="array",
        categoryarray=ordered_tags,
        autorange="reversed",
    )
    return figure


def build_tag_pair_heatmap(toxicity_df: pl.DataFrame, horizon_seconds: float) -> go.Figure:
    pair_order = ["EURUSD", "USDJPY", "GBPUSD"]
    sorted_tags = (
        toxicity_df.group_by("tag")
        .agg(pl.col("toxicity_score").mean().alias("mean_toxicity"))
        .sort("mean_toxicity", descending=True)["tag"]
        .to_list()
    )
    pair_columns = toxicity_df["pair"].unique().to_list()
    pivot = toxicity_df.pivot(on="pair", index="tag", values="toxicity_score").with_columns(
        [pl.col(pair).fill_null(0.0) for pair in pair_order if pair in pair_columns]
    )

    z_values = []
    for tag in sorted_tags:
        tag_row = pivot.filter(pl.col("tag") == tag)
        z_values.append([float(tag_row[pair][0]) if pair in tag_row.columns else 0.0 for pair in pair_order])

    figure = go.Figure(
        data=
        [
            go.Heatmap(
                z=z_values,
                x=pair_order,
                y=sorted_tags,
                colorscale="Reds",
                zmin=-0.1,
                zmax=1.0,
                colorbar={"title": "Score"},
                hovertemplate="Tag: %{y}<br>Pair: %{x}<br>Toxicity: %{z:+.2f}<extra></extra>",
            )
        ]
    )

    for row_idx, tag in enumerate(sorted_tags):
        for col_idx, pair in enumerate(pair_order):
            value = z_values[row_idx][col_idx]
            figure.add_annotation(
                x=pair,
                y=tag,
                text=f"{value:+.2f}",
                showarrow=False,
                font={"color": "white" if value >= 0.45 else "#1F2937", "size": 12},
            )

    figure.update_layout(
        template="plotly_white",
        title={"text": f"Tag × pair toxicity score at {int(horizon_seconds)}s", "font": {"size": 16}},
        height=360,
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
    )
    figure.update_xaxes(title=None, side="top", showgrid=False)
    figure.update_yaxes(title=None, autorange="reversed", showgrid=False)
    return figure


def build_pnl_waterfall(totals: Mapping[str, float]) -> go.Figure:
    spread_capture = totals["spread_capture_usd"]
    as_drag = totals["adverse_selection_usd"]
    hedge_cost = totals["hedge_cost_usd"]
    net_pnl = totals["net_pnl_usd"]
    y_values = [spread_capture, -as_drag, -hedge_cost, net_pnl]
    labels = ["Spread capture", "AS drag", "Hedge cost", "Net PnL"]
    text_values = [fmt_usd_signed(value) for value in y_values[:-1]] + [fmt_usd_signed(net_pnl)]

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=labels,
            y=y_values,
            text=text_values,
            textposition="outside",
            connector={"line": {"color": "#888888", "dash": "dot"}},
            increasing={"marker": {"color": POSITIVE_GREEN}},
            decreasing={"marker": {"color": NEGATIVE_RED}},
            totals={"marker": {"color": "#1A2B40"}},
            hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": "Total PnL attribution", "font": {"size": 16}},
        height=380,
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
    )
    figure.update_yaxes(tickformat="$,.0f", gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#888888")
    figure.update_xaxes(showgrid=False)
    return figure


def build_pnl_bars_by_category(summary_df: pl.DataFrame, category_col: str, title: str) -> go.Figure:
    ordered = summary_df.sort("pnl_usd", descending=True)
    categories = ordered[category_col].to_list()
    pnl_values = ordered["pnl_usd"].to_list()
    colors = [POSITIVE_GREEN if value >= 0 else NEGATIVE_RED for value in pnl_values]

    figure = go.Figure(
        go.Bar(
            y=categories,
            x=pnl_values,
            orientation="h",
            marker_color=colors,
            text=[fmt_usd_signed(value) for value in pnl_values],
            textposition="auto",
            hovertemplate="%{y}<br>PnL: %{x:$,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 16}},
        height=340,
        margin={"l": 20, "r": 20, "t": 56, "b": 20},
    )
    figure.update_xaxes(tickformat="$,.0f", gridcolor="#E0E0E0")
    figure.update_yaxes(title=None, categoryorder="array", categoryarray=categories[::-1])
    return figure


def build_pnl_composition_stacked(summary_df: pl.DataFrame) -> go.Figure:
    ordered = summary_df.sort("pnl_usd", descending=True)
    categories = ordered["tag"].to_list()
    spread_capture = ordered["spread_capture_usd"].to_list()
    as_drag = [-value for value in ordered["as_usd"].to_list()]
    hedge_cost = [-value for value in ordered["hedge_cost_usd"].to_list()]
    net_pnl = ordered["pnl_usd"].to_list()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=categories,
            x=spread_capture,
            orientation="h",
            name="Spread capture",
            marker_color=CAPTURE_GREEN,
            hovertemplate="%{y}<br>Spread capture: %{x:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=categories,
            x=as_drag,
            orientation="h",
            name="AS drag",
            marker_color=ADVERSE_RED,
            hovertemplate="%{y}<br>AS drag: %{x:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=categories,
            x=hedge_cost,
            orientation="h",
            name="Hedge cost",
            marker_color=HEDGE_BLUE,
            hovertemplate="%{y}<br>Hedge cost: %{x:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=net_pnl,
            y=categories,
            mode="markers",
            name="Net PnL",
            marker={"symbol": "diamond", "size": 9, "color": "#111111"},
            hovertemplate="%{y}<br>Net PnL: %{x:$,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": "PnL composition by tag", "font": {"size": 16}},
        height=420,
        barmode="relative",
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    figure.update_xaxes(tickformat="$,.0f", gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#888888")
    figure.update_yaxes(title=None, categoryorder="array", categoryarray=categories[::-1])
    return figure


def build_pnl_by_hour(hourly_df: pl.DataFrame) -> go.Figure:
    x_values = hourly_df["hour_start"].to_list()
    spread_capture = hourly_df["spread_capture_usd"].to_list()
    as_drag = [-value for value in hourly_df["adverse_selection_usd"].to_list()]
    hedge_cost = [-value for value in hourly_df["hedge_cost_usd"].to_list()]
    net_pnl = hourly_df["net_pnl_usd"].to_list()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=x_values,
            y=spread_capture,
            name="Spread capture",
            marker_color=CAPTURE_GREEN,
            hovertemplate="%{x|%H:%M}<br>Spread capture: %{y:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=x_values,
            y=as_drag,
            name="AS drag",
            marker_color=ADVERSE_RED,
            hovertemplate="%{x|%H:%M}<br>AS drag: %{y:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=x_values,
            y=hedge_cost,
            name="Hedge cost",
            marker_color=HEDGE_BLUE,
            hovertemplate="%{x|%H:%M}<br>Hedge cost: %{y:$,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=net_pnl,
            mode="lines+markers",
            name="Net PnL",
            line={"color": "#1A2B40", "width": 3},
            marker={"size": 7, "color": "#1A2B40"},
            hovertemplate="%{x|%H:%M}<br>Net PnL: %{y:$,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": "PnL by hour, by source", "font": {"size": 16}},
        height=360,
        barmode="relative",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 56, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    figure.update_yaxes(tickformat="$,.0f", gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#888888")
    figure.update_xaxes(showgrid=False, tickformat="%H:%M")
    return figure


def build_pnl_delta_by_tag(delta_df: pl.DataFrame) -> go.Figure:
    ordered = delta_df.with_columns(pl.col("pnl_delta_usd").abs().alias("abs_delta")).sort("abs_delta", descending=True)
    categories = ordered["tag"].to_list()
    deltas = ordered["pnl_delta_usd"].to_list()
    colors = [
        "#CCCCCC" if abs(value) < 50.0 else POSITIVE_GREEN if value > 0 else NEGATIVE_RED
        for value in deltas
    ]
    has_material_change = any(abs(value) >= 50.0 for value in deltas)

    figure = go.Figure()
    if has_material_change:
        figure.add_trace(
            go.Bar(
                y=categories,
                x=deltas,
                orientation="h",
                marker_color=colors,
                text=[fmt_usd_signed(value) for value in deltas],
                textposition="auto",
                hovertemplate="%{y}<br>PnL delta: %{x:$+,.0f}<extra></extra>",
                showlegend=False,
            )
        )
    else:
        figure.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Adjust controls above to see projected impact.",
            showarrow=False,
            font={"color": "#6B7280", "size": 14},
        )

    figure.add_vline(x=0.0, line_dash="dash", line_color="#888888", line_width=1)
    figure.update_layout(
        template="plotly_white",
        title={"text": "PnL change by tag (projected − current)", "font": {"size": 16}},
        height=340,
        margin={"l": 20, "r": 20, "t": 56, "b": 20},
    )
    if has_material_change:
        figure.update_xaxes(tickformat="$+,.0f", gridcolor="#E0E0E0")
        figure.update_yaxes(title=None, categoryorder="array", categoryarray=categories[::-1])
    else:
        figure.update_xaxes(range=[-1.0, 1.0], showticklabels=False, gridcolor="#E0E0E0")
        figure.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return figure
