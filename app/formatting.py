from __future__ import annotations


def fmt_count(value: int | float) -> str:
    return f"{int(round(value)):,}"


def fmt_usd_compact(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}bn"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}mm"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:,.0f}"


def fmt_usd_signed(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}"


def fmt_usd_per_million(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def fmt_bp(value: float) -> str:
    return f"{value:.2f} bp"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"
