"""Shared visual styles for all dashboard pages.

Streamlit's native theme config only supports preset font families
(sans-serif / serif / monospace). For Inter specifically we inject
a Google Fonts import + CSS override on every page via
``apply_global_styles``.
"""

from __future__ import annotations

import streamlit as st

_INTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body,
[class*="css"],
.stMarkdown, .stMetric, .stDataFrame, .stButton,
h1, h2, h3, h4, h5, h6,
p, span, div, label, input, button, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

h1, h2, h3 {
    letter-spacing: -0.015em !important;
}
</style>
"""


def apply_global_styles() -> None:
    """Inject the global font + style overrides for a dashboard page."""
    st.markdown(_INTER_CSS, unsafe_allow_html=True)
