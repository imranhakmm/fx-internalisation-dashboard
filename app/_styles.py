"""Shared visual styles for all dashboard pages.

Streamlit's native theme config only supports preset font families
(sans-serif / serif / monospace). For Inter specifically we inject
a Google Fonts import + CSS override on every page via
``apply_global_styles``.

Important: the override must NOT touch Streamlit's icon spans, which
use Material Symbols. Forcing Inter onto those elements causes icon
names (e.g. "keyboard_double_arrow_left", "arrow_right") to render
as raw text in place of the glyphs.
"""

from __future__ import annotations

import streamlit as st

_INTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Apply Inter to text elements only — leave icon containers alone. */
html, body,
.stApp, .stApp p, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stMetric, .stMetric label, .stMetric div,
.stDataFrame,
.stButton button,
.stSelectbox, .stSlider, .stTextInput, .stNumberInput, .stTextArea,
.stCaption, .stTabs button,
[data-testid="stSidebarNav"] a {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Slightly tighter headings — Inter benefits from negative letter-spacing. */
.stApp h1, .stApp h2, .stApp h3 {
    letter-spacing: -0.015em;
}

/* CRITICAL: explicitly preserve Material Symbols on Streamlit's icon spans.
   Without this, the Inter override above leaks into icon containers and the
   icon names render as text (e.g. "keyboard_double_arrow_left"). */
[data-testid="stIconMaterial"],
[data-testid="stIcon"],
span.material-symbols-outlined,
span.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-rounded {
    font-family: 'Material Symbols Outlined', 'Material Symbols Rounded' !important;
}
</style>
"""


def apply_global_styles() -> None:
    """Inject the global font + style overrides. Call once per page,
    immediately after ``st.set_page_config``."""
    st.markdown(_INTER_CSS, unsafe_allow_html=True)
