"""
In-app Light/Dark theme support.

Streamlit 1.58 doesn't expose a runtime theme switcher in the deployed menu, and
st.dataframe's grid can't be re-themed by config at runtime — so we drive theming
ourselves: a sidebar toggle sets ``st.session_state.dark_mode``; this module then
provides the dark CSS skin, Plotly layout colors, and a dataframe Styler so the
chrome, charts, and tables all follow the toggle consistently.
"""

from __future__ import annotations

import streamlit as st

# palette
DARK_BG = "#0e1117"
DARK_PANEL = "#161b22"
DARK_INPUT = "#1c2330"
DARK_TEXT = "#e6edf3"
DARK_GRID = "rgba(255,255,255,.12)"
LIGHT_TEXT = "#1a1a1a"
LIGHT_GRID = "rgba(128,128,128,.20)"


def is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def plotly_colors() -> dict:
    """Font + gridline colors for charts, following the active theme."""
    dark = is_dark()
    return {
        "font": DARK_TEXT if dark else LIGHT_TEXT,
        "grid": DARK_GRID if dark else LIGHT_GRID,
    }


def style_plotly(fig):
    """Apply theme-aware fonts/gridlines to a Plotly figure (transparent bg)."""
    c = plotly_colors()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=c["font"],
    )
    fig.update_xaxes(gridcolor=c["grid"], zerolinecolor=c["grid"])
    fig.update_yaxes(gridcolor=c["grid"], zerolinecolor=c["grid"])
    return fig


def style_table(styler):
    """Give a pandas Styler dark cell bg/text when dark mode is on."""
    if is_dark():
        styler = styler.set_properties(**{
            "background-color": DARK_PANEL,
            "color": DARK_TEXT,
        })
    return styler


def dark_css() -> str:
    """Dark skin injected over style.css when the toggle is on."""
    return f"""
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {DARK_BG} !important;
    }}
    .stApp, .stApp p, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {{
        color: {DARK_TEXT};
    }}
    h1, h2, h3, h4, h5 {{ color: #f0f3f6 !important; }}
    section[data-testid="stSidebar"] {{
        background-color: {DARK_PANEL} !important;
        border-right-color: rgba(255,255,255,.08) !important;
    }}
    .status-card, div[data-testid="stMetric"] {{
        background: rgba(255,255,255,.04) !important;
        border-color: rgba(255,255,255,.10) !important;
    }}
    .stTextInput input, .stNumberInput input, textarea,
    div[data-baseweb="select"] > div, div[data-baseweb="input"] {{
        background-color: {DARK_INPUT} !important;
        color: {DARK_TEXT} !important;
        border-color: rgba(255,255,255,.14) !important;
    }}
    [data-baseweb="popover"], [role="listbox"], [data-baseweb="menu"] {{
        background-color: {DARK_INPUT} !important;
    }}
    [data-testid="stChatMessage"] {{
        background: {DARK_PANEL} !important;
        border-color: rgba(255,255,255,.08) !important;
    }}
    .stButton > button {{
        background: {DARK_INPUT} !important;
        color: {DARK_TEXT} !important;
        border-color: rgba(255,255,255,.14) !important;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background: {DARK_PANEL} !important;
        border-color: rgba(255,255,255,.10) !important;
    }}
    code {{ background: rgba(255,255,255,.10) !important; color: {DARK_TEXT} !important; }}
    [data-testid="stMetricValue"] {{ color: #ffffff !important; }}
    """
