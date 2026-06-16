"""
Prompt Firewall — dashboard entrypoint.

Run with:
    streamlit run src/dashboard/app.py

Config:
    FIREWALL_API_URL   FastAPI backend base URL (default http://localhost:8000)

Four pages, navigated from the sidebar: Chat, Analytics, Event log, Settings.
The sidebar also shows live service health, model/LLM metadata, and a running
count of this session's firewall decisions.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from utils import api_client
from utils.formatters import VERDICT_COLORS

st.set_page_config(
    page_title="Prompt Firewall",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── one-time setup ───────────────────────────────────────────────────────────

def _inject_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault(
        "session_counts", {"ALLOW": 0, "SANITIZE": 0, "BLOCK": 0}
    )
    st.session_state.setdefault("analytics_cache", None)
    st.session_state.setdefault("analytics_last_refresh", 0.0)
    st.session_state.setdefault("auto_refresh", True)


def record_decision(action: str) -> None:
    """Increment this session's verdict counter (called by the chat page)."""
    a = (action or "ALLOW").upper()
    if a in st.session_state.session_counts:
        st.session_state.session_counts[a] += 1


# ── sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🛡️ Prompt Firewall")
        st.caption("Jailbreak & prompt-injection gateway")

        page = st.radio(
            "Navigate",
            ["💬 Chat", "📊 Analytics", "📋 Event log", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.divider()

        # live service status
        try:
            health = api_client.get_health()
            online = str(health.get("status", "")).lower() == "ok"
        except api_client.FirewallAPIError:
            health, online = api_client.mock_health(), False

        dot = "dot-online" if online else "dot-offline"
        label = "online" if online else "offline"
        st.markdown(
            f'<div class="status-line">'
            f'<span class="dot {dot}"></span><b>Firewall API</b> · {label}<br>'
            f'<b>Model</b> · {health.get("model_version", "—")}<br>'
            f'<b>Classifier</b> · {"loaded" if health.get("classifier_loaded") else "not loaded"}<br>'
            f'<b>LLM</b> · {health.get("llm_provider", "—")} · {health.get("llm_model", "—")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # session decision summary
        counts = st.session_state.session_counts
        total = sum(counts.values())
        st.markdown(f"**Session requests:** {total}")
        c1, c2, c3 = st.columns(3)
        for col, key, glyph in (
            (c1, "ALLOW", "✓"), (c2, "SANITIZE", "⚠"), (c3, "BLOCK", "✗")
        ):
            col.markdown(
                f'<span style="background:{VERDICT_COLORS[key]};color:#fff;'
                f'padding:3px 9px;border-radius:999px;font-size:12px;'
                f'font-weight:600;">{counts[key]} {glyph}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(f"API: `{os.getenv('FIREWALL_API_URL', api_client.API_URL)}`")

    return page


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _init_state()
    page = _render_sidebar()

    if "Chat" in page:
        from views import chat
        chat.render()
    elif "Analytics" in page:
        from views import analytics
        analytics.render()
    elif "Event log" in page:
        from views import event_log
        event_log.render()
    elif "Settings" in page:
        from views import settings
        settings.render()


main()
