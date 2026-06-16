"""
Page 4 — Settings.

Read-only view of the backend configuration (health endpoint) plus client-side
dashboard preferences. The firewall's thresholds and policy live server-side;
this page surfaces them and the connection details for operators.
"""

from __future__ import annotations

import os

import streamlit as st

from utils import api_client


def render() -> None:
    st.markdown("## ⚙️ Settings")

    # ── connection ───────────────────────────────────────────────────────────
    st.markdown("### Connection")
    st.caption("Set via the `FIREWALL_API_URL` environment variable before launch.")
    st.code(f"FIREWALL_API_URL = {os.getenv('FIREWALL_API_URL', api_client.API_URL)}",
            language="bash")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔌 Test connection", use_container_width=True):
            st.session_state["_settings_health"] = None  # force refetch below

    # ── live service info ────────────────────────────────────────────────────
    st.markdown("### Service status")
    try:
        health = api_client.get_health()
        online = str(health.get("status", "")).lower() == "ok"
    except api_client.FirewallAPIError as exc:
        st.error(str(exc))
        health, online = api_client.mock_health(), False

    if online:
        st.success("Firewall API is online and responding.")
    else:
        st.warning("Firewall API is offline — dashboard is running on mock data.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model version", health.get("model_version", "—"))
    c2.metric("Classifier", "loaded" if health.get("classifier_loaded") else "not loaded")
    c3.metric("LLM provider", health.get("llm_provider", "—"))
    st.caption(f"LLM model: `{health.get('llm_model', '—')}`")

    st.divider()

    # ── dashboard preferences ────────────────────────────────────────────────
    st.markdown("### Dashboard preferences")
    st.session_state.auto_refresh = st.toggle(
        "Auto-refresh analytics every 30s",
        value=st.session_state.get("auto_refresh", True),
        help="Applies to the Analytics page while it is open.",
    )
    st.caption("Theme (light/dark) follows your Streamlit setting — change it from the ☰ menu → Settings.")

    st.divider()

    # ── session reset ────────────────────────────────────────────────────────
    st.markdown("### Session")
    counts = st.session_state.get("session_counts", {})
    st.write(f"Conversation turns: **{len(st.session_state.get('messages', []))}**")
    st.write(
        f"Decisions this session — "
        f"ALLOW: **{counts.get('ALLOW', 0)}**, "
        f"SANITIZE: **{counts.get('SANITIZE', 0)}**, "
        f"BLOCK: **{counts.get('BLOCK', 0)}**"
    )
    if st.button("🗑️ Reset session state"):
        st.session_state.messages = []
        st.session_state.session_counts = {"ALLOW": 0, "SANITIZE": 0, "BLOCK": 0}
        st.success("Session reset.")
