"""
Page 3 — Event log.

Paginated, filterable table of every firewall event from GET /events. Raw
prompts are never stored — only hashes and redacted snippets. Supports search,
decision/score filters, CSV export, and a request-ID drill-down.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import api_client
from utils.formatters import VERDICT_COLORS, fmt_rules


def _decision_bg(decision: str) -> str:
    color = VERDICT_COLORS.get(decision, "#888888")
    return f"background-color: {color}33; font-weight: 600;"  # 33 = ~20% alpha


def render() -> None:
    st.markdown("## 📋 Event log")
    st.caption("Every firewall decision. Raw prompts are never stored — hashes and redacted snippets only.")

    # ── filters ──────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([2, 1, 1.4, 1])
    with f1:
        search = st.text_input("Search", placeholder="Search redacted prompt…",
                               label_visibility="collapsed")
    with f2:
        decision_filter = st.selectbox("Decision", ["All", "Allow", "Sanitize", "Block"],
                                       label_visibility="collapsed")
    with f3:
        min_score = st.slider("Min risk score", 0.0, 1.0, 0.0, 0.05)
    with f4:
        limit = st.selectbox("Rows", [25, 50, 100, 200, 500], index=1,
                             label_visibility="collapsed")

    st.divider()

    # ── fetch ────────────────────────────────────────────────────────────────
    server_filter = None if decision_filter == "All" else decision_filter.upper()
    try:
        events = api_client.get_events(limit=limit, decision=server_filter)
    except api_client.FirewallAPIError as exc:
        st.error(str(exc))
        st.caption("Showing mock data so the layout stays previewable.")
        events = api_client.mock_events(limit)

    if not events:
        st.info("No events recorded yet.")
        return

    # normalize into a flat frame
    df = pd.DataFrame([{
        "Time": ev.get("timestamp", ""),
        "Request ID": ev.get("request_id", ""),
        "Prompt hash": ev.get("prompt_hash", ""),
        "Redacted prompt": ev.get("redacted_prompt") or "—",
        "Decision": ev.get("policy_decision", ev.get("action", "")),
        "Risk score": round(float(ev.get("injection_risk_score", ev.get("score", 0)) or 0), 3),
        "Triggered rules": fmt_rules(ev.get("triggered_rules")),
        "Latency (ms)": round(api_client.total_latency_ms(ev.get("latency_ms", 0))),
        "Output": ev.get("output_status") or "—",
    } for ev in events])

    # ── client-side filters ──────────────────────────────────────────────────
    if search:
        df = df[df["Redacted prompt"].str.contains(search, case=False, na=False)]
    if decision_filter != "All":
        df = df[df["Decision"] == decision_filter.upper()]
    if min_score > 0:
        df = df[df["Risk score"] >= min_score]

    # ── summary badges ───────────────────────────────────────────────────────
    counts = df["Decision"].value_counts()
    b0, b1, b2, b3 = st.columns(4)
    b0.markdown(f"**{len(df)}** events")
    for col, key, glyph in ((b1, "ALLOW", "allowed"), (b2, "SANITIZE", "sanitized"), (b3, "BLOCK", "blocked")):
        col.markdown(
            f'<span style="background:{VERDICT_COLORS[key]};color:#fff;'
            f'padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;">'
            f'{int(counts.get(key, 0))} {glyph}</span>',
            unsafe_allow_html=True,
        )

    st.write("")

    if df.empty:
        st.info("No events match your filters.")
        return

    # ── colored table ────────────────────────────────────────────────────────
    display = df.drop(columns=["Request ID", "Prompt hash"])
    styled = (
        display.style
        .map(lambda v: _decision_bg(v) if v in VERDICT_COLORS else "", subset=["Decision"])
        .format({"Risk score": "{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)

    st.download_button(
        "⬇️ Export as CSV",
        df.to_csv(index=False),
        file_name="firewall_events.csv",
        mime="text/csv",
    )

    st.divider()

    # ── drill-down ───────────────────────────────────────────────────────────
    st.markdown("**Drill-down — inspect an event**")
    req_id = st.text_input("Request ID", placeholder="req-1042 / uuid…",
                           label_visibility="collapsed")
    if req_id:
        match = df[df["Request ID"] == req_id]
        if match.empty:
            st.warning(f"No event found with ID `{req_id}`.")
        else:
            row = match.iloc[0]
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**Request ID:** `{row['Request ID']}`")
                st.markdown(f"**Timestamp:** {row['Time']}")
                st.markdown(f"**Decision:** {row['Decision']}")
                st.markdown(f"**Risk score:** {row['Risk score']:.3f}")
            with cb:
                st.markdown(f"**Prompt hash:** `{row['Prompt hash']}`")
                st.markdown(f"**Rules triggered:** {row['Triggered rules']}")
                st.markdown(f"**Latency:** {row['Latency (ms)']} ms")
                st.markdown(f"**Output status:** {row['Output']}")
            st.markdown(f"**Redacted prompt:** {row['Redacted prompt']}")
