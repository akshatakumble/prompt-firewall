"""
Page 2 — Analytics dashboard.

Layout: metric cards row → two-column charts (decision donut + top rules) →
hourly attack volume → near-misses panel → recent event log table.

The backend's /analytics endpoint returns aggregate counts only, so the hourly
volume and near-miss detail panels are derived client-side from /events.
Auto-refreshes every 30s while the page is open (toggleable).
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import api_client
from utils.formatters import VERDICT_COLORS, delta_pct, fmt_rules

REFRESH_SECS = 30


# ── charts ───────────────────────────────────────────────────────────────────

def _donut(decisions: dict[str, int]) -> go.Figure:
    labels = ["Allowed", "Sanitized", "Blocked"]
    values = [decisions["ALLOW"], decisions["SANITIZE"], decisions["BLOCK"]]
    colors = [VERDICT_COLORS["ALLOW"], VERDICT_COLORS["SANITIZE"], VERDICT_COLORS["BLOCK"]]
    total = sum(values)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker_colors=colors, textinfo="none",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=0, r=0, t=10, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{total:,}</b><br>requests",
                          x=0.5, y=0.5, showarrow=False, font_size=14)],
    )
    return fig


def _rules_chart(top_rules: list[dict]) -> go.Figure:
    if not top_rules:
        return go.Figure()
    df = pd.DataFrame(top_rules).sort_values("count")
    # block-associated heuristic: high count / known block rules in red, else amber
    block_markers = ("override", "extract", "DAN", "jailbreak", "ignore")
    colors = [
        VERDICT_COLORS["BLOCK"] if any(m in r.lower() for m in block_markers)
        else VERDICT_COLORS["SANITIZE"]
        for r in df["rule"]
    ]
    fig = go.Figure(go.Bar(
        x=df["count"], y=df["rule"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _hourly_chart(hourly: list[dict]) -> go.Figure:
    if not hourly:
        return go.Figure()
    df = pd.DataFrame(hourly)
    fig = go.Figure()
    fig.add_bar(name="Allowed", x=df["hour"], y=df["allowed"],
                marker_color=VERDICT_COLORS["ALLOW"])
    fig.add_bar(name="Sanitized", x=df["hour"], y=df["sanitized"],
                marker_color=VERDICT_COLORS["SANITIZE"])
    fig.add_bar(name="Blocked", x=df["hour"], y=df["blocked"],
                marker_color=VERDICT_COLORS["BLOCK"])
    fig.update_layout(
        barmode="stack", height=300,
        margin=dict(l=0, r=0, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


# ── render ───────────────────────────────────────────────────────────────────

def render() -> None:
    header, ctrl = st.columns([4, 1.4])
    with header:
        st.markdown("## 📊 Analytics dashboard")
    with ctrl:
        st.session_state.auto_refresh = st.toggle(
            "Auto-refresh (30s)", value=st.session_state.get("auto_refresh", True)
        )

    # fetch — analytics + events; fall back to mock on failure
    offline = False
    try:
        analytics = api_client.get_analytics()
        events = api_client.get_events(limit=200)
    except api_client.FirewallAPIError as exc:
        st.error(str(exc))
        st.caption("Showing mock data so the layout stays previewable.")
        analytics = api_client.mock_analytics()
        events = api_client.mock_events(120)
        offline = True

    decisions = api_client.normalize_decisions(analytics)
    top_rules = api_client.normalize_top_rules(analytics)
    total = analytics.get("total_requests", sum(decisions.values())) or 0
    hourly = api_client.derive_hourly(events)
    near_misses = api_client.derive_near_misses(events)

    st.session_state.analytics_last_refresh = time.time()
    st.divider()

    # ── metric cards ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total requests", f"{total:,}")
    m2.metric("Blocked", f"{decisions['BLOCK']:,}",
              delta_pct(decisions["BLOCK"], total), delta_color="inverse")
    m3.metric("Sanitized", f"{decisions['SANITIZE']:,}",
              delta_pct(decisions["SANITIZE"], total), delta_color="off")
    m4.metric("Allowed", f"{decisions['ALLOW']:,}",
              delta_pct(decisions["ALLOW"], total), delta_color="normal")
    m5.metric("Avg risk score", f"{analytics.get('avg_risk_score', 0):.3f}")

    st.write("")

    # ── charts row ───────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**Decision distribution**")
        st.plotly_chart(_donut(decisions), use_container_width=True,
                        theme="streamlit", config={"displayModeBar": False})
    with right:
        st.markdown("**Top rule triggers (last 24h)**")
        if top_rules:
            st.plotly_chart(_rules_chart(top_rules), use_container_width=True,
                            theme="streamlit", config={"displayModeBar": False})
        else:
            st.info("No rules triggered yet.")

    st.divider()

    # ── hourly attack volume ─────────────────────────────────────────────────
    st.markdown("**Hourly request volume** — stacked by decision")
    if hourly:
        st.plotly_chart(_hourly_chart(hourly), use_container_width=True,
                        theme="streamlit", config={"displayModeBar": False})
    else:
        st.info("Not enough event history to build an hourly breakdown yet.")

    st.divider()

    # ── near-misses panel ────────────────────────────────────────────────────
    st.markdown("**Near-misses — high-risk prompts that were allowed**")
    st.caption("Allowed prompts scoring 0.45–0.55. Your most valuable regression cases.")
    if near_misses:
        for nm in near_misses[:8]:
            st.markdown(
                f'<div class="near-miss">'
                f'<span class="near-miss-text">"{nm["redacted_snippet"]}"<br>'
                f'<span class="near-miss-hash">{nm["prompt_hash"]} · {nm["timestamp"]}</span></span>'
                f'<span class="near-miss-score">score {nm["score"]:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No near-misses in this window. 🎉")

    st.divider()

    # ── recent events table ──────────────────────────────────────────────────
    st.markdown("**Recent events**")
    _events_table(events)

    if offline:
        st.caption("⚠️ API offline — figures above are mock data.")

    # ── auto-refresh ─────────────────────────────────────────────────────────
    if st.session_state.get("auto_refresh"):
        time.sleep(REFRESH_SECS)
        st.rerun()


def _events_table(events: list[dict]) -> None:
    if not events:
        st.info("No events to display.")
        return
    rows = []
    for ev in events[:50]:
        rows.append({
            "Timestamp": ev.get("timestamp", ""),
            "Redacted prompt": ev.get("redacted_prompt") or "—",
            "Decision": ev.get("policy_decision", ev.get("action", "")),
            "Risk score": round(float(ev.get("injection_risk_score", ev.get("score", 0)) or 0), 3),
            "Triggered rules": fmt_rules(ev.get("triggered_rules")),
            "Latency (ms)": round(api_client.total_latency_ms(ev.get("latency_ms", 0))),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True, height=380,
        column_config={
            "Risk score": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Redacted prompt": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "⬇️ Export as CSV",
        df.to_csv(index=False),
        file_name="firewall_events.csv",
        mime="text/csv",
    )
