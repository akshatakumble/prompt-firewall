"""
Presentation helpers: verdict badges, score bars, decision colors, and a few
small formatting utilities shared across pages.

Color policy: the only hardcoded hex values are the three verdict colors (which
must stay constant for security legibility across light/dark themes). Everything
else inherits from Streamlit's active theme.
"""

from __future__ import annotations

# ── canonical verdict palette ────────────────────────────────────────────────
# (kept constant across themes — operators learn these colors)
VERDICT_COLORS = {
    "ALLOW":    "#639922",   # green
    "SANITIZE": "#EF9F27",   # amber
    "BLOCK":    "#E24B4A",   # red
}

VERDICT_ICONS = {
    "ALLOW": "✓",
    "SANITIZE": "⚠",
    "BLOCK": "✗",
}

# Streamlit status-box helper to use per verdict (st.success/warning/error)
VERDICT_BOX = {
    "ALLOW": "success",
    "SANITIZE": "warning",
    "BLOCK": "error",
}


def normalize_decision(action: str | None) -> str:
    a = (action or "ALLOW").upper()
    return a if a in VERDICT_COLORS else "ALLOW"


def verdict_color(action: str | None) -> str:
    return VERDICT_COLORS[normalize_decision(action)]


def verdict_icon(action: str | None) -> str:
    return VERDICT_ICONS[normalize_decision(action)]


def verdict_badge(action: str | None) -> str:
    """Inline HTML pill for a verdict, e.g. in tables and headers."""
    decision = normalize_decision(action)
    color = VERDICT_COLORS[decision]
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:999px;font-size:12px;font-weight:600;'
        f'white-space:nowrap;">{VERDICT_ICONS[decision]} {decision}</span>'
    )


def rule_pills(rules: list[str] | None) -> str:
    """Render triggered rules as small pills (theme-neutral)."""
    if not rules:
        return '<span style="font-size:12px;opacity:0.55;">no rules triggered</span>'
    pills = "".join(
        '<span style="display:inline-block;border:1px solid rgba(128,128,128,0.4);'
        'border-radius:999px;padding:1px 9px;font-size:11px;margin:2px 4px 0 0;'
        f'opacity:0.85;">{r}</span>'
        for r in rules
    )
    return pills


def score_bar(score: float) -> str:
    """
    Horizontal risk-score bar (0–1). Green < 0.45, amber < 0.70, red otherwise.
    Track uses a translucent neutral so it works on light and dark backgrounds.
    """
    score = max(0.0, min(1.0, float(score)))
    pct = int(round(score * 100))
    if score < 0.45:
        fill = VERDICT_COLORS["ALLOW"]
    elif score < 0.70:
        fill = VERDICT_COLORS["SANITIZE"]
    else:
        fill = VERDICT_COLORS["BLOCK"]
    return (
        '<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
        '<div style="flex:1;height:6px;background:rgba(128,128,128,0.25);'
        'border-radius:999px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;background:{fill};'
        'border-radius:999px;"></div></div>'
        f'<span style="font-size:12px;font-weight:600;opacity:0.85;">{score:.2f}</span>'
        '</div>'
    )


def fmt_rules(rules: list[str] | None) -> str:
    """Plain-text join for dataframe cells."""
    return ", ".join(rules) if rules else "—"


def delta_pct(part: int, whole: int) -> str:
    """Format a share-of-total as a signed-looking percentage string."""
    if not whole:
        return "0.0%"
    return f"{part / whole * 100:.1f}%"
