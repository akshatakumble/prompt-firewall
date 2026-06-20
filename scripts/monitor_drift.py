#!/usr/bin/env python3
"""
Production monitoring: detect data drift and operational decay from the live
firewall's telemetry, and decide whether retraining should be triggered.

What it checks (thresholds in config/monitoring.yaml):
  • Data drift  — Population Stability Index (PSI) on the injection-risk-score
                  distribution, current window vs a stored reference baseline.
  • Decision shift — L1 movement in ALLOW/SANITIZE/BLOCK proportions.
  • Operational  — near-miss rate and average latency regressions.

Data source (pick one):
  --api-url https://...run.app   pull GET /events + GET /analytics (default)
  --events-file events.json      read a list of event dicts (offline/CI demo)

Baseline:
  The first run with no reference writes one to reports/monitoring/reference.json
  and reports "bootstrapped" (no drift). Subsequent runs compare against it.
  Refresh the baseline intentionally with --update-reference.

Outputs:
  reports/monitoring/drift_report.json   full signals + verdict
  GITHUB_OUTPUT: drift=true|false         (when run in GitHub Actions)
  exit code 0 always (monitoring should not fail the job); the verdict is in the
  report and the `drift` output. Use --fail-on-drift to exit 2 instead.

Optional: if `evidently` is installed, a richer HTML report is also written, but
the PSI/decision-shift logic here is self-contained and always runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "monitoring"
REFERENCE_PATH = REPORT_DIR / "reference.json"
REPORT_PATH = REPORT_DIR / "drift_report.json"
DECISIONS = ("ALLOW", "SANITIZE", "BLOCK")


# ── data loading ─────────────────────────────────────────────────────────────

def load_events_from_api(api_url: str, limit: int) -> tuple[list[dict], dict]:
    import requests

    base = api_url.rstrip("/")
    events = requests.get(f"{base}/events", params={"limit": limit}, timeout=15).json()
    if isinstance(events, dict):
        events = events.get("events", [])
    analytics = {}
    try:
        analytics = requests.get(f"{base}/analytics", timeout=15).json()
    except Exception:  # noqa: BLE001 — analytics is supplementary
        pass
    return events, analytics


def load_events_from_file(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("events", []), data.get("analytics", {})
    return data, {}


def _scores(events: list[dict]) -> np.ndarray:
    vals = [float(e.get("injection_risk_score", e.get("score", 0)) or 0) for e in events]
    return np.asarray(vals, dtype=float)


def _decision_props(events: list[dict]) -> dict[str, float]:
    counts = {d: 0 for d in DECISIONS}
    for e in events:
        d = e.get("policy_decision", e.get("action", "ALLOW"))
        if d in counts:
            counts[d] += 1
    total = sum(counts.values()) or 1
    return {d: counts[d] / total for d in DECISIONS}


# ── drift maths ──────────────────────────────────────────────────────────────

def histogram(scores: np.ndarray, bins: list[float]) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(scores, bins=bins)
    return counts.astype(float)


def population_stability_index(ref: np.ndarray, cur: np.ndarray) -> float:
    """PSI = Σ (cur% - ref%) * ln(cur% / ref%), with smoothing to avoid div-by-zero."""
    eps = 1e-6
    ref_pct = ref / (ref.sum() or 1) + eps
    cur_pct = cur / (cur.sum() or 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def decision_l1(ref: dict[str, float], cur: dict[str, float]) -> float:
    return float(sum(abs(cur.get(d, 0) - ref.get(d, 0)) for d in DECISIONS))


def near_miss_rate(events: list[dict], lo: float = 0.45, hi: float = 0.55) -> float:
    if not events:
        return 0.0
    n = sum(
        1 for e in events
        if e.get("policy_decision", e.get("action")) == "ALLOW"
        and lo <= float(e.get("injection_risk_score", e.get("score", 0)) or 0) < hi
    )
    return n / len(events)


def avg_latency_ms(events: list[dict], analytics: dict) -> float:
    al = analytics.get("avg_latency_ms")
    if isinstance(al, dict) and al:
        return float(al.get("total", sum(al.values())))
    lat = []
    for e in events:
        lm = e.get("latency_ms")
        if isinstance(lm, dict):
            lat.append(float(lm.get("total", sum(lm.values()) if lm else 0)))
        elif lm is not None:
            lat.append(float(lm))
    return float(np.mean(lat)) if lat else 0.0


# ── main ─────────────────────────────────────────────────────────────────────

def build_reference(scores: np.ndarray, props: dict, bins: list[float]) -> dict:
    return {
        "score_hist": histogram(scores, bins).tolist(),
        "decision_props": props,
        "n": int(scores.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect drift / decay from firewall telemetry.")
    parser.add_argument("--api-url", default=os.getenv("FIREWALL_API_URL"))
    parser.add_argument("--events-file")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "monitoring.yaml"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--update-reference", action="store_true",
                        help="Overwrite the reference baseline with the current window.")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit 2 when drift is detected (default exits 0).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    drift_cfg, op_cfg = cfg["drift"], cfg["operational"]
    bins = drift_cfg["score_bins"]

    # 1. load current window
    if args.events_file:
        events, analytics = load_events_from_file(Path(args.events_file))
    elif args.api_url:
        try:
            events, analytics = load_events_from_api(args.api_url, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not pull telemetry from {args.api_url}: {exc}", file=sys.stderr)
            return 0  # don't fail the monitoring job on a transient fetch error
    else:
        print("ERROR: provide --api-url or --events-file (or set FIREWALL_API_URL).", file=sys.stderr)
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = _scores(events)
    props = _decision_props(events)

    # 2. bootstrap or refresh reference
    if args.update_reference or not REFERENCE_PATH.exists():
        ref = build_reference(scores, props, bins)
        REFERENCE_PATH.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        report = {
            "status": "bootstrapped" if not args.update_reference else "reference_updated",
            "drift_detected": False,
            "n_events": int(scores.size),
            "message": f"Reference baseline written to {REFERENCE_PATH.relative_to(PROJECT_ROOT)}.",
        }
        _emit(report, drift=False, fail_on_drift=args.fail_on_drift)
        return 0

    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    # 3. low-traffic guard
    if scores.size < drift_cfg["min_events"]:
        report = {
            "status": "insufficient_data",
            "drift_detected": False,
            "n_events": int(scores.size),
            "min_events": drift_cfg["min_events"],
            "message": f"Only {scores.size} events (< {drift_cfg['min_events']}); skipping drift check.",
        }
        _emit(report, drift=False, fail_on_drift=args.fail_on_drift)
        return 0

    # 4. compute signals
    psi = population_stability_index(np.asarray(reference["score_hist"], dtype=float),
                                     histogram(scores, bins))
    dshift = decision_l1(reference.get("decision_props", {}), props)
    nm_rate = near_miss_rate(events)
    latency = avg_latency_ms(events, analytics)

    signals = {
        "psi_score": round(psi, 4),
        "psi_warn": drift_cfg["psi_warn"],
        "psi_alert": drift_cfg["psi_alert"],
        "decision_shift_l1": round(dshift, 4),
        "decision_shift_alert": drift_cfg["decision_shift_alert"],
        "near_miss_rate": round(nm_rate, 4),
        "near_miss_rate_alert": op_cfg["near_miss_rate_alert"],
        "avg_latency_ms": round(latency, 1),
        "latency_ms_alert": op_cfg["latency_ms_alert"],
    }

    reasons = []
    if psi >= drift_cfg["psi_alert"]:
        reasons.append(f"PSI {psi:.3f} >= alert {drift_cfg['psi_alert']} (major score-distribution drift)")
    if dshift >= drift_cfg["decision_shift_alert"]:
        reasons.append(f"decision shift {dshift:.3f} >= {drift_cfg['decision_shift_alert']}")
    if nm_rate >= op_cfg["near_miss_rate_alert"]:
        reasons.append(f"near-miss rate {nm_rate:.3f} >= {op_cfg['near_miss_rate_alert']}")

    warnings = []
    if drift_cfg["psi_warn"] <= psi < drift_cfg["psi_alert"]:
        warnings.append(f"PSI {psi:.3f} in warn band [{drift_cfg['psi_warn']}, {drift_cfg['psi_alert']})")
    if latency >= op_cfg["latency_ms_alert"]:
        warnings.append(f"avg latency {latency:.0f}ms >= {op_cfg['latency_ms_alert']}ms")

    drift_detected = bool(reasons)
    report = {
        "status": "evaluated",
        "drift_detected": drift_detected,
        "n_events": int(scores.size),
        "current_decision_props": {k: round(v, 4) for k, v in props.items()},
        "reference_decision_props": reference.get("decision_props", {}),
        "signals": signals,
        "alert_reasons": reasons,
        "warnings": warnings,
        "recommendation": "TRIGGER_RETRAIN" if drift_detected else "OK",
    }
    _try_evidently(events, reference, bins)
    _emit(report, drift=drift_detected, fail_on_drift=args.fail_on_drift)
    return 2 if (drift_detected and args.fail_on_drift) else 0


def _emit(report: dict, *, drift: bool, fail_on_drift: bool) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"drift={'true' if drift else 'false'}\n")
            fh.write(f"recommendation={report.get('recommendation', 'OK')}\n")


def _try_evidently(events: list[dict], reference: dict, bins: list[float]) -> None:
    """Best-effort richer report if Evidently is available; never fatal."""
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except Exception:  # noqa: BLE001
        return
    try:
        cur = pd.DataFrame({"risk_score": _scores(events)})
        # synthesize a reference sample from the stored histogram
        ref_hist = np.asarray(reference.get("score_hist", []), dtype=float)
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        ref_scores = np.repeat(centers, ref_hist.astype(int)) if ref_hist.size else np.array([])
        ref = pd.DataFrame({"risk_score": ref_scores})
        if not ref.empty and not cur.empty:
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref, current_data=cur)
            rep.save_html(str(REPORT_DIR / "evidently_drift.html"))
            print("Evidently report written to reports/monitoring/evidently_drift.html")
    except Exception as exc:  # noqa: BLE001
        print(f"(evidently report skipped: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
