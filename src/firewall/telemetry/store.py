from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from firewall.config import PROJECT_ROOT, get_settings
from firewall.telemetry.redact import hash_prompt, redact_pii


class Base(DeclarativeBase):
    pass


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    request_id = Column(String(64), nullable=False, index=True)
    prompt_hash = Column(String(64), nullable=False)
    redacted_prompt = Column(Text, nullable=True)
    triggered_rules = Column(Text, nullable=True)
    injection_risk_score = Column(Float, nullable=False)
    policy_decision = Column(String(16), nullable=False)
    sanitization_method = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=False)
    policy_version = Column(String(32), nullable=False)
    latency_ms = Column(Text, nullable=True)
    output_status = Column(String(16), nullable=True)
    output_violation_reasons = Column(Text, nullable=True)
    attack_type = Column(String(64), nullable=True)
    source = Column(String(64), nullable=True)


class TelemetryStore:
    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        url = database_url or settings.database_url
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "")
            Path(PROJECT_ROOT / db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def log_event(self, event: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            row = TelemetryEvent(
                timestamp=datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                if isinstance(event["timestamp"], str)
                else event["timestamp"],
                request_id=event["request_id"],
                prompt_hash=event["prompt_hash"],
                redacted_prompt=event.get("redacted_prompt"),
                triggered_rules=json.dumps(event.get("triggered_rules", [])),
                injection_risk_score=event["injection_risk_score"],
                policy_decision=event["policy_decision"],
                sanitization_method=event.get("sanitization_method"),
                model_version=event["model_version"],
                policy_version=event["policy_version"],
                latency_ms=json.dumps(event.get("latency_ms", {})),
                output_status=event.get("output_status"),
                output_violation_reasons=json.dumps(event.get("output_violation_reasons", [])),
                attack_type=event.get("attack_type"),
                source=event.get("source"),
            )
            session.add(row)
            session.commit()

    def build_event(
        self,
        request_id: str,
        prompt: str,
        triggered_rules: list[str],
        risk_score: float,
        policy_decision: str,
        model_version: str,
        policy_version: str,
        latency_ms: dict[str, float],
        output_status: str | None = None,
        output_violation_reasons: list[str] | None = None,
        sanitization_method: str | None = None,
        log_snippet: bool = True,
        snippet_max_chars: int = 200,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "prompt_hash": hash_prompt(prompt),
            "redacted_prompt": redact_pii(prompt, snippet_max_chars) if log_snippet else None,
            "triggered_rules": triggered_rules,
            "injection_risk_score": risk_score,
            "policy_decision": policy_decision,
            "sanitization_method": sanitization_method,
            "model_version": model_version,
            "policy_version": policy_version,
            "latency_ms": latency_ms,
            "output_status": output_status,
            "output_violation_reasons": output_violation_reasons or [],
        }

    def query_events(self, limit: int = 100, decision: str | None = None) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            q = session.query(TelemetryEvent).order_by(TelemetryEvent.timestamp.desc())
            if decision:
                q = q.filter(TelemetryEvent.policy_decision == decision)
            rows = q.limit(limit).all()
            return [self._row_to_dict(r) for r in rows]

    def analytics_summary(self) -> dict[str, Any]:
        with self.SessionLocal() as session:
            rows = session.query(TelemetryEvent).all()
            if not rows:
                return {
                    "total_requests": 0,
                    "decisions": {},
                    "avg_risk_score": 0.0,
                    "top_rules": [],
                    "near_misses": 0,
                    "avg_latency_ms": {},
                }

            decisions: dict[str, int] = {}
            rule_counts: dict[str, int] = {}
            near_misses = 0
            latencies: dict[str, list[float]] = {}
            risk_scores: list[float] = []

            for row in rows:
                decisions[row.policy_decision] = decisions.get(row.policy_decision, 0) + 1
                risk_scores.append(row.injection_risk_score)
                if row.policy_decision == "ALLOW" and row.injection_risk_score >= 0.55:
                    near_misses += 1
                for rule_id in json.loads(row.triggered_rules or "[]"):
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
                for stage, ms in json.loads(row.latency_ms or "{}").items():
                    latencies.setdefault(stage, []).append(ms)

            top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            avg_latency = {k: sum(v) / len(v) for k, v in latencies.items() if v}

            return {
                "total_requests": len(rows),
                "decisions": decisions,
                "avg_risk_score": sum(risk_scores) / len(risk_scores),
                "top_rules": [{"rule_id": r, "count": c} for r, c in top_rules],
                "near_misses": near_misses,
                "avg_latency_ms": avg_latency,
            }

    @staticmethod
    def _row_to_dict(row: TelemetryEvent) -> dict[str, Any]:
        return {
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "request_id": row.request_id,
            "prompt_hash": row.prompt_hash,
            "redacted_prompt": row.redacted_prompt,
            "triggered_rules": json.loads(row.triggered_rules or "[]"),
            "injection_risk_score": row.injection_risk_score,
            "policy_decision": row.policy_decision,
            "sanitization_method": row.sanitization_method,
            "model_version": row.model_version,
            "policy_version": row.policy_version,
            "latency_ms": json.loads(row.latency_ms or "{}"),
            "output_status": row.output_status,
            "output_violation_reasons": json.loads(row.output_violation_reasons or "[]"),
            "attack_type": row.attack_type,
            "source": row.source,
        }
