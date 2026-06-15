from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from firewall.classifier.scorer import ClassifierScorer
from firewall.config import AppConfig, load_app_config, load_rules_config
from firewall.llm.client import LLMClient
from firewall.output_monitor.monitor import OutputMonitor
from firewall.policy.engine import PolicyAction, PolicyEngine
from firewall.rules.engine import RuleEngine
from firewall.telemetry.store import TelemetryStore


@dataclass
class InspectResult:
    request_id: str
    action: str
    risk_score: float
    triggered_rules: list[str]
    sanitized_prompt: str | None
    reason: str
    is_near_miss: bool
    latency_ms: dict[str, float]


@dataclass
class ChatResult:
    request_id: str
    action: str
    response: str
    output_status: str
    risk_score: float
    triggered_rules: list[str]
    violation_reasons: list[str]
    latency_ms: dict[str, float]


class FirewallService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_app_config()
        rules_cfg = load_rules_config()
        self.rule_engine = RuleEngine(rules_cfg)
        self.classifier = ClassifierScorer(self.config.classifier)
        self.policy_engine = PolicyEngine(
            self.config.thresholds,
            rules_cfg.get("sanitization_patterns", []),
            classifier_operating_threshold=self.classifier.operating_threshold,
        )
        self.output_monitor = OutputMonitor(rules_cfg.get("output_rules", []))
        self.telemetry = TelemetryStore()
        self.llm = LLMClient(
            provider=self.config.llm.get("provider"),
            model=self.config.llm.get("model"),
        )

    def inspect(self, prompt: str) -> InspectResult:
        request_id = str(uuid.uuid4())
        latencies: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        rule_result = self.rule_engine.evaluate(prompt)
        latencies["rules"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        classifier_score = self.classifier.score(prompt)
        latencies["classifier"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        decision = self.policy_engine.decide(
            prompt, rule_result, classifier_score
        )
        latencies["policy"] = (time.perf_counter() - t0) * 1000
        latencies["total"] = (time.perf_counter() - total_start) * 1000

        event = self.telemetry.build_event(
            request_id=request_id,
            prompt=prompt,
            triggered_rules=rule_result.triggered_rule_ids,
            risk_score=decision.risk_score,
            policy_decision=decision.action.value,
            model_version=self.config.model_version,
            policy_version=self.config.policy_version,
            latency_ms=latencies,
            sanitization_method="delimiter_wrap" if decision.action == PolicyAction.SANITIZE else None,
            log_snippet=self.config.telemetry.get("log_redacted_snippet", True),
            snippet_max_chars=self.config.telemetry.get("snippet_max_chars", 200),
        )
        self.telemetry.log_event(event)

        return InspectResult(
            request_id=request_id,
            action=decision.action.value,
            risk_score=decision.risk_score,
            triggered_rules=rule_result.triggered_rule_ids,
            sanitized_prompt=decision.sanitized_text,
            reason=decision.reason,
            is_near_miss=decision.is_near_miss,
            latency_ms=latencies,
        )

    async def chat(self, prompt: str) -> ChatResult:
        inspect_result = self.inspect(prompt)

        if inspect_result.action == PolicyAction.BLOCK.value:
            return ChatResult(
                request_id=inspect_result.request_id,
                action=inspect_result.action,
                response="This request cannot be processed.",
                output_status="BLOCKED",
                risk_score=inspect_result.risk_score,
                triggered_rules=inspect_result.triggered_rules,
                violation_reasons=["input_blocked"],
                latency_ms=inspect_result.latency_ms,
            )

        llm_prompt = inspect_result.sanitized_prompt or prompt
        system_prompt = self.config.llm.get("system_prompt", "")

        t0 = time.perf_counter()
        llm_response = await self.llm.generate(llm_prompt, system_prompt)
        llm_latency = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        output_result = self.output_monitor.evaluate(llm_response)
        output_latency = (time.perf_counter() - t0) * 1000

        latencies = dict(inspect_result.latency_ms)
        latencies["llm"] = llm_latency
        latencies["output_monitor"] = output_latency
        latencies["total"] = latencies.get("total", 0) + llm_latency + output_latency

        event = self.telemetry.build_event(
            request_id=inspect_result.request_id,
            prompt=prompt,
            triggered_rules=inspect_result.triggered_rules,
            risk_score=inspect_result.risk_score,
            policy_decision=inspect_result.action,
            model_version=self.config.model_version,
            policy_version=self.config.policy_version,
            latency_ms=latencies,
            output_status=output_result.status.value,
            output_violation_reasons=output_result.violation_reasons,
            log_snippet=self.config.telemetry.get("log_redacted_snippet", True),
        )
        self.telemetry.log_event(event)

        return ChatResult(
            request_id=inspect_result.request_id,
            action=inspect_result.action,
            response=output_result.text,
            output_status=output_result.status.value,
            risk_score=inspect_result.risk_score,
            triggered_rules=inspect_result.triggered_rules,
            violation_reasons=output_result.violation_reasons,
            latency_ms=latencies,
        )
