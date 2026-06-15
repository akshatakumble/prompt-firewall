from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleMatch:
    rule_id: str
    name: str
    severity: str
    score_boost: float


@dataclass
class RuleResult:
    matches: list[RuleMatch] = field(default_factory=list)
    max_score: float = 0.0

    @property
    def triggered_rule_ids(self) -> list[str]:
        return [m.rule_id for m in self.matches]

    @property
    def hard_rule_fired(self) -> bool:
        return any(m.severity == "high" for m in self.matches)

    @property
    def soft_rule_fired(self) -> bool:
        return any(m.severity in {"medium", "low"} for m in self.matches)

    @property
    def any_rule_fired(self) -> bool:
        return bool(self.matches)


class RuleEngine:
    def __init__(self, rules_config: dict[str, Any]) -> None:
        self._rules = []
        for rule in rules_config.get("rules", []):
            self._rules.append(
                {
                    "id": rule["id"],
                    "name": rule["name"],
                    "pattern": re.compile(rule["pattern"]),
                    "severity": rule.get("severity", "medium"),
                    "score_boost": float(rule.get("score_boost", 0.5)),
                }
            )

    def evaluate(self, text: str) -> RuleResult:
        matches: list[RuleMatch] = []
        max_score = 0.0
        for rule in self._rules:
            if rule["pattern"].search(text):
                match = RuleMatch(
                    rule_id=rule["id"],
                    name=rule["name"],
                    severity=rule["severity"],
                    score_boost=rule["score_boost"],
                )
                matches.append(match)
                max_score = max(max_score, rule["score_boost"])
        return RuleResult(matches=matches, max_score=max_score)
