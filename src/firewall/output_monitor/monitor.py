from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputStatus(str, Enum):
    OK = "OK"
    REDACT = "REDACT"
    REFUSE = "REFUSE"


@dataclass
class OutputResult:
    status: OutputStatus
    text: str
    violation_reasons: list[str] = field(default_factory=list)


class OutputMonitor:
    def __init__(self, output_rules: list[dict[str, Any]]) -> None:
        self._rules = [
            {
                "id": r["id"],
                "name": r["name"],
                "pattern": re.compile(r["pattern"]),
                "action": r.get("action", "redact"),
            }
            for r in output_rules
        ]

    def evaluate(self, text: str) -> OutputResult:
        violations: list[str] = []
        result_text = text
        refuse = False

        for rule in self._rules:
            if rule["pattern"].search(result_text):
                violations.append(rule["name"])
                if rule["action"] == "refuse":
                    refuse = True
                elif rule["action"] == "redact":
                    result_text = rule["pattern"].sub("[REDACTED]", result_text)

        if refuse:
            return OutputResult(
                status=OutputStatus.REFUSE,
                text="I cannot provide that response due to policy restrictions.",
                violation_reasons=violations,
            )

        if violations:
            return OutputResult(
                status=OutputStatus.REDACT,
                text=result_text,
                violation_reasons=violations,
            )

        return OutputResult(status=OutputStatus.OK, text=text)
