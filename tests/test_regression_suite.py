from __future__ import annotations

import json
from pathlib import Path

import pytest

from firewall.service import FirewallService

SUITE_PATH = Path(__file__).parent / "regression" / "attack_suite.jsonl"


def load_attack_suite() -> list[dict]:
    cases = []
    with open(SUITE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


@pytest.fixture(scope="module")
def firewall() -> FirewallService:
    return FirewallService()


@pytest.mark.parametrize("case", load_attack_suite(), ids=lambda c: c["text"][:40])
def test_regression_attack_suite(firewall: FirewallService, case: dict) -> None:
    result = firewall.inspect(case["text"])
    expected = case["expected_action"]

    if expected == "BLOCK":
        assert result.action == "BLOCK", f"Expected BLOCK for: {case['text']}"
    elif expected == "SANITIZE":
        assert result.action in ("SANITIZE", "BLOCK"), f"Expected SANITIZE/BLOCK for: {case['text']}"
    else:
        assert result.action in ("ALLOW", "SANITIZE"), f"Expected ALLOW/SANITIZE for: {case['text']}"


def test_regression_metrics(firewall: FirewallService) -> None:
    cases = load_attack_suite()
    injection_cases = [c for c in cases if c["label"] == "INJECTION"]
    benign_cases = [c for c in cases if c["label"] == "BENIGN"]

    tp = sum(1 for c in injection_cases if firewall.inspect(c["text"]).action in ("BLOCK", "SANITIZE"))
    fp = sum(1 for c in benign_cases if firewall.inspect(c["text"]).action == "BLOCK")

    recall = tp / len(injection_cases) if injection_cases else 1.0
    fpr = fp / len(benign_cases) if benign_cases else 0.0

    assert recall >= 0.85, f"Attack recall {recall:.2f} below 0.85"
    assert fpr <= 0.05, f"Benign FPR {fpr:.2f} above 0.05"
