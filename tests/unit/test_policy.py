from firewall.config import Thresholds
from firewall.policy.engine import PolicyAction, PolicyEngine
from firewall.rules.engine import RuleMatch, RuleResult


def _soft_rule_result() -> RuleResult:
    return RuleResult(
        matches=[
            RuleMatch(
                rule_id="roleplay_override",
                name="Roleplay override",
                severity="medium",
                score_boost=0.65,
            )
        ],
        max_score=0.65,
    )


def _hard_rule_result() -> RuleResult:
    return RuleResult(
        matches=[
            RuleMatch(
                rule_id="dan_jailbreak",
                name="DAN jailbreak",
                severity="high",
                score_boost=0.80,
            )
        ],
        max_score=0.80,
    )


def test_allow_low_risk():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("hello", RuleResult(), classifier_score=0.1)
    assert decision.action == PolicyAction.ALLOW


def test_sanitize_mid_band_with_soft_rule():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("suspicious", _soft_rule_result(), classifier_score=0.6)
    assert decision.action == PolicyAction.SANITIZE
    assert decision.sanitized_text is not None
    assert "flagged as potentially adversarial" in decision.sanitized_text


def test_block_high_classifier_with_rule():
    engine = PolicyEngine(Thresholds())
    soft = _soft_rule_result()
    decision = engine.decide("attack", soft, classifier_score=0.85)
    assert decision.action == PolicyAction.BLOCK


def test_high_classifier_without_rule_sanitizes():
    engine = PolicyEngine(Thresholds(), classifier_operating_threshold=0.3)
    decision = engine.decide("write code", RuleResult(), classifier_score=0.99)
    assert decision.action == PolicyAction.SANITIZE


def test_block_hard_rule():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("attack", _hard_rule_result(), classifier_score=0.2)
    assert decision.action == PolicyAction.BLOCK
