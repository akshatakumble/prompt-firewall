from firewall.config import Thresholds
from firewall.policy.engine import PolicyAction, PolicyEngine


def test_allow_low_risk():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("hello", rule_score=0.0, classifier_score=0.1)
    assert decision.action == PolicyAction.ALLOW


def test_sanitize_medium_risk():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("suspicious", rule_score=0.0, classifier_score=0.5)
    assert decision.action == PolicyAction.SANITIZE
    assert decision.sanitized_text is not None


def test_block_high_risk():
    engine = PolicyEngine(Thresholds())
    decision = engine.decide("attack", rule_score=0.9, classifier_score=0.2)
    assert decision.action == PolicyAction.BLOCK
