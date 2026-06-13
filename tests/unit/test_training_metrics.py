"""Unit tests for training metrics helpers."""

from __future__ import annotations

from firewall.training.metrics import compute_binary_metrics, meets_security_sla


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.9, 0.8]
    m = compute_binary_metrics(y_true, y_scores, threshold=0.5)
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0


def test_meets_security_sla():
    assert meets_security_sla({"recall": 0.9, "fpr": 0.03}, min_recall=0.85, max_fpr=0.05)
    assert not meets_security_sla({"recall": 0.8, "fpr": 0.03}, min_recall=0.85, max_fpr=0.05)
