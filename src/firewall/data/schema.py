"""Unified dataset schema constants."""

from __future__ import annotations

REQUIRED_COLUMNS = ("id", "text", "label", "source", "attack_type")
OPTIONAL_COLUMNS = ("metadata",)

UNIFIED_SCHEMA = {
    "id": "string",
    "text": "string",
    "label": "string",  # INJECTION | BENIGN
    "source": "string",
    "attack_type": "string",
    "metadata": "string",
}

VALID_LABELS = frozenset({"INJECTION", "BENIGN"})

INJECTION_DATA_TYPES = frozenset({"vanilla_harmful", "adversarial_harmful"})
BENIGN_DATA_TYPES = frozenset({"vanilla_benign", "adversarial_benign"})
