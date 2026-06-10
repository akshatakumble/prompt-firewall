"""Unit tests for ge_runner validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.ge_runner import generate_baseline_schema, validate_dataframe


def test_generate_baseline_schema():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "text": "hello",
                "label": "BENIGN",
                "source": "train_wildjailbreak",
                "attack_type": "benign",
            }
        ]
    )
    schema = generate_baseline_schema(df)
    assert "id" in schema["required_columns"]
    assert schema["row_count"] == 1


def test_validate_dataframe_detects_unknown_label():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "text": "hello",
                "label": "BAD",
                "source": "s",
                "attack_type": "a",
            }
        ]
    )
    result = validate_dataframe(df)
    assert "unknown_labels_detected" in result["anomalies"]["hard_fail"]


def test_ge_runner_baseline_writes_schema(tmp_path, monkeypatch):
    import scripts.ge_runner as ge_runner

    monkeypatch.setattr(ge_runner, "PROJECT_ROOT", tmp_path)
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "text": "hello",
                "label": "BENIGN",
                "source": "s",
                "attack_type": "a",
            }
        ]
    )
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    schema_path = tmp_path / "data" / "metrics" / "schema" / "baseline" / "schema.json"
    schema = generate_baseline_schema(df)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    result = validate_dataframe(df, schema)
    assert result["anomalies"]["hard_fail"] == []
