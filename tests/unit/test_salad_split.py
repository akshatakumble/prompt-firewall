"""Tests for leakage-safe Salad-Data splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from firewall.data.salad_split import (
    assert_no_id_leakage,
    salad_split_metadata,
    split_salad_train_holdout,
)


def _salad_rows(n: int) -> pd.DataFrame:
    rows = []
    attack_types = ["O1: Toxic Content", "O14: Illegal Activities", "jb", "gptfuzz"]
    for i in range(n):
        attack = attack_types[i % len(attack_types)]
        rows.append(
            {
                "id": f"salad-{i}",
                "text": f"harmful prompt {i} for {attack}",
                "label": "INJECTION",
                "source": "eval_salad-data",
                "attack_type": attack,
                "metadata": "{}",
            }
        )
    return pd.DataFrame(rows)


def test_split_is_disjoint_and_preserves_rows():
    df = _salad_rows(1000)
    train_df, holdout_df = split_salad_train_holdout(df, holdout_fraction=0.2, random_state=42)

    assert len(train_df) == 800
    assert len(holdout_df) == 200
    assert set(train_df["id"]).isdisjoint(set(holdout_df["id"]))
    assert train_df["source"].eq("train_salad-data").all()
    assert holdout_df["source"].eq("eval_salad-data").all()


def test_assert_no_id_leakage_raises():
    train_df = _salad_rows(10)
    holdout_df = train_df.iloc[:2].copy()
    with pytest.raises(ValueError, match="leaked"):
        assert_no_id_leakage(train_df, holdout_df)


def test_salad_split_metadata_reports_passed():
    df = _salad_rows(200)
    train_df, holdout_df = split_salad_train_holdout(df, holdout_fraction=0.25, random_state=7)
    meta = salad_split_metadata(train_df, holdout_df, holdout_fraction=0.25, random_state=7)
    assert meta["leakage_check"] == "passed"
    assert meta["leakage_ids_overlap"] == 0
    assert meta["train_pool_rows"] + meta["benchmark_holdout_rows"] == 200
