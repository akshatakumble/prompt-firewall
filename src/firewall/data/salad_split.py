"""Leakage-safe train / benchmark split for Salad-Data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SALAD_TRAIN_SOURCE = "train_salad-data"
SALAD_BENCHMARK_SOURCE = "eval_salad-data"


def assert_no_id_leakage(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> None:
    overlap = set(train_df["id"]) & set(holdout_df["id"])
    if overlap:
        raise ValueError(f"Salad split leaked {len(overlap)} ids between train and benchmark.")


def split_salad_train_holdout(
    df: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Salad-Data into disjoint train-pool and held-out benchmark sets.

    Uses stable id-based splitting (stratified by attack_type when possible).
    """
    if df.empty:
        empty = df.copy()
        return empty, empty

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    stratify = None
    if "attack_type" in df.columns:
        counts = df["attack_type"].value_counts()
        if counts.min() >= 2:
            stratify = df["attack_type"]

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = train_df.copy()
    holdout_df = holdout_df.copy()
    train_df["source"] = SALAD_TRAIN_SOURCE
    holdout_df["source"] = SALAD_BENCHMARK_SOURCE

    assert_no_id_leakage(train_df, holdout_df)

    logger.info(
        "Salad split: train_pool=%s benchmark_holdout=%s (holdout_fraction=%.2f)",
        len(train_df),
        len(holdout_df),
        holdout_fraction,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def salad_split_metadata(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    holdout_fraction: float,
    random_state: int,
) -> dict[str, Any]:
    overlap = set(train_df["id"]) & set(holdout_df["id"])
    return {
        "holdout_fraction": holdout_fraction,
        "random_state": random_state,
        "train_pool_rows": len(train_df),
        "benchmark_holdout_rows": len(holdout_df),
        "leakage_ids_overlap": len(overlap),
        "leakage_check": "passed" if not overlap else "failed",
    }
