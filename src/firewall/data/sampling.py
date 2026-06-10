"""Balanced sampling and train/val/test splitting."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def balanced_sample(
    df: pd.DataFrame,
    target_total: int,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample up to target_total rows with a balanced INJECTION/BENIGN split."""

    if target_total <= 0 or df.empty:
        return df

    per_class_target = target_total // 2
    class_counts = df["label"].value_counts()
    injection_available = int(class_counts.get("INJECTION", 0))
    benign_available = int(class_counts.get("BENIGN", 0))
    take_per_class = min(injection_available, benign_available, per_class_target)

    if take_per_class == 0:
        logger.warning("Cannot build a balanced sample; one class is missing.")
        return df.iloc[0:0].copy()

    if take_per_class < per_class_target:
        logger.warning(
            "Requested %s rows per class but only %s available in the limiting class.",
            per_class_target,
            take_per_class,
        )

    frames: list[pd.DataFrame] = []
    for label in ("INJECTION", "BENIGN"):
        subset = df[df["label"] == label]
        frames.append(subset.sample(n=take_per_class, random_state=random_state))

    combined = pd.concat(frames, ignore_index=True)
    return combined.sample(frac=1, random_state=random_state).reset_index(drop=True)


def create_stratified_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 splits by label."""

    if df.empty:
        empty = df.copy()
        return empty, empty, empty

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["label"],
        random_state=random_state,
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
