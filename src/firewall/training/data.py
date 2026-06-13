"""Load versioned training data from the data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from firewall.config import PROJECT_ROOT


def load_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / config["paths"]["registry"] / "manifest.json"
    if not manifest_path.exists():
        return {"version": config.get("dataset_version", "unknown"), "note": "manifest missing"}
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_splits(
    config: dict[str, Any],
    *,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = PROJECT_ROOT / config["paths"]["curated"] / "splits"
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "validation.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    if max_train and len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=42)
    if max_val and len(val_df) > max_val:
        val_df = val_df.sample(n=max_val, random_state=42)
    if max_test and len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=42)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def label_to_int(label: str) -> int:
    return 1 if label == "INJECTION" else 0


def labels_to_int(series: pd.Series) -> list[int]:
    return [label_to_int(str(x)) for x in series]
