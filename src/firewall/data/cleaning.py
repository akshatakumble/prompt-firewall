"""Data cleaning and enrichment helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from firewall.data.schema import REQUIRED_COLUMNS, VALID_LABELS


def token_count(text: str) -> int:
    return len(text.split())


def add_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach whitespace token counts into the metadata JSON column."""

    def enrich(row: pd.Series) -> str:
        meta: dict[str, Any] = {}
        if isinstance(row.get("metadata"), str) and row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except json.JSONDecodeError:
                meta = {"raw_metadata": row["metadata"]}
        meta["token_count"] = token_count(str(row["text"]))
        meta["char_length"] = len(str(row["text"]))
        return json.dumps(meta, sort_keys=True)

    df = df.copy()
    df["metadata"] = df.apply(enrich, axis=1)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty prompts, invalid labels, and exact duplicates."""

    if df.empty:
        return df

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[cleaned["label"].isin(VALID_LABELS)]
    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first")
    return cleaned.reset_index(drop=True)
