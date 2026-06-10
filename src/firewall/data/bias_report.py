"""Data slicing for bias and distribution analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def slice_distribution(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Compute label distribution within each slice of a categorical column."""

    if column not in df.columns or df.empty:
        return {}

    slices: dict[str, dict[str, Any]] = {}
    for value, group in df.groupby(column, dropna=False):
        key = str(value)
        total = len(group)
        label_counts = group["label"].value_counts().to_dict()
        injection_rate = float(label_counts.get("INJECTION", 0) / total) if total else 0.0
        slices[key] = {
            "rows": total,
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "injection_rate": round(injection_rate, 4),
        }
    return slices


def build_bias_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a bias-slicing report across source and attack_type."""

    total = len(df)
    overall = {
        "rows": total,
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
    }
    if total:
        injection = overall["label_counts"].get("INJECTION", 0)
        overall["injection_rate"] = round(injection / total, 4)
        overall["benign_rate"] = round(1 - overall["injection_rate"], 4)

    by_source = slice_distribution(df, "source")
    by_attack_type = slice_distribution(df, "attack_type")

    imbalance_threshold = 0.15
    warnings: list[str] = []
    for slice_name, stats in by_source.items():
        rate = stats.get("injection_rate", 0.0)
        if rate < imbalance_threshold or rate > (1 - imbalance_threshold):
            warnings.append(
                f"source={slice_name} has skewed injection_rate={rate:.2%}; "
                "consider re-sampling if used for training."
            )

    return {
        "overall": overall,
        "by_source": by_source,
        "by_attack_type": by_attack_type,
        "warnings": warnings,
        "mitigation_notes": [
            "Training corpus uses balanced INJECTION/BENIGN sampling before splitting.",
            "Held-out Salad-Data benchmark is kept separate to avoid contamination.",
        ],
    }
