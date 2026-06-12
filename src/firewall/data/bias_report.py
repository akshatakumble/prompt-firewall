"""Data slicing and bias analysis for the firewall training corpus.

Uses Fairlearn MetricFrame for slice-wise selection-rate (injection rate) disparity,
similar to TFMA / SliceFinder workflows described in the MLOps assignment.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_IMBALANCE_THRESHOLD = 0.15
DEFAULT_MIN_REPRESENTATION = 0.05
DEFAULT_MIN_ROWS_PER_SLICE = 100


def _parse_metadata_series(df: pd.DataFrame) -> pd.DataFrame:
    """Extract char_length and token_count from metadata JSON when present."""

    work = df.copy()
    if "metadata" not in work.columns:
        work["char_length"] = work["text"].astype(str).str.len()
        work["token_count"] = work["text"].astype(str).str.split().str.len()
        return work

    char_lengths: list[int] = []
    token_counts: list[int] = []
    for raw in work["metadata"]:
        meta: dict[str, Any] = {}
        if isinstance(raw, str) and raw:
            try:
                meta = json.loads(raw)
            except json.JSONDecodeError:
                meta = {}
        char_lengths.append(int(meta.get("char_length", 0)))
        token_counts.append(int(meta.get("token_count", 0)))

    work["char_length"] = [
        length if length else len(str(text))
        for length, text in zip(char_lengths, work["text"], strict=True)
    ]
    work["token_count"] = [
        count if count else len(str(text).split())
        for count, text in zip(token_counts, work["text"], strict=True)
    ]
    return work


def add_prompt_length_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical prompt-length buckets for slicing."""

    work = _parse_metadata_series(df)
    bins = [0, 50, 150, 400, 1000, 10_000]
    labels = ["short", "medium", "long", "very_long", "extreme"]
    work["prompt_length_bucket"] = pd.cut(
        work["char_length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    return work


def slice_distribution(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Compute label distribution within each slice of a categorical column."""

    if column not in df.columns or df.empty:
        return {}

    total_rows = len(df)
    slices: dict[str, dict[str, Any]] = {}
    for value, group in df.groupby(column, dropna=False):
        key = str(value)
        count = len(group)
        label_counts = group["label"].value_counts().to_dict()
        injection_rate = float(label_counts.get("INJECTION", 0) / count) if count else 0.0
        slices[key] = {
            "rows": count,
            "share_of_dataset": round(count / total_rows, 4) if total_rows else 0.0,
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "injection_rate": round(injection_rate, 4),
            "benign_rate": round(1 - injection_rate, 4),
        }
    return slices


def fairlearn_slice_metrics(
    df: pd.DataFrame,
    slice_column: str,
    *,
    label_column: str = "label",
) -> dict[str, Any]:
    """Fairlearn MetricFrame: injection selection rate per slice."""

    if df.empty or slice_column not in df.columns:
        return {"available": False, "reason": f"missing column {slice_column}"}

    try:
        from fairlearn.metrics import MetricFrame, selection_rate
    except ImportError:
        logger.warning("fairlearn not installed; skipping MetricFrame analysis")
        return {"available": False, "reason": "fairlearn not installed"}

    # Dataset-level slicing: measure injection label rate per subgroup.
    y_true = (df[label_column] == "INJECTION").astype(int)
    metric_frame = MetricFrame(
        metrics={"injection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_true,
        sensitive_features=df[slice_column].astype(str),
    )

    by_group = metric_frame.by_group["injection_rate"].to_dict()
    rates = [float(v) for v in by_group.values() if pd.notna(v)]
    disparity = max(rates) - min(rates) if rates else 0.0

    return {
        "available": True,
        "tool": "fairlearn.MetricFrame",
        "slice_column": slice_column,
        "by_group": {str(k): round(float(v), 4) for k, v in by_group.items()},
        "max_injection_rate": round(max(rates), 4) if rates else 0.0,
        "min_injection_rate": round(min(rates), 4) if rates else 0.0,
        "disparity": round(disparity, 4),
    }


def detect_slice_warnings(
    slices: dict[str, dict[str, Any]],
    *,
    slice_name: str,
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    min_representation: float = DEFAULT_MIN_REPRESENTATION,
    min_rows: int = DEFAULT_MIN_ROWS_PER_SLICE,
) -> list[str]:
    """Flag underrepresented or label-skewed slices."""

    warnings: list[str] = []
    for name, stats in slices.items():
        rows = stats.get("rows", 0)
        share = stats.get("share_of_dataset", 0.0)
        rate = stats.get("injection_rate", 0.0)

        if rows < min_rows:
            warnings.append(
                f"{slice_name}={name} underrepresented ({rows} rows < {min_rows}); "
                "consider oversampling."
            )
        if share < min_representation:
            warnings.append(
                f"{slice_name}={name} low representation ({share:.1%} of corpus); "
                "may hurt generalization."
            )
        if rate < imbalance_threshold or rate > (1 - imbalance_threshold):
            warnings.append(
                f"{slice_name}={name} skewed injection_rate={rate:.1%}; "
                "consider re-sampling before training."
            )
    return warnings


def mitigate_slice_representation(
    df: pd.DataFrame,
    slice_column: str,
    *,
    min_per_slice: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Oversample underrepresented slices within each label class (re-sampling mitigation)."""

    if slice_column not in df.columns or df.empty:
        return df, {"applied": False, "reason": f"missing {slice_column}"}

    frames: list[pd.DataFrame] = []
    actions: list[str] = []

    for label in df["label"].unique():
        label_df = df[df["label"] == label]
        for slice_value, group in label_df.groupby(slice_column, dropna=False):
            count = len(group)
            if count == 0:
                continue
            if count < min_per_slice:
                sampled = group.sample(n=min_per_slice, replace=True, random_state=random_state)
                actions.append(
                    f"oversampled {slice_column}={slice_value} label={label}: "
                    f"{count} -> {min_per_slice}"
                )
                frames.append(sampled)
            else:
                frames.append(group)

    if not frames:
        return df, {"applied": False, "reason": "no slices to mitigate"}

    mitigated = pd.concat(frames, ignore_index=True)
    mitigated = mitigated.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return mitigated, {"applied": True, "actions": actions, "rows_after": len(mitigated)}


def build_bias_report(
    df: pd.DataFrame,
    *,
    stage: str = "current",
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    min_representation: float = DEFAULT_MIN_REPRESENTATION,
    min_rows_per_slice: int = DEFAULT_MIN_ROWS_PER_SLICE,
) -> dict[str, Any]:
    """Build a bias-slicing report across source, attack_type, and prompt length."""

    sliced = add_prompt_length_bucket(df)
    total = len(sliced)

    overall = {
        "rows": total,
        "label_counts": {str(k): int(v) for k, v in sliced["label"].value_counts().to_dict().items()},
    }
    if total:
        injection = overall["label_counts"].get("INJECTION", 0)
        overall["injection_rate"] = round(injection / total, 4)
        overall["benign_rate"] = round(1 - overall["injection_rate"], 4)

    by_source = slice_distribution(sliced, "source")
    by_attack_type = slice_distribution(sliced, "attack_type")
    by_prompt_length = slice_distribution(sliced, "prompt_length_bucket")

    fairlearn_metrics = {
        "by_source": fairlearn_slice_metrics(sliced, "source"),
        "by_attack_type": fairlearn_slice_metrics(sliced, "attack_type"),
        "by_prompt_length_bucket": fairlearn_slice_metrics(sliced, "prompt_length_bucket"),
    }

    warnings: list[str] = []
    warnings.extend(
        detect_slice_warnings(
            by_source,
            slice_name="source",
            imbalance_threshold=imbalance_threshold,
            min_representation=min_representation,
            min_rows=min_rows_per_slice,
        )
    )
    warnings.extend(
        detect_slice_warnings(
            by_attack_type,
            slice_name="attack_type",
            imbalance_threshold=imbalance_threshold,
            min_representation=min_representation,
            min_rows=min_rows_per_slice,
        )
    )
    warnings.extend(
        detect_slice_warnings(
            by_prompt_length,
            slice_name="prompt_length_bucket",
            imbalance_threshold=imbalance_threshold,
            min_representation=min_representation,
            min_rows=min_rows_per_slice,
        )
    )

    for name, metrics in fairlearn_metrics.items():
        if metrics.get("available") and metrics.get("disparity", 0) > 0.25:
            warnings.append(
                f"fairlearn {name}: injection-rate disparity={metrics['disparity']:.1%} "
                "(threshold 25%); consider re-sampling or threshold tuning."
            )

    return {
        "stage": stage,
        "overall": overall,
        "by_source": by_source,
        "by_attack_type": by_attack_type,
        "by_prompt_length_bucket": by_prompt_length,
        "fairlearn_metrics": fairlearn_metrics,
        "warnings": warnings,
        "mitigation_notes": [
            "Label balancing via balanced_sample() before splitting.",
            "Optional attack_type oversampling via mitigate_slice_representation().",
            "Held-out benchmarks (Salad-Data) excluded from training corpus.",
            "Policy thresholds tuned on validation set to limit benign false positives.",
        ],
    }


def build_comparative_bias_report(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    mitigation_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Compare corpus bias before and after mitigation steps."""

    before = build_bias_report(before_df, stage="before_mitigation")
    after = build_bias_report(after_df, stage="after_mitigation")

    return {
        "before_mitigation": before,
        "after_mitigation": after,
        "mitigation_applied": mitigation_actions or [],
        "warnings": sorted(set(before["warnings"]) - set(after["warnings"])),
        "remaining_warnings": after["warnings"],
    }
