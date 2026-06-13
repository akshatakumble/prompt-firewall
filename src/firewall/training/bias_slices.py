"""Model-level bias detection via Fairlearn data slicing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from firewall.data.bias_report import add_prompt_length_bucket
from firewall.training.metrics import compute_binary_metrics


def slice_bias_report(
    df: pd.DataFrame,
    y_true: list[int],
    y_scores: list[float],
    *,
    threshold: float = 0.5,
    slice_columns: tuple[str, ...] = ("attack_type", "source", "prompt_length_bucket"),
    recall_drop_threshold: float = 0.15,
) -> dict[str, Any]:
    """Compute per-slice recall/FPR and flag disparities vs global recall."""

    work = df.copy()
    if "prompt_length_bucket" not in work.columns:
        work = add_prompt_length_bucket(work)

    global_metrics = compute_binary_metrics(y_true, y_scores, threshold=threshold)
    global_recall = global_metrics["recall"]

    try:
        from fairlearn.metrics import MetricFrame, false_positive_rate
        from sklearn.metrics import recall_score
    except ImportError:
        return {
            "available": False,
            "reason": "fairlearn not installed",
            "global_metrics": global_metrics,
        }

    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    flagged: list[str] = []
    slices: dict[str, Any] = {}

    for column in slice_columns:
        if column not in work.columns:
            continue

        frame = MetricFrame(
            metrics={"recall": recall_score, "false_positive_rate": false_positive_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=work[column].astype(str),
        )
        by_group = {
            str(k): {
                "recall": round(float(frame.by_group["recall"][k]), 4),
                "fpr": round(float(frame.by_group["false_positive_rate"][k]), 4),
                "rows": int((work[column].astype(str) == str(k)).sum()),
            }
            for k in frame.by_group["recall"].index
        }

        for name, stats in by_group.items():
            drop = global_recall - stats["recall"]
            if drop > recall_drop_threshold:
                flagged.append(
                    f"{column}={name}: recall={stats['recall']:.1%} "
                    f"({drop:.1%} below global {global_recall:.1%})"
                )

        slices[column] = {
            "by_group": by_group,
            "recall_disparity": round(
                float(frame.by_group["recall"].max() - frame.by_group["recall"].min()), 4
            ),
        }

    return {
        "available": True,
        "tool": "fairlearn.MetricFrame",
        "threshold": threshold,
        "global_metrics": global_metrics,
        "global_recall": global_recall,
        "slices": slices,
        "flagged_disparities": flagged,
        "selection_policy": (
            "Primary constraint: recall >= 85% (missed attacks worse than false alarms). "
            "Secondary: FPR <= 5%. Flag slices where recall drops >15pp below global."
        ),
    }

