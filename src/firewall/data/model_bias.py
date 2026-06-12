"""Model performance slicing for bias analysis (Fairlearn MetricFrame)."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def evaluate_slice_performance(
    df: pd.DataFrame,
    predict_fn: Callable[[str], bool],
    *,
    slice_columns: tuple[str, ...] = ("attack_type", "source", "prompt_length_bucket"),
) -> dict[str, Any]:
    """
    Compute precision/recall/FPR per slice using Fairlearn MetricFrame.

    predict_fn: returns True when the model flags an attack (BLOCK or SANITIZE).
    """

    try:
        from fairlearn.metrics import MetricFrame, false_positive_rate
        from sklearn.metrics import precision_score, recall_score
    except ImportError:
        return {"available": False, "reason": "fairlearn or sklearn not installed"}

    work = df.copy()
    if "prompt_length_bucket" not in work.columns:
        from firewall.data.bias_report import add_prompt_length_bucket

        work = add_prompt_length_bucket(work)

    y_true = (work["label"] == "INJECTION").astype(int)
    y_pred = work["text"].map(lambda text: int(predict_fn(str(text))))

    metrics = {
        "precision": precision_score,
        "recall": recall_score,
        "false_positive_rate": false_positive_rate,
    }

    results: dict[str, Any] = {"available": True, "tool": "fairlearn.MetricFrame", "slices": {}}
    for column in slice_columns:
        if column not in work.columns:
            continue
        frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=work[column].astype(str),
        )
        by_group = frame.by_group.to_dict()
        slice_metrics = {
            metric: {str(k): round(float(v), 4) for k, v in values.items()}
            for metric, values in by_group.items()
        }

        recalls = list(slice_metrics.get("recall", {}).values())
        fprs = list(slice_metrics.get("false_positive_rate", {}).values())
        results["slices"][column] = {
            **slice_metrics,
            "recall_disparity": round(max(recalls) - min(recalls), 4) if recalls else 0.0,
            "fpr_disparity": round(max(fprs) - min(fprs), 4) if fprs else 0.0,
        }

    return results
