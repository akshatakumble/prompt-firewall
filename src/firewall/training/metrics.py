"""Binary classification metrics for injection detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_arrays(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    result: dict[str, Any] = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "support": int(len(y_true)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(np.unique(y_true)) > 1:
            result["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        else:
            result["pr_auc"] = 0.0
            result["roc_auc"] = 0.0

    return result


def compute_binary_metrics(
    y_true: list[int],
    y_scores: list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    return metrics_from_arrays(y_true, y_pred, y_scores)


def meets_security_sla(metrics: dict[str, Any], *, min_recall: float, max_fpr: float) -> bool:
    return metrics.get("recall", 0.0) >= min_recall and metrics.get("fpr", 1.0) <= max_fpr
