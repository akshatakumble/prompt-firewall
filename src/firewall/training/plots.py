"""Save evaluation plots for reports and presentations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    out_path: Path,
    *,
    title: str = "Confusion Matrix",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["BENIGN", "INJECTION"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve(
    y_true: list[int],
    y_prob: list[float],
    out_path: Path,
    *,
    title: str = "Precision-Recall Curve",
) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(pr_auc)


def save_roc_curve(
    y_true: list[int],
    y_prob: list[float],
    out_path: Path,
    *,
    title: str = "ROC Curve",
) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


def save_threshold_sweep(
    sweep: list[dict[str, Any]],
    out_path: Path,
    *,
    chosen_threshold: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in sweep]
    recalls = [row["recall"] for row in sweep]
    precisions = [row["precision"] for row in sweep]
    fprs = [row["fpr"] for row in sweep]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.plot(thresholds, precisions, marker="s", label="Precision")
    ax.plot(thresholds, fprs, marker="^", label="FPR")
    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, color="red", linestyle="--", label=f"chosen={chosen_threshold}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold sweep (INJECTION score)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
