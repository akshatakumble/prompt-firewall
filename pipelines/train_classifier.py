#!/usr/bin/env python3
"""
Train Prompt Firewall injection classifier end-to-end.

Steps:
  1. Load versioned train/val/test splits from the data pipeline
  2. TF-IDF + Logistic Regression baseline (MLflow run_name=baseline)
  3. DistilBERT fine-tuning with class weights, checkpointing by val F1
  4. Hyperparameter grid: lr x decision threshold; select by security SLA
  5. Hold-out test evaluation + plots
  6. Fairlearn slice bias report on test set
  7. MLflow Model Registry (Staging) + HuggingFace artifacts on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from firewall.training.baseline import build_baseline
from firewall.training.bias_slices import slice_bias_report
from firewall.training.data import load_config, load_manifest, load_splits, labels_to_int
from firewall.training.distilbert_trainer import DistilBertClassifier
from firewall.training.metrics import meets_security_sla, metrics_from_arrays
from firewall.training.plots import save_confusion_matrix, save_pr_curve, save_roc_curve

logger = logging.getLogger("train_classifier")

# Security-first model selection (defend at Sandia review):
# missed injections are worse than false alarms → recall is the primary constraint.
MIN_RECALL = 0.85
MAX_FPR = 0.05
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
THRESHOLDS = [0.3, 0.5, 0.7]
REGISTERED_MODEL_NAME = "PromptFirewallClassifier"


def _setup_mlflow(config: dict[str, Any]) -> str:
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///./mlruns/mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "prompt-firewall-classifier")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def _register_model(classifier: DistilBertClassifier, config: dict[str, Any], metrics: dict) -> str | None:
    """Log to MLflow Model Registry and transition to Staging."""
    try:
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": classifier.model, "tokenizer": classifier.tokenizer},
            artifact_path="prompt-firewall-classifier",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={
                "dataset_version": config.get("dataset_version"),
                "decision_threshold": str(metrics.get("threshold", 0.5)),
            },
        )
        version = getattr(model_info, "registered_model_version", None)
        if version:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            logger.info("Registered %s v%s → Staging", REGISTERED_MODEL_NAME, version)
            return str(version)
    except Exception as exc:
        logger.warning("MLflow model registry push failed (%s); local artifacts still saved.", exc)
    return None


def run_hyperparam_search(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    max_length: int,
) -> tuple[DistilBertClassifier, dict[str, Any], list[dict[str, Any]]]:
    """Grid search lr (re-train) x threshold (inference); nested MLflow child runs."""
    candidates: list[dict[str, Any]] = []
    best_classifier: DistilBertClassifier | None = None
    best_selection: dict[str, Any] | None = None
    trained_by_lr: dict[float, DistilBertClassifier] = {}

    with mlflow.start_run(run_name="hyperparam_search"):
        mlflow.log_param("search_lrs", LEARNING_RATES)
        mlflow.log_param("search_thresholds", THRESHOLDS)
        mlflow.log_text(
            "Selection policy: recall >= 85% AND FPR <= 5%. "
            "Recall is primary — missed attacks are worse than false alarms in a security firewall.",
            "selection_policy.txt",
        )

        for lr in LEARNING_RATES:
            with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
                mlflow.log_params({"lr": lr, "epochs": epochs, "batch_size": batch_size})
                trial_clf = DistilBertClassifier(model_name=model_name, max_length=max_length)
                trial_clf.build(train_labels)
                history =                 trial_clf.train(
                    train_texts,
                    train_labels,
                    val_texts,
                    val_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                trained_by_lr[lr] = trial_clf
                mlflow.log_metrics(
                    {
                        "best_val_f1": max(h["f1"] for h in history),
                        "best_val_recall": max(h["recall"] for h in history),
                    }
                )

                for threshold in THRESHOLDS:
                    with mlflow.start_run(run_name=f"thresh_{threshold}", nested=True):
                        val_metrics = trial_clf.evaluate(
                            val_texts, val_labels, batch_size=batch_size, threshold=threshold
                        )
                        val_metrics["lr"] = lr
                        val_metrics["threshold"] = threshold
                        val_metrics["meets_sla"] = meets_security_sla(
                            val_metrics, min_recall=MIN_RECALL, max_fpr=MAX_FPR
                        )
                        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
                        mlflow.log_param("threshold", threshold)
                        candidates.append(val_metrics)

                        if val_metrics["meets_sla"]:
                            if best_selection is None or (
                                val_metrics["recall"] > best_selection["recall"]
                                or (
                                    val_metrics["recall"] == best_selection["recall"]
                                    and val_metrics["fpr"] < best_selection["fpr"]
                                )
                            ):
                                best_selection = val_metrics
                                best_classifier = trial_clf

    if best_classifier is None:
        logger.warning("No config met SLA (recall>=85%%, FPR<=5%%); picking highest recall.")
        best_selection = max(candidates, key=lambda m: (m["recall"], -m["fpr"]))
        best_classifier = trained_by_lr[best_selection["lr"]]

    assert best_classifier is not None and best_selection is not None
    return best_classifier, best_selection, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prompt Firewall injection classifier.")
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Subset data, skip full hyperparam grid")
    parser.add_argument("--max-train", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(PROJECT_ROOT / args.config)
    train_cfg = config.get("training", {})
    clf_cfg = config["classifier"]

    epochs = args.epochs or train_cfg.get("epochs", 2)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    max_length = train_cfg.get("max_length", 128)
    model_name = clf_cfg["model_name"]
    model_path = PROJECT_ROOT / clf_cfg["model_path"]
    reports_dir = PROJECT_ROOT / config["paths"]["reports"]
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    max_train = args.max_train or (2000 if args.quick else None)
    max_val = 500 if args.quick else None
    max_test = 500 if args.quick else None

    manifest = load_manifest(config)
    train_df, val_df, test_df = load_splits(
        config, max_train=max_train, max_val=max_val, max_test=max_test
    )
    train_texts = train_df["text"].astype(str).tolist()
    train_labels = labels_to_int(train_df["label"])
    val_texts = val_df["text"].astype(str).tolist()
    val_labels = labels_to_int(val_df["label"])
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = labels_to_int(test_df["label"])

    _setup_mlflow(config)
    summary: dict[str, Any] = {
        "dataset_version": manifest.get("version", config.get("dataset_version")),
        "manifest_checksum": manifest.get("artifacts", {}),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

    # --- Step 2: Baseline ---
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params(
            {
                "model_type": "tfidf_logistic_regression",
                "dataset_version": summary["dataset_version"],
            }
        )
        baseline = build_baseline()
        baseline.fit(train_texts, train_labels)
        baseline_val = baseline.evaluate(val_texts, val_labels)
        baseline_test = baseline.evaluate(test_texts, test_labels)
        mlflow.log_metrics({f"val_{k}": v for k, v in baseline_val.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"test_{k}": v for k, v in baseline_test.items() if isinstance(v, (int, float))})
        baseline.save(str(reports_dir / "baseline_model.joblib"))
        summary["baseline_val"] = baseline_val
        summary["baseline_test"] = baseline_test
        logger.info("Baseline test recall=%.3f FPR=%.3f", baseline_test["recall"], baseline_test["fpr"])

    # --- Steps 3-4: DistilBERT + hyperparameter search ---
    if args.quick:
        with mlflow.start_run(run_name="distilbert_quick"):
            quick_lr = LEARNING_RATES[0]
            mlflow.log_params({"model_name": model_name, "epochs": epochs, "batch_size": batch_size, "lr": quick_lr})
            classifier = DistilBertClassifier(model_name=model_name, max_length=max_length)
            classifier.build(train_labels)
            classifier.train(
                train_texts, train_labels, val_texts, val_labels,
                epochs=epochs, batch_size=batch_size, lr=quick_lr,
            )
            selection = classifier.evaluate(val_texts, val_labels, batch_size=batch_size, threshold=0.5)
            selection["lr"] = quick_lr
            selection["threshold"] = 0.5
            selection["meets_sla"] = meets_security_sla(selection, min_recall=MIN_RECALL, max_fpr=MAX_FPR)
            candidates = [selection]
    else:
        classifier, selection, candidates = run_hyperparam_search(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
        )

    chosen_threshold = float(selection["threshold"])
    chosen_lr = float(selection["lr"])
    summary["hyperparam_candidates"] = candidates
    summary["selected_config"] = selection

    with mlflow.start_run(run_name="champion"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "lr": chosen_lr,
                "threshold": chosen_threshold,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "dataset_version": summary["dataset_version"],
            }
        )

        # --- Step 5: Hold-out test evaluation ---
        test_probs = classifier.predict_proba(test_texts, batch_size=batch_size)
        test_pred = [1 if p >= chosen_threshold else 0 for p in test_probs]
        test_metrics = metrics_from_arrays(test_labels, test_pred, test_probs)
        test_metrics["threshold"] = chosen_threshold
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})

        save_confusion_matrix(
            test_labels, test_pred, figures_dir / "test_confusion_matrix.png", title="Test Set"
        )
        save_pr_curve(test_labels, test_probs, figures_dir / "test_pr_curve.png", title="Test PR Curve")
        save_roc_curve(test_labels, test_probs, figures_dir / "test_roc_curve.png", title="Test ROC Curve")
        mlflow.log_artifacts(str(figures_dir))

        # --- Step 6: Bias slicing on test set ---
        bias_report = slice_bias_report(
            test_df, test_labels, test_probs, threshold=chosen_threshold
        )
        bias_path = reports_dir / "bias_report.json"
        with open(bias_path, "w", encoding="utf-8") as handle:
            json.dump(bias_report, handle, indent=2)
        mlflow.log_artifact(str(bias_path))

        # Also write classifier-specific bias path for Airflow script
        bias_classifier_path = PROJECT_ROOT / "data" / "bias" / "bias_report_classifier.json"
        bias_classifier_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bias_classifier_path, "w", encoding="utf-8") as handle:
            json.dump(bias_report, handle, indent=2)

        # --- Step 7: Save artifacts + registry ---
        model_path.mkdir(parents=True, exist_ok=True)
        classifier.save(str(model_path))

        selection_meta = {
            "threshold": chosen_threshold,
            "lr": chosen_lr,
            "test_metrics": test_metrics,
            "selection_policy": bias_report.get("selection_policy"),
        }
        with open(model_path / "selection.json", "w", encoding="utf-8") as handle:
            json.dump(selection_meta, handle, indent=2)

        registry_version = _register_model(classifier, config, test_metrics)

        train_metrics_path = reports_dir / "train_metrics.json"
        summary.update(
            {
                "selected_threshold": chosen_threshold,
                "selected_lr": chosen_lr,
                "test_metrics": test_metrics,
                "bias_flagged": bias_report.get("flagged_disparities", []),
                "registry_version": registry_version,
            }
        )
        with open(train_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        mlflow.log_artifact(str(train_metrics_path))

    print(json.dumps(summary, indent=2))
    print(f"Champion model saved to {model_path}")


if __name__ == "__main__":
    main()
