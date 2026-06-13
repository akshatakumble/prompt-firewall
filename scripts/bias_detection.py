#!/usr/bin/env python3
"""
Standalone classifier bias detection (Fairlearn slicing).

Mirrors yashichawla/MLOps-Project scripts/bias_detection.py structure.
Can be called from Airflow after evaluate_firewall produces predictions.

Usage:
  python scripts/bias_detection.py \\
    --parquet data/curated/v1.0/benchmarks/salad-data_normalized.parquet \\
    --predictions-csv reports/salad_predictions.csv \\
    --output data/bias/bias_report_classifier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from firewall.data.bias_report import add_prompt_length_bucket
from firewall.training.bias_slices import slice_bias_report
from firewall.training.data import labels_to_int, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier bias detection via data slicing.")
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--parquet", required=True, help="Input parquet with labels and slice columns")
    parser.add_argument("--predictions-csv", help="CSV with columns: id or row index, score (0-1)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for scores")
    parser.add_argument("--output", default="data/bias/bias_report_classifier.json")
    parser.add_argument("--recall-drop-threshold", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    threshold = args.threshold or float(config.get("training", {}).get("decision_threshold", 0.5))

    df = add_prompt_length_bucket(pd.read_parquet(args.parquet))
    y_true = labels_to_int(df["label"])

    if args.predictions_csv:
        pred_df = pd.read_csv(args.predictions_csv)
        if "score" in pred_df.columns:
            y_scores = pred_df["score"].astype(float).tolist()
        elif "predicted_attack" in pred_df.columns:
            y_scores = pred_df["predicted_attack"].astype(float).tolist()
        else:
            raise SystemExit("predictions CSV must have 'score' or 'predicted_attack' column")
        if len(y_scores) != len(df):
            raise SystemExit(f"Row count mismatch: parquet={len(df)} predictions={len(y_scores)}")
    else:
        # Read from latest eval report if available
        eval_path = PROJECT_ROOT / config["paths"]["reports"] / "eval_salad-data.json"
        if not eval_path.exists():
            raise SystemExit("Provide --predictions-csv or run evaluate_firewall first.")
        with open(eval_path, encoding="utf-8") as handle:
            eval_report = json.load(handle)
        slice_perf = eval_report.get("slice_performance", {})
        report = {
            "source": "eval_salad-data.json",
            "operating_threshold": eval_report.get("operating_threshold"),
            "global_metrics": {
                "recall": eval_report.get("recall"),
                "fpr": eval_report.get("fpr"),
                "precision": eval_report.get("precision"),
            },
            "slice_performance": slice_perf,
            "flagged_disparities": slice_perf.get("flagged_disparities", []),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        print(f"Bias report saved to {out_path}")
        return

    report = slice_bias_report(
        df,
        y_true,
        y_scores,
        threshold=threshold,
        recall_drop_threshold=args.recall_drop_threshold,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Bias report saved to {out_path}")


if __name__ == "__main__":
    main()
