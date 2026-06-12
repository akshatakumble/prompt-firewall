#!/usr/bin/env python3
"""Evaluate firewall on held-out benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from firewall.data.bias_report import add_prompt_length_bucket
from firewall.data.model_bias import evaluate_slice_performance
from firewall.service import FirewallService


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_dataframe(firewall: FirewallService, df: pd.DataFrame) -> dict:
    tp = fp = tn = fn = 0
    for _, row in df.iterrows():
        result = firewall.inspect(row["text"])
        predicted_attack = result.action in ("BLOCK", "SANITIZE")
        actual_attack = row["label"] == "INJECTION"
        if predicted_attack and actual_attack:
            tp += 1
        elif predicted_attack and not actual_attack:
            fp += 1
        elif not predicted_attack and not actual_attack:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "total": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--benchmark", default="simsonsun")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    curated_dir = PROJECT_ROOT / config["paths"]["curated"]
    benchmark_path = curated_dir / "benchmarks" / f"{args.benchmark}_normalized.parquet"
    legacy_path = curated_dir / f"{args.benchmark}_normalized.parquet"
    if not benchmark_path.exists() and legacy_path.exists():
        benchmark_path = legacy_path

    if not benchmark_path.exists():
        print(f"Benchmark not found: {benchmark_path}. Run ingest first.")
        sys.exit(1)

    df = add_prompt_length_bucket(pd.read_parquet(benchmark_path))
    firewall = FirewallService()
    metrics = evaluate_dataframe(firewall, df)
    metrics["slice_performance"] = evaluate_slice_performance(
        df,
        predict_fn=lambda text: firewall.inspect(text).action in ("BLOCK", "SANITIZE"),
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"eval_{args.benchmark}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
