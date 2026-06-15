#!/usr/bin/env python3
"""List MLflow experiments, runs, and registered models."""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}")
client = MlflowClient()

print("=== EXPERIMENTS ===")
for exp in client.search_experiments():
    print(f"  id={exp.experiment_id}  name={exp.name}")

for exp in client.search_experiments():
    print(f'\n=== RUNS in "{exp.name}" ===')
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time ASC"])
    for run in runs:
        parent = run.data.tags.get("mlflow.parentRunId", "")
        prefix = "    > " if parent else "  "
        name = run.info.run_name or run.info.run_id
        print(f"{prefix}{name}  status={run.info.status}")

        key_params = {k: run.data.params[k] for k in sorted(run.data.params) if k in (
            "model_type", "model_name", "lr", "threshold", "epochs", "batch_size", "dataset_version"
        )}
        key_metrics = {
            k: round(float(run.data.metrics[k]), 4)
            for k in sorted(run.data.metrics)
            if any(k.startswith(p) for p in ("test_", "val_", "best_"))
        }
        if key_params:
            print(f"      params: {key_params}")
        if key_metrics:
            print(f"      metrics: {key_metrics}")

print("\n=== REGISTERED MODELS ===")
for rm in client.search_registered_models():
    print(f"  {rm.name}")
    for version in client.search_model_versions(f"name='{rm.name}'"):
        print(
            f"    v{version.version}  stage={version.current_stage}  "
            f"run_id={version.run_id[:8]}..."
        )

if not client.search_experiments():
    print("No experiments found. Copy mlruns/ from Colab if you trained there.", file=sys.stderr)
    sys.exit(1)
