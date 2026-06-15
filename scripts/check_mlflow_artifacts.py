#!/usr/bin/env python3
from pathlib import Path

from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
client = MlflowClient(f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}")

for run in client.search_runs(experiment_ids=["1"]):
    if run.info.run_name != "champion":
        continue
    run_id = run.info.run_id
    print("run_id:", run_id)
    print("artifact_uri:", run.info.artifact_uri)
    artifact_dir = PROJECT_ROOT / "mlruns" / "1" / run_id / "artifacts"
    print("on_disk:", artifact_dir.exists())
    if artifact_dir.exists():
        for p in sorted(artifact_dir.rglob("*")):
            if p.is_file():
                print("  ", p.relative_to(artifact_dir))
    try:
        listed = client.list_artifacts(run_id)
        print("mlflow list_artifacts:", [a.path for a in listed])
    except Exception as exc:
        print("list_artifacts error:", exc)
