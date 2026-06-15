#!/usr/bin/env python3
"""
Upload a trained classifier bundle to GCP (GCS model registry).

ML model weights live in GCS (DVC remote). GCP Artifact Registry is optional
and used for Docker inference images — see scripts/setup_gcp_artifact_registry.ps1.

Usage:
  set GOOGLE_APPLICATION_CREDENTIALS=.secrets/gcp-key.json
  python scripts/push_model_gcp.py --version v2 --model-path models/classifier-v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "gs://prompt-firewall-mlops-dvc/models"


def _find_gcloud() -> str:
    import shutil

    found = shutil.which("gcloud")
    if found:
        return found
    win_default = Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd"
    if win_default.exists():
        return str(win_default)
    raise RuntimeError("gcloud CLI not found. Install Google Cloud SDK.")


def _run(cmd: list[str]) -> None:
    if cmd[0] == "gcloud":
        cmd = [_find_gcloud(), *cmd[1:]]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, shell=False)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_registry_manifest(
    *,
    version: str,
    model_path: Path,
    selection: dict | None,
    eval_report: dict | None,
    gate_report: dict | None,
) -> dict:
    files = sorted(p.name for p in model_path.iterdir() if p.is_file())
    return {
        "model_name": "PromptFirewallClassifier",
        "version": version,
        "gcs_prefix": f"{DEFAULT_BUCKET}/{version}/",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "model_files": files,
        "selection": selection,
        "salad_eval": {
            "recall": eval_report.get("recall") if eval_report else None,
            "fpr": eval_report.get("fpr") if eval_report else None,
            "operating_threshold": eval_report.get("operating_threshold") if eval_report else None,
            "support": eval_report.get("support") if eval_report else None,
        },
        "regression_gate_passed": gate_report.get("passed") if gate_report else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Push classifier bundle to GCS model registry.")
    parser.add_argument("--version", default="v2", help="Registry version tag (e.g. v2)")
    parser.add_argument("--model-path", default="models/classifier-v1")
    parser.add_argument("--bucket-prefix", default=DEFAULT_BUCKET)
    parser.add_argument("--eval-json", default="reports/eval_salad-data.json")
    parser.add_argument("--gate-json", default="reports/regression_gate.json")
    args = parser.parse_args()

    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        print(
            "Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON "
            "(e.g. .secrets/gcp-key.json).",
            file=sys.stderr,
        )
        return 1

    model_path = PROJECT_ROOT / args.model_path
    if not (model_path / "config.json").exists():
        print(f"Model not found: {model_path}. Copy v2 artifacts from Colab first.", file=sys.stderr)
        return 1

    selection = _load_json(model_path / "selection.json")
    eval_report = _load_json(PROJECT_ROOT / args.eval_json)
    gate_report = _load_json(PROJECT_ROOT / args.gate_json)

    if eval_report is None:
        print(f"Warning: missing {args.eval_json} — registry manifest will omit Salad metrics.")
    if gate_report is None:
        print(f"Warning: missing {args.gate_json} — registry manifest will omit gate status.")

    manifest = build_registry_manifest(
        version=args.version,
        model_path=model_path,
        selection=selection,
        eval_report=eval_report,
        gate_report=gate_report,
    )

    dest_prefix = f"{args.bucket_prefix.rstrip('/')}/{args.version}"
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / args.version
        shutil.copytree(model_path, bundle_dir)
        manifest_path = bundle_dir / "registry_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        _run(["gcloud", "storage", "cp", "-r", str(bundle_dir), f"{args.bucket_prefix.rstrip('/')}/"])

    local_manifest = PROJECT_ROOT / "data" / "registry" / "v1.0" / f"model_{args.version}_gcs.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(local_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"\nUploaded to {dest_prefix}/")
    print(f"Local copy: {local_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
