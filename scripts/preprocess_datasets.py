#!/usr/bin/env python3
"""Wrapper to run dataset ingestion from Airflow or CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--target-samples", type=int, default=None)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipelines" / "ingest_dataset.py"),
        "--config",
        args.config,
    ]
    if args.target_samples is not None:
        cmd.extend(["--target-samples", str(args.target_samples)])

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
