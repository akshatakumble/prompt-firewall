#!/usr/bin/env python3
"""Ensure data pipeline directories exist before Airflow or DVC runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ensure_pipeline_dirs")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "app.yaml"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dirs = [
        PROJECT_ROOT / config["paths"]["raw"],
        PROJECT_ROOT / config["paths"]["curated"],
        PROJECT_ROOT / config["paths"]["curated"] / "benchmarks",
        PROJECT_ROOT / config["paths"]["curated"] / "splits",
        PROJECT_ROOT / config["paths"]["registry"],
        PROJECT_ROOT / "data" / "metrics" / "stats",
        PROJECT_ROOT / "data" / "metrics" / "validation",
        PROJECT_ROOT / "data" / "metrics" / "schema" / "baseline",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "reports" / "figures",
        PROJECT_ROOT / "data" / "bias",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "mlruns",
        PROJECT_ROOT / "airflow_artifacts" / "logs",
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory: %s", path.relative_to(PROJECT_ROOT))

    summary = {
        "project_root": str(PROJECT_ROOT),
        "dataset_version": config.get("dataset_version"),
        "curated_dir": config["paths"]["curated"],
        "registry_dir": config["paths"]["registry"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
