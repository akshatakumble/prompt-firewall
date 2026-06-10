#!/usr/bin/env python3
"""Download, normalize, version, and split jailbreak datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from firewall.data.bias_report import build_bias_report
from firewall.data.cleaning import add_token_counts, clean_dataframe
from firewall.data.manifest import build_manifest, write_manifest
from firewall.data.normalizers import BENCHMARK_LOADERS, TRAIN_LOADERS
from firewall.data.sampling import balanced_sample, create_stratified_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest_dataset")


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ingest_training_data(
    config: dict,
    *,
    target_samples: int,
    raw_dir: Path,
    curated_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    artifacts: dict[str, Path] = {}
    frames: list[pd.DataFrame] = []

    for name in config["datasets"]["train"]:
        loader = TRAIN_LOADERS.get(name)
        if loader is None:
            logger.warning("Unknown training dataset: %s", name)
            continue

        logger.info("Ingesting training dataset: %s", name)
        dataset_raw_dir = raw_dir / name
        df = loader(dataset_raw_dir)
        df = clean_dataframe(df)
        df = add_token_counts(df)

        dataset_path = curated_dir / f"{name}_normalized.parquet"
        df.to_parquet(dataset_path, index=False)
        artifacts[name] = dataset_path
        frames.append(df)
        logger.info("Saved %s rows to %s", len(df), dataset_path)

    if not frames:
        raise RuntimeError("No training datasets ingested. Check config and HF access.")

    combined = clean_dataframe(pd.concat(frames, ignore_index=True))
    logger.info(
        "Combined training pool: %s rows (%s)",
        len(combined),
        combined["label"].value_counts().to_dict(),
    )

    sampled = balanced_sample(combined, target_samples)
    logger.info(
        "Balanced sample: %s rows (%s)",
        len(sampled),
        sampled["label"].value_counts().to_dict(),
    )

    combined_path = curated_dir / "training_combined.parquet"
    sampled.to_parquet(combined_path, index=False)
    artifacts["training_combined"] = combined_path

    return sampled, artifacts


def ingest_benchmarks(
    config: dict,
    *,
    raw_dir: Path,
    curated_dir: Path,
) -> dict[str, Path]:
    benchmarks_dir = curated_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    for name in config["datasets"].get("eval_held_out", []):
        loader = BENCHMARK_LOADERS.get(name)
        if loader is None:
            logger.warning("Unknown benchmark dataset: %s", name)
            continue

        logger.info("Ingesting held-out benchmark: %s", name)
        dataset_raw_dir = raw_dir / name
        df = loader(dataset_raw_dir)
        df = clean_dataframe(df)
        df = add_token_counts(df)

        benchmark_path = benchmarks_dir / f"{name}_normalized.parquet"
        df.to_parquet(benchmark_path, index=False)
        artifacts[name] = benchmark_path
        logger.info("Saved %s benchmark rows to %s", len(df), benchmark_path)

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and normalize firewall datasets.")
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Balanced INJECTION/BENIGN target size (defaults to config.datasets.target_samples)",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    target_samples = args.target_samples or config["datasets"].get("target_samples", 60000)

    raw_dir = PROJECT_ROOT / config["paths"]["raw"]
    curated_dir = PROJECT_ROOT / config["paths"]["curated"]
    registry_dir = PROJECT_ROOT / config["paths"]["registry"]
    curated_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_artifacts = ingest_training_data(
        config,
        target_samples=target_samples,
        raw_dir=raw_dir,
        curated_dir=curated_dir,
    )

    splits_dir = curated_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = config["datasets"].get("splits", {})
    train_split, val_split, test_split = create_stratified_splits(
        train_df,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
    )

    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "validation.parquet"
    test_path = splits_dir / "test.parquet"
    train_split.to_parquet(train_path, index=False)
    val_split.to_parquet(val_path, index=False)
    test_split.to_parquet(test_path, index=False)

    benchmark_artifacts = ingest_benchmarks(
        config,
        raw_dir=raw_dir,
        curated_dir=curated_dir,
    )

    all_artifacts = {
        **train_artifacts,
        **benchmark_artifacts,
        "splits/train": train_path,
        "splits/validation": val_path,
        "splits/test": test_path,
    }

    bias_report = build_bias_report(train_df)
    bias_path = registry_dir / "bias_report.json"
    with open(bias_path, "w", encoding="utf-8") as handle:
        json.dump(bias_report, handle, indent=2)

    manifest = build_manifest(
        version=config["dataset_version"],
        artifacts=all_artifacts,
        splits={
            "train": train_split,
            "validation": val_split,
            "test": test_split,
        },
        project_root=PROJECT_ROOT,
        extra={
            "target_samples": target_samples,
            "actual_training_rows": len(train_df),
            "datasets": {
                "train": config["datasets"]["train"],
                "eval_held_out": config["datasets"].get("eval_held_out", []),
            },
            "bias_report_path": str(bias_path.relative_to(PROJECT_ROOT)),
        },
    )
    manifest_path = registry_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    logger.info(
        "Ingestion complete | train=%s val=%s test=%s | manifest=%s",
        len(train_split),
        len(val_split),
        len(test_split),
        manifest_path,
    )
    print(
        json.dumps(
            {
                "train_rows": len(train_split),
                "validation_rows": len(val_split),
                "test_rows": len(test_split),
                "target_samples": target_samples,
                "label_counts": train_df["label"].value_counts().to_dict(),
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
