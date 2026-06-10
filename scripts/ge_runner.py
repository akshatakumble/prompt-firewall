#!/usr/bin/env python3
"""Great Expectations-style validation runner for curated datasets."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("ge_runner")

REQUIRED_COLUMNS = ("id", "text", "label", "source", "attack_type")
VALID_LABELS = frozenset({"INJECTION", "BENIGN"})


def generate_baseline_schema(df: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "required_columns": list(REQUIRED_COLUMNS),
        "columns": {col: str(df[col].dtype) for col in df.columns if col in df.columns},
        "valid_labels": sorted(VALID_LABELS),
        "row_count": len(df),
    }


def validate_against_schema(df: pd.DataFrame, baseline: dict) -> list[str]:
    failures: list[str] = []
    for column in baseline.get("required_columns", REQUIRED_COLUMNS):
        if column not in df.columns:
            failures.append(f"missing_column:{column}")
    return failures


def validate_dataframe(df: pd.DataFrame, baseline: dict | None = None) -> dict:
    anomalies = {"hard_fail": [], "soft_warn": [], "info": []}
    stats = {
        "row_count": len(df),
        "null_prompts": int(df["text"].isna().sum()) if "text" in df.columns else 0,
        "duplicates": int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else 0,
        "unknown_label_rate": 0.0,
        "text_len_min": int(df["text"].str.len().min()) if "text" in df.columns and len(df) else 0,
        "text_len_max": int(df["text"].str.len().max()) if "text" in df.columns and len(df) else 0,
    }

    if baseline is not None:
        schema_failures = validate_against_schema(df, baseline)
        if schema_failures:
            anomalies["hard_fail"].extend(schema_failures)

    if stats["null_prompts"] > 0:
        anomalies["hard_fail"].append("null_prompts_detected")
    if stats["duplicates"] > 0:
        anomalies["soft_warn"].append("duplicate_prompts_detected")

    if "label" in df.columns:
        unknown = ~df["label"].isin(VALID_LABELS)
        stats["unknown_label_rate"] = float(unknown.mean())
        stats["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
        if stats["unknown_label_rate"] > 0:
            anomalies["hard_fail"].append("unknown_labels_detected")

        injection_rate = stats["label_counts"].get("INJECTION", 0) / stats["row_count"] if stats["row_count"] else 0
        stats["injection_rate"] = round(injection_rate, 4)
        if stats["row_count"] >= 100:
            if injection_rate < 0.35 or injection_rate > 0.65:
                anomalies["soft_warn"].append("label_imbalance_detected")

    if "source" in df.columns:
        stats["source_counts"] = {
            str(k): int(v) for k, v in df["source"].value_counts().head(10).to_dict().items()
        }

    if stats["text_len_max"] > 8000:
        anomalies["soft_warn"].append("very_long_prompts_detected")

    return {"stats": stats, "anomalies": anomalies}


def write_artifacts(result: dict, date: str) -> tuple[Path, Path]:
    stats_dir = PROJECT_ROOT / "data" / "metrics" / "stats" / date
    val_dir = PROJECT_ROOT / "data" / "metrics" / "validation" / date
    stats_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / "stats.json"
    anomalies_path = val_dir / "anomalies.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(result["stats"], handle, indent=2)
    with open(anomalies_path, "w", encoding="utf-8") as handle:
        json.dump(result["anomalies"], handle, indent=2)
    return stats_path, anomalies_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "validate"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--baseline_schema", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)

    if args.mode == "baseline":
        schema_dir = PROJECT_ROOT / "data" / "metrics" / "schema" / "baseline"
        schema_dir.mkdir(parents=True, exist_ok=True)
        schema_path = schema_dir / "schema.json"
        schema = generate_baseline_schema(df)
        with open(schema_path, "w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2)
        logger.info("Baseline schema written to %s", schema_path)
        result = validate_dataframe(df, schema)
        write_artifacts(result, args.date)
        print(f"Baseline created: {schema_path}")
        return

    baseline = None
    baseline_path = Path(args.baseline_schema) if args.baseline_schema else (
        PROJECT_ROOT / "data" / "metrics" / "schema" / "baseline" / "schema.json"
    )
    if not baseline_path.exists():
        print(f"Baseline schema missing: {baseline_path}")
        raise SystemExit(2)

    with open(baseline_path, encoding="utf-8") as handle:
        baseline = json.load(handle)

    result = validate_dataframe(df, baseline)
    stats_path, anomalies_path = write_artifacts(result, args.date)

    if result["anomalies"]["hard_fail"]:
        logger.error("Validation FAILED: %s", result["anomalies"]["hard_fail"])
        raise SystemExit(1)

    logger.info("Validation passed. stats=%s anomalies=%s", stats_path, anomalies_path)
    print("Validation passed.")


if __name__ == "__main__":
    main()
