#!/usr/bin/env python3
"""Great Expectations validation runner for curated firewall datasets."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import great_expectations as gx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("ge_runner")

SUITE_NAME = "firewall_training_suite"
REQUIRED_COLUMNS = ("id", "text", "label", "source", "attack_type")
VALID_LABELS = frozenset({"INJECTION", "BENIGN"})
SCHEMA_DIR = PROJECT_ROOT / "data" / "metrics" / "schema" / "baseline"
SUITE_PATH = SCHEMA_DIR / "expectation_suite.json"
SCHEMA_PATH = SCHEMA_DIR / "schema.json"


def _expectation_type(result) -> str:
    config = result.expectation_config
    return getattr(config, "expectation_type", None) or getattr(config, "type", "unknown")


def apply_expectations(ge_df, row_count: int):
    """Attach expectations to a Great Expectations PandasDataset (GE 0.18 API)."""
    for column in REQUIRED_COLUMNS:
        ge_df.expect_column_to_exist(column)

    ge_df.expect_column_values_to_not_be_null("text")
    ge_df.expect_column_values_to_not_be_null("label")
    ge_df.expect_column_values_to_be_in_set("label", sorted(VALID_LABELS))
    ge_df.expect_column_values_to_be_unique("text")

    if row_count >= 100:
        min_rows = max(100, int(row_count * 0.5))
        max_rows = int(row_count * 1.5) + 1000
        ge_df.expect_table_row_count_to_be_between(min_value=min_rows, max_value=max_rows)

    return ge_df


def ge_results_to_artifacts(df: pd.DataFrame, validation_result) -> dict:
    """Map Great Expectations validation output to stats + anomalies JSON."""
    anomalies = {"hard_fail": [], "soft_warn": [], "info": []}
    text_lengths = df["text"].dropna().astype(str).str.len() if "text" in df.columns else pd.Series(dtype=int)
    stats = {
        "row_count": len(df),
        "null_prompts": int(df["text"].isna().sum()) if "text" in df.columns else 0,
        "duplicates": int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else 0,
        "unknown_label_rate": 0.0,
        "text_len_min": int(text_lengths.min()) if not text_lengths.empty else 0,
        "text_len_max": int(text_lengths.max()) if not text_lengths.empty else 0,
        "ge_success": validation_result.success,
        "ge_evaluated_expectations": validation_result.statistics.get("evaluated_expectations", 0),
        "ge_successful_expectations": validation_result.statistics.get("successful_expectations", 0),
        "ge_unsuccessful_expectations": validation_result.statistics.get("unsuccessful_expectations", 0),
    }

    if "label" in df.columns:
        unknown = ~df["label"].isin(VALID_LABELS)
        stats["unknown_label_rate"] = float(unknown.mean())
        stats["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
        injection_rate = stats["label_counts"].get("INJECTION", 0) / stats["row_count"] if stats["row_count"] else 0
        stats["injection_rate"] = round(injection_rate, 4)
        if stats["row_count"] >= 100 and (injection_rate < 0.35 or injection_rate > 0.65):
            anomalies["soft_warn"].append("label_imbalance_detected")

    if "source" in df.columns:
        stats["source_counts"] = {
            str(k): int(v) for k, v in df["source"].value_counts().head(10).to_dict().items()
        }

    if stats["text_len_max"] > 8000:
        anomalies["soft_warn"].append("very_long_prompts_detected")

    for result in validation_result.results:
        exp_type = _expectation_type(result)
        success = result.success
        column = result.expectation_config.kwargs.get("column", "")
        key = f"ge:{exp_type}:{column}" if column else f"ge:{exp_type}"

        if success:
            anomalies["info"].append(key)
            continue

        if exp_type in {
            "expect_column_values_to_not_be_null",
            "expect_column_values_to_be_in_set",
            "expect_column_to_exist",
            "expect_column_values_to_be_unique",
        }:
            anomalies["hard_fail"].append(key)
        else:
            anomalies["soft_warn"].append(key)

    return {"stats": stats, "anomalies": anomalies}


def save_baseline_schema(df: pd.DataFrame, ge_df) -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    schema = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "validator": "great_expectations",
        "suite_name": SUITE_NAME,
        "required_columns": list(REQUIRED_COLUMNS),
        "columns": {col: str(df[col].dtype) for col in df.columns if col in df.columns},
        "valid_labels": sorted(VALID_LABELS),
        "row_count": len(df),
    }
    suite = ge_df.get_expectation_suite()
    suite_doc = {
        "expectation_suite_name": SUITE_NAME,
        "expectations": [exp.to_json_dict() for exp in suite.expectations],
    }
    with open(SCHEMA_PATH, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
    with open(SUITE_PATH, "w", encoding="utf-8") as handle:
        json.dump(suite_doc, handle, indent=2)


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


def run_validation(df: pd.DataFrame, *, baseline_mode: bool) -> dict:
    ge_df = apply_expectations(gx.from_pandas(df), len(df))
    if baseline_mode:
        save_baseline_schema(df, ge_df)

    validation_result = ge_df.validate()
    return ge_results_to_artifacts(df, validation_result)


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
        result = run_validation(df, baseline_mode=True)
        write_artifacts(result, args.date)
        logger.info("Great Expectations baseline suite written to %s", SUITE_PATH)
        print(f"Baseline created: {SCHEMA_PATH}")
        return

    if not SCHEMA_PATH.exists() or not SUITE_PATH.exists():
        print(f"Baseline missing. Run baseline mode first. Expected: {SCHEMA_PATH}")
        raise SystemExit(2)

    result = run_validation(df, baseline_mode=False)
    stats_path, anomalies_path = write_artifacts(result, args.date)

    if result["anomalies"]["hard_fail"]:
        logger.error("Great Expectations validation FAILED: %s", result["anomalies"]["hard_fail"])
        raise SystemExit(1)

    logger.info("Great Expectations validation passed. stats=%s anomalies=%s", stats_path, anomalies_path)
    print("Validation passed.")


if __name__ == "__main__":
    main()
