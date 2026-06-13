"""Airflow DAG for prompt firewall data + model pipeline.

Reference flow (yashichawla/MLOps-Project):
  dvc_pull -> ensure_dirs -> ingest -> validate -> [report_status, enforce_policy]
  -> dvc_push_data -> train -> evaluate -> regression_gate -> dvc_push_final -> email_success
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator, get_current_context
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
TRAIN_PARQUET = REPO_ROOT / "data" / "curated" / "v1.0" / "splits" / "train.parquet"
METRICS_DIR = REPO_ROOT / "data" / "metrics"
BASELINE_SCHEMA = METRICS_DIR / "schema" / "baseline" / "schema.json"
SCRIPT_GE = REPO_ROOT / "scripts" / "ge_runner.py"
EMAIL_RECIPIENTS = os.environ.get("AIRFLOW_SMTP_USER", "admin@example.com")
SMTP_CONFIGURED = bool(os.environ.get("AIRFLOW_SMTP_USER") and os.environ.get("AIRFLOW_SMTP_PASSWORD"))


def _run(cmd: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


@dag(
    dag_id="firewall_ml_pipeline_v1",
    description="Ingest, validate, version (DVC), train, evaluate, and regression-gate firewall models",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={
        "owner": "akshatakumble",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["firewall", "mlops", "security", "dvc"],
)
def firewall_ml_pipeline_v1():
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"""
            set -euo pipefail
            cd "{REPO_ROOT}"
            python -m dvc checkout -f || true
            python -m dvc pull -v -f || true
        """,
    )

    ensure_dirs = BashOperator(
        task_id="ensure_dirs",
        bash_command=f'cd "{REPO_ROOT}" && python scripts/ensure_pipeline_dirs.py',
    )

    ingest_data = BashOperator(
        task_id="ingest_datasets",
        bash_command=f'cd "{REPO_ROOT}" && python scripts/preprocess_datasets.py --config config/app.yaml',
    )

    @task
    def validate_curated_data() -> dict:
        """Run GE baseline + validate; return metrics for emails and gating."""
        run_date = get_current_context()["ds_nodash"]
        train_path = str(TRAIN_PARQUET)
        if not TRAIN_PARQUET.exists():
            raise AirflowFailException(f"Missing training split: {TRAIN_PARQUET}")

        if not BASELINE_SCHEMA.exists():
            baseline = _run(
                ["python", str(SCRIPT_GE), "baseline", "--input", train_path, "--date", run_date],
                timeout=600,
            )
            if baseline.returncode != 0:
                raise AirflowFailException(f"Baseline creation failed: {baseline.stderr}")

        result = _run(
            [
                "python",
                str(SCRIPT_GE),
                "validate",
                "--input",
                train_path,
                "--baseline_schema",
                str(BASELINE_SCHEMA),
                "--date",
                run_date,
            ],
            timeout=600,
        )
        stats_path = METRICS_DIR / "stats" / run_date / "stats.json"
        anomalies_path = METRICS_DIR / "validation" / run_date / "anomalies.json"

        if result.returncode != 0:
            raise AirflowFailException(
                f"Validation failed (exit {result.returncode}). stderr={result.stderr[-500:]}"
            )

        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        with open(anomalies_path, encoding="utf-8") as handle:
            anomalies = json.load(handle)

        return {
            **stats,
            "hard_fail": anomalies.get("hard_fail", []),
            "soft_warn": anomalies.get("soft_warn", []),
            "report_paths": [str(anomalies_path), str(stats_path)],
        }

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def report_validation_status(metrics: dict | None) -> None:
        if not metrics:
            logger.error("Validation metrics missing.")
            return
        hard = metrics.get("hard_fail") or []
        soft = metrics.get("soft_warn") or []
        if hard:
            logger.error("Validation HARD FAIL: %s", hard)
        elif soft:
            logger.warning("Validation passed with warnings: %s", soft)
        else:
            logger.info("Validation passed cleanly: rows=%s", metrics.get("row_count"))

    @task
    def enforce_validation_policy(metrics: dict | None) -> None:
        if not metrics:
            raise AirflowFailException("Validation metrics missing.")
        hard = metrics.get("hard_fail") or []
        if hard:
            raise AirflowFailException(f"Validation hard-failed: {hard}")

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def verify_bias_report() -> dict:
        bias_path = REPO_ROOT / "data" / "registry" / "v1.0" / "bias_report.json"
        manifest_path = REPO_ROOT / "data" / "registry" / "v1.0" / "manifest.json"
        if not bias_path.exists() or not manifest_path.exists():
            raise AirflowFailException("Missing registry manifest or bias report after ingest.")
        with open(bias_path, encoding="utf-8") as handle:
            bias = json.load(handle)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        after = bias.get("after_mitigation", bias)
        fairlearn = after.get("fairlearn_metrics", {})
        attack_disparity = fairlearn.get("by_attack_type", {}).get("disparity")
        return {
            "bias_sources": len(after.get("by_source", {})),
            "attack_slices": len(after.get("by_attack_type", {})),
            "fairlearn_attack_disparity": attack_disparity,
            "warnings_count": len(after.get("warnings", [])),
            "manifest_version": manifest.get("version"),
            "training_rows": manifest.get("actual_training_rows"),
        }

    dvc_push_data = BashOperator(
        task_id="dvc_push_data",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        bash_command=f"""
            set -euo pipefail
            cd "{REPO_ROOT}"
            python -m dvc status -c || true
            python -m dvc commit ingest validate -f || true
            python -m dvc push -v
        """,
    )

    train_classifier = BashOperator(
        task_id="train_classifier",
        bash_command=f'cd "{REPO_ROOT}" && python pipelines/train_classifier.py --config config/app.yaml --epochs 2',
    )

    evaluate_firewall = BashOperator(
        task_id="evaluate_firewall",
        bash_command=(
            f'cd "{REPO_ROOT}" && python pipelines/evaluate_firewall.py '
            "--config config/app.yaml --benchmark salad-data"
        ),
    )

    compute_bias_slices = BashOperator(
        task_id="compute_bias_slices",
        bash_command=f'cd "{REPO_ROOT}" && python scripts/bias_detection.py --config config/app.yaml --parquet data/curated/v1.0/benchmarks/salad-data_normalized.parquet --output data/bias/bias_report_classifier.json',
    )

    regression_gate = BashOperator(
        task_id="regression_gate",
        bash_command=f'cd "{REPO_ROOT}" && python scripts/eval_regression_gate.py',
    )

    red_team_gate = BashOperator(
        task_id="red_team_gate",
        bash_command=f'cd "{REPO_ROOT}" && python scripts/red_team_eval.py',
    )

    dvc_push_final = BashOperator(
        task_id="dvc_push_final",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        bash_command=f"""
            set -euo pipefail
            cd "{REPO_ROOT}"
            python -m dvc commit train evaluate -f || true
            python -m dvc push -v
        """,
    )

    def send_success_email(**context):
        if not SMTP_CONFIGURED:
            raise AirflowSkipException("SMTP not configured; skipping success email.")
        ti = context["ti"]
        metrics = ti.xcom_pull(task_ids="validate_curated_data") or {}
        bias = ti.xcom_pull(task_ids="verify_bias_report") or {}
        html = f"""
        <h3>Prompt Firewall pipeline succeeded</h3>
        <p><b>Run ID:</b> {context.get('run_id')}</p>
        <p><b>Rows:</b> {metrics.get('row_count')}</p>
        <p><b>Injection rate:</b> {metrics.get('injection_rate')}</p>
        <p><b>Training rows (manifest):</b> {bias.get('training_rows')}</p>
        <p><b>Soft warnings:</b> {metrics.get('soft_warn')}</p>
        """
        from airflow.utils.email import send_email_smtp

        send_email_smtp(
            to=[EMAIL_RECIPIENTS],
            subject="Prompt Firewall DAG succeeded",
            html_content=html,
        )

    email_success = PythonOperator(
        task_id="email_success",
        python_callable=send_success_email,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    email_failure = EmailOperator(
        task_id="email_failure",
        to=EMAIL_RECIPIENTS if SMTP_CONFIGURED else "admin@example.com",
        subject="Prompt Firewall DAG failed",
        html_content="""
        <h3>Prompt Firewall pipeline failed</h3>
        <p>Check the Airflow UI logs for the failed task.</p>
        <p>Run ID: {{ run_id }}</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    metrics = validate_curated_data()
    report_task = report_validation_status(metrics)
    enforce_task = enforce_validation_policy(metrics)
    bias_task = verify_bias_report()

    dvc_pull >> ensure_dirs >> ingest_data >> metrics
    metrics >> [report_task, enforce_task, bias_task]
    [enforce_task, bias_task] >> dvc_push_data >> train_classifier >> evaluate_firewall
    evaluate_firewall >> [compute_bias_slices, regression_gate, red_team_gate]
    [compute_bias_slices, regression_gate, red_team_gate] >> dvc_push_final >> email_success
    [dvc_pull, ensure_dirs, ingest_data, metrics, enforce_task, dvc_push_data,
     train_classifier, evaluate_firewall, compute_bias_slices, regression_gate,
     red_team_gate, dvc_push_final] >> email_failure


firewall_ml_pipeline_v1()
