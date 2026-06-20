# Prompt Firewall MLOps

End-to-end MLOps pipeline for a **Prompt Firewall** that detects jailbreak and prompt-injection attacks. Includes dataset ingestion, Great Expectations-style validation, DVC versioning, Airflow orchestration, classifier training, and regression gates.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/akshatakumble/prompt-firewall-mlops.git
cd prompt-firewall-mlops
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face read token for dataset download |
| `AIRFLOW_SMTP_USER` | Gmail address for Airflow alerts |
| `AIRFLOW_SMTP_PASSWORD` | Gmail app password (16 chars) |

### 3. Run unit tests

```bash
pytest -q
pytest tests/unit/test_ingest_pipeline.py tests/unit/test_ge_runner.py -q
```

### 4. DVC pipeline (local, no Airflow)

```bash
python scripts/ensure_pipeline_dirs.py
python -m dvc repro ingest validate
python -m dvc push
```

### 5. Airflow (full DAG)

```bash
mkdir -p airflow_artifacts/logs
docker compose -f docker-compose.airflow.yml run --rm airflow-init
docker compose -f docker-compose.airflow.yml up -d webserver scheduler
```

Open **http://localhost:8080** (admin / admin) and trigger **`firewall_ml_pipeline_v1`**.

## DAG flow

```
dvc_pull → ensure_dirs → ingest_datasets → validate_curated_data
  → [report_validation_status, enforce_validation_policy, verify_bias_report]
  → dvc_push_data → train_classifier → evaluate_firewall
  → regression_gate → dvc_push_final → email_success
```

## Repository structure

```
prompt-firewall-mlops/
├── dags/                  # Airflow DAG definitions
├── pipelines/             # Ingest, train, evaluate scripts
├── scripts/               # GE validation, preprocessing wrappers
├── src/firewall/          # Firewall service, data modules, rules
├── tests/                 # Unit + regression tests
├── config/                # app.yaml, rules.yaml
├── data/
│   ├── raw/               # Downloaded datasets (gitignored)
│   ├── curated/           # Normalized parquet (DVC-tracked)
│   ├── registry/          # Manifest + bias report (DVC-tracked)
│   └── metrics/           # Validation stats + anomalies
├── documents/             # Bias mitigation report
├── dvc.yaml               # DVC pipeline stages
└── docker-compose.airflow.yml
```

## Data versioning (DVC)

Default remote: **local storage** at `./dvc-storage` at repo root (gitignored; DVC resolves `../dvc-storage` from `.dvc/config`). No GCP required for local runs.

```bash
python -m dvc pull      # restore tracked artifacts
python -m dvc push      # upload after pipeline run
python -m dvc repro     # reproduce ingest → validate → train → evaluate
```

Optional GCS remote (`gcsremote`) is pre-configured in `.dvc/config` but disabled by default. See [DVC + GCP](#dvc--gcp-optional) below.

## Validation source of truth

`scripts/ge_runner.py` produces:

- `data/metrics/schema/baseline/schema.json`
- `data/metrics/stats/<YYYYMMDD>/stats.json`
- `data/metrics/validation/<YYYYMMDD>/anomalies.json`

The Airflow DAG reads these for gating and email reports.

## Datasets

| Dataset | Role |
|---|---|
| WildJailbreak | Training (download via HF) |
| Salad-Data | Held-out benchmark |

Ingestion: `python pipelines/ingest_dataset.py --config config/app.yaml`

## DVC + GCP (optional)

**You do not need GCP** for the data pipeline assignment if you:

1. Set `HF_TOKEN` so datasets re-download from Hugging Face, and
2. Use the local DVC remote (`./dvc-storage`) or commit `dvc.lock` so others can reproduce via `dvc repro`.

Enable GCS only if you want shared cloud storage for large artifacts:

1. Create bucket `gs://prompt-firewall-mlops-dvc/data`
2. Place service account JSON at `.secrets/gcp-key.json`
3. Uncomment `gcsfs` / `dvc[gs]` in `requirements-docker.txt`
4. Run: `dvc remote modify --local localstorage remote gcsremote` (or set `core.remote = gcsremote`)

## Cloud deployment, monitoring & retraining

The model is deployed to **GCP Cloud Run** with keyless GitHub Actions CI/CD
(Workload Identity Federation), scheduled drift/decay monitoring, and automated
threshold-triggered retraining. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the
full architecture and step-by-step replication.

```bash
bash deploy/setup_gcp_deploy.sh          # one-time GCP provisioning
# set the printed GitHub repo variables, then push to main → auto-deploys
```

## Makefile targets

| Target | Description |
|---|---|
| `make test` | Run pytest |
| `make airflow-init` | Initialize Airflow DB |
| `make airflow-up` | Start webserver + scheduler |
| `make dvc-repro` | Reproduce ingest + validate stages |

## Team

Group 16 — IE7374 Machine Learning Operations, Northeastern University.
