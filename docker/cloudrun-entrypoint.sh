#!/usr/bin/env bash
# Cloud Run container entrypoint.
#  1. Pull the latest classifier bundle from the GCS model registry.
#  2. Launch the FastAPI app with uvicorn, binding to Cloud Run's $PORT.
#
# Model fetch is best-effort (MODEL_FETCH_OPTIONAL=1): if GCS is unreachable the
# API still boots in rules-only fallback mode rather than crash-looping, and the
# /health endpoint will report classifier_loaded=false so monitoring can alert.
set -euo pipefail

: "${PORT:=8080}"
: "${MODEL_FETCH_OPTIONAL:=1}"
export MODEL_FETCH_OPTIONAL

echo "[entrypoint] Fetching model from ${MODEL_GCS_URI:-gs://prompt-firewall-mlops-dvc/models/v2}"
python scripts/fetch_model.py || echo "[entrypoint] model fetch returned non-zero; continuing"

echo "[entrypoint] Starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn firewall.api.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WEB_CONCURRENCY:-1}"
