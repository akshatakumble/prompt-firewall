#!/usr/bin/env bash
###############################################################################
# Cloud Monitoring: uptime check + alert policy for the deployed firewall API.
#
# Creates:
#   • an HTTPS uptime check that polls /health every 60s
#   • an alert policy that fires if the uptime check fails
#
# Cloud Run already streams stdout/stderr to Cloud Logging automatically (the
# entrypoint and uvicorn logs are visible in Logs Explorer), and the API exposes
# Prometheus metrics at /metrics. This adds active availability monitoring.
#
# Usage (after the service is deployed):
#   SERVICE_URL=https://prompt-firewall-api-xxxx-uc.a.run.app \
#     bash deploy/monitoring/setup_uptime_check.sh
###############################################################################
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-prompt-firewall-mlops}"
SERVICE="${SERVICE:-prompt-firewall-api}"
REGION="${REGION:-us-central1}"

# Resolve the service URL if not provided.
if [ -z "${SERVICE_URL:-}" ]; then
  SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
    --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')"
fi
HOST="${SERVICE_URL#https://}"
echo "==> Service host: ${HOST}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# ── 1. uptime check on /health ───────────────────────────────────────────────
echo "==> Creating uptime check (HTTPS /health, 60s)"
cat > /tmp/uptime.json <<EOF
{
  "displayName": "prompt-firewall-health",
  "monitoredResource": {
    "type": "uptime_url",
    "labels": { "host": "${HOST}", "project_id": "${PROJECT_ID}" }
  },
  "httpCheck": { "path": "/health", "port": 443, "useSsl": true, "validateSsl": true },
  "period": "60s",
  "timeout": "10s"
}
EOF
gcloud monitoring uptime create-config-from-json /tmp/uptime.json 2>/dev/null \
  || gcloud monitoring uptime create "prompt-firewall-health" \
       --resource-type=uptime-url --resource-labels=host="${HOST}",project_id="${PROJECT_ID}" \
       --path="/health" --port=443 --period=1 --timeout=10 \
  || echo "(uptime check may already exist — skipping)"

# ── 2. alert policy on uptime-check failure ──────────────────────────────────
echo "==> Creating alert policy (uptime check failed)"
cat > /tmp/alert.json <<'EOF'
{
  "displayName": "Prompt Firewall API down",
  "combiner": "OR",
  "conditions": [{
    "displayName": "Uptime check failing",
    "conditionThreshold": {
      "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\"",
      "comparison": "COMPARISON_LT",
      "thresholdValue": 1,
      "duration": "120s",
      "trigger": { "count": 1 },
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_NEXT_OLDER",
        "crossSeriesReducer": "REDUCE_COUNT_FALSE",
        "groupByFields": ["resource.label.host"]
      }]
    }
  }]
}
EOF
gcloud alpha monitoring policies create --policy-from-file=/tmp/alert.json 2>/dev/null \
  || echo "(alert policy may already exist, or 'gcloud alpha' not installed — create it from the JSON in the console if needed)"

echo ""
echo "Done. View in console:"
echo "  Uptime checks: https://console.cloud.google.com/monitoring/uptime?project=${PROJECT_ID}"
echo "  Logs:          https://console.cloud.google.com/run/detail/${REGION}/${SERVICE}/logs?project=${PROJECT_ID}"
