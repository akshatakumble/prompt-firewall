#!/usr/bin/env bash
###############################################################################
# One-time GCP setup for automated Cloud Run deployment via GitHub Actions.
#
# Provisions, idempotently:
#   • Enables required APIs (run, cloudbuild, artifactregistry, iam,
#     iamcredentials, secretmanager, storage, monitoring, logging)
#   • Artifact Registry Docker repo
#   • Runtime service account (what the Cloud Run service runs as)
#   • Deployer service account (what GitHub Actions impersonates)
#   • Workload Identity Federation pool + provider bound to your GitHub repo
#   • Secret Manager secret for the Groq API key
#
# Run it in Google Cloud Shell (recommended — clean, authenticated environment)
# or any machine with gcloud + an authenticated owner/editor account:
#
#   bash deploy/setup_gcp_deploy.sh
#
# Override defaults with env vars, e.g.:
#   PROJECT_ID=my-proj GITHUB_REPO=me/my-repo bash deploy/setup_gcp_deploy.sh
###############################################################################
set -euo pipefail

# ── configuration (override via env) ─────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-prompt-firewall-mlops}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-prompt-firewall-models}"
SERVICE="${SERVICE:-prompt-firewall-api}"
GITHUB_REPO="${GITHUB_REPO:-akshatakumble/prompt-firewall}"   # owner/name
BUCKET="${BUCKET:-prompt-firewall-mlops-dvc}"

RUNTIME_SA="prompt-firewall-run"
DEPLOYER_SA="prompt-firewall-deployer"
POOL="github-pool"
PROVIDER="github-provider"
SECRET_NAME="groq-api-key"

RUNTIME_SA_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA_EMAIL="${DEPLOYER_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Project: ${PROJECT_ID}  Region: ${REGION}  GitHub repo: ${GITHUB_REPO}"
gcloud config set project "${PROJECT_ID}" >/dev/null

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
echo "==> Project number: ${PROJECT_NUMBER}"

# ── 1. enable APIs ───────────────────────────────────────────────────────────
echo "==> Enabling APIs (this can take a minute)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com

# ── 2. Artifact Registry ─────────────────────────────────────────────────────
echo "==> Ensuring Artifact Registry repo: ${REPO}"
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker --location="${REGION}" \
    --description="Prompt Firewall API images"
fi

# ── 3. runtime service account (Cloud Run identity) ──────────────────────────
echo "==> Ensuring runtime SA: ${RUNTIME_SA_EMAIL}"
gcloud iam service-accounts describe "${RUNTIME_SA_EMAIL}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${RUNTIME_SA}" \
    --display-name="Prompt Firewall Cloud Run runtime"

# read model from GCS, read the Groq secret, write logs/metrics
for ROLE in \
  roles/storage.objectViewer \
  roles/secretmanager.secretAccessor \
  roles/logging.logWriter \
  roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${RUNTIME_SA_EMAIL}" --role="${ROLE}" \
    --condition=None >/dev/null
done

# ── 4. deployer service account (impersonated by GitHub Actions) ─────────────
echo "==> Ensuring deployer SA: ${DEPLOYER_SA_EMAIL}"
gcloud iam service-accounts describe "${DEPLOYER_SA_EMAIL}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${DEPLOYER_SA}" \
    --display-name="Prompt Firewall GitHub deployer"

for ROLE in \
  roles/run.admin \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.writer \
  roles/storage.admin \
  roles/iam.serviceAccountUser \
  roles/secretmanager.admin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER_SA_EMAIL}" --role="${ROLE}" \
    --condition=None >/dev/null
done

# deployer must be able to actAs the runtime SA when deploying Cloud Run
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA_EMAIL}" \
  --member="serviceAccount:${DEPLOYER_SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser" >/dev/null

# ── 5. Workload Identity Federation (keyless GitHub → GCP) ───────────────────
echo "==> Ensuring Workload Identity pool: ${POOL}"
gcloud iam workload-identity-pools describe "${POOL}" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "${POOL}" \
    --location=global --display-name="GitHub Actions pool"

echo "==> Ensuring OIDC provider: ${PROVIDER}"
if ! gcloud iam workload-identity-pools providers describe "${PROVIDER}" \
      --location=global --workload-identity-pool="${POOL}" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" \
    --location=global --workload-identity-pool="${POOL}" \
    --display-name="GitHub OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
fi

POOL_ID="$(gcloud iam workload-identity-pools describe "${POOL}" \
  --location=global --format='value(name)')"
PROVIDER_RESOURCE="${POOL_ID}/providers/${PROVIDER}"

# allow only this GitHub repo to impersonate the deployer SA
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" >/dev/null

# ── 6. Secret Manager: Groq API key ──────────────────────────────────────────
echo "==> Ensuring secret: ${SECRET_NAME}"
if ! gcloud secrets describe "${SECRET_NAME}" >/dev/null 2>&1; then
  gcloud secrets create "${SECRET_NAME}" --replication-policy=automatic
  # placeholder version so Cloud Run --update-secrets resolves; replace with real key below.
  printf '' | gcloud secrets versions add "${SECRET_NAME}" --data-file=-
fi
echo "    To set the real key:  echo -n 'gsk_...' | gcloud secrets versions add ${SECRET_NAME} --data-file=-"

# ── done: emit the GitHub config to set ──────────────────────────────────────
cat <<EOF

============================================================================
 ✅ GCP setup complete. Add these to your GitHub repo
    (Settings → Secrets and variables → Actions):

   REPOSITORY VARIABLES (Variables tab):
     GCP_PROJECT_ID          = ${PROJECT_ID}
     GCP_REGION              = ${REGION}
     GCP_ARTIFACT_REPO       = ${REPO}
     CLOUD_RUN_SERVICE       = ${SERVICE}
     GCP_WORKLOAD_IDENTITY_PROVIDER = ${PROVIDER_RESOURCE}
     GCP_DEPLOYER_SA         = ${DEPLOYER_SA_EMAIL}
     GCP_RUNTIME_SA          = ${RUNTIME_SA_EMAIL}

 No service-account JSON key is created or stored — auth is keyless via WIF.
============================================================================
EOF
