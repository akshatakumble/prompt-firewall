# Create GCP resources for Prompt Firewall artifact storage.
#
# ML models  -> GCS: gs://prompt-firewall-mlops-dvc/models/<version>/
# Docker API -> Artifact Registry (optional): prompt-firewall-models
#
# Prerequisites:
#   gcloud auth login   OR   service account key in .secrets/gcp-key.json
#   Project: prompt-firewall-mlops

$ErrorActionPreference = "Stop"
$Project = "prompt-firewall-mlops"
$Region = "us-central1"
$Bucket = "gs://prompt-firewall-mlops-dvc"
$Repo = "prompt-firewall-models"

if ($env:GOOGLE_APPLICATION_CREDENTIALS) {
    gcloud auth activate-service-account --key-file=$env:GOOGLE_APPLICATION_CREDENTIALS
}

gcloud config set project $Project

Write-Host "==> Ensuring GCS bucket exists: $Bucket"
gcloud storage buckets describe $Bucket 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud storage buckets create $Bucket --location=$Region --uniform-bucket-level-access
}

Write-Host "==> Creating models prefix placeholder"
"prompt-firewall model registry" | gcloud storage cp - "$Bucket/models/.keep"

Write-Host "==> Creating Artifact Registry repo (Docker, for inference images)"
gcloud artifacts repositories describe $Repo --location=$Region 2>$null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repo `
        --repository-format=docker `
        --location=$Region `
        --description="Prompt Firewall inference / API images"
}

Write-Host ""
Write-Host "Done."
Write-Host "  Model files (GCS): $Bucket/models/v2/"
Write-Host "  Docker registry:   $Region-docker.pkg.dev/$Project/$Repo"
Write-Host ""
Write-Host "Required IAM for service account (dvc-airflow-sa):"
Write-Host "  - roles/storage.objectAdmin  on bucket $Bucket"
Write-Host "  - roles/artifactregistry.writer (optional, for Docker pushes)"
