# Start the Prompt Firewall API (Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".\.secrets\groq.env")) {
    Write-Error "Missing .secrets\groq.env — copy .secrets\groq.env.example and add your GROQ_API_KEY."
}

$env:PYTHONPATH = "src"
& ".\venv\Scripts\python.exe" -m uvicorn firewall.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
