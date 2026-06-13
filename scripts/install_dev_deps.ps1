# Install all Python dependencies for local data + model pipeline development.
# Usage: .\scripts\install_dev_deps.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
$Pip = Join-Path $Root "venv\Scripts\pip.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating venv..."
    python -m venv venv
}

Write-Host "Upgrading pip..."
& $Python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch (CPU)..."
& $Pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

Write-Host "Installing pinned requirements-dev.txt..."
& $Pip install -r requirements-dev.txt --no-cache-dir

Write-Host "Verifying imports..."
& $Python -c @"
mods = [
    'pandas', 'sklearn', 'pyarrow', 'yaml', 'datasets', 'transformers',
    'torch', 'mlflow', 'fairlearn', 'great_expectations', 'matplotlib', 'joblib',
]
failed = []
for m in mods:
    try:
        __import__(m)
        print(f'  OK  {m}')
    except Exception as e:
        failed.append((m, str(e)))
        print(f'  FAIL {m}: {e}')
if failed:
    raise SystemExit(1)
print('All core imports OK.')
"@

Write-Host "Done. Activate with: .\venv\Scripts\activate"
