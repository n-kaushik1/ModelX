# One-time setup on Windows:
#   1. .venv with Python 3.11 and the exact leaderboard library versions
#   2. the development CSVs unzipped from public_data.zip into data\
# usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv }
    else { python -m venv .venv }
    if (-not (Test-Path $py)) { throw "Could not create .venv - install Python 3.11 (python.org or Microsoft Store) and re-run" }
}
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -ne "3.11") {
    Write-Warning "The venv uses Python $ver but the leaderboard uses 3.11. Delete .venv, install Python 3.11 and re-run."
}

& $py -m pip install -r submission\improved\requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& $py -c "import zipfile; zipfile.ZipFile('public_data.zip').extractall('data')"
& $py -c "import os, sys, numpy, pandas, sklearn, lightgbm, xgboost, catboost; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__, '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| catboost', catboost.__version__); print('data:', sorted(os.listdir('data')))"
