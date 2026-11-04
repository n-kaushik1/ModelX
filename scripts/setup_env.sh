#!/usr/bin/env bash
# One-time setup for Linux, macOS and the Claude Code cloud VM:
#   1. .venv with Python 3.11 and the exact leaderboard library versions
#   2. the development CSVs unzipped from public_data.zip into data/
# usage (from anywhere):  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REQ=submission/improved/requirements.txt

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv          # uv downloads Python 3.11 itself if it is not installed
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    echo "Python 3.11 is needed: install uv (https://docs.astral.sh/uv/) or python3.11, then re-run." >&2
    exit 1
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r "$REQ"
else
  .venv/bin/python -m pip install -r "$REQ"
fi

.venv/bin/python -c "import zipfile; zipfile.ZipFile('public_data.zip').extractall('data')"

.venv/bin/python - <<'EOF'
import os, sys
import catboost, lightgbm, numpy, pandas, sklearn, xgboost
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__,
      "| scikit-learn", sklearn.__version__, "| lightgbm", lightgbm.__version__,
      "| xgboost", xgboost.__version__, "| catboost", catboost.__version__)
print("data/:", sorted(os.listdir("data")))
if not sys.version.startswith("3.11."):
    print("WARNING: the leaderboard uses Python 3.11", file=sys.stderr)
EOF
