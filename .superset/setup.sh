#!/usr/bin/env bash
# Superset workspace setup: builds a uv-managed .venv and links shared datasets.
# Safe to re-run: the venv is reused and dataset links are repaired in place.
set -euo pipefail

# Run from the workspace root no matter where the script is called from.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# 3.12 is the newest Python with numpy<2 wheels (see constraints.txt).
PYTHON_VERSION="3.12"

# --- sanity checks -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found in PATH. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

# --- virtualenv --------------------------------------------------------------
venv_version() {
  .venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}
if [ ! -x .venv/bin/python ] || [ -z "$(venv_version)" ]; then
  # Replace a missing or broken .venv (e.g. its interpreter was uninstalled).
  rm -rf .venv
  uv venv --python "$PYTHON_VERSION" .venv
elif [ "$(venv_version)" != "$PYTHON_VERSION" ]; then
  echo "Error: .venv uses Python $(venv_version), but setup needs $PYTHON_VERSION. Remove .venv and re-run." >&2
  exit 1
fi
export VIRTUAL_ENV="$PWD/.venv"

# --- python deps -------------------------------------------------------------
# torch is not listed in requirements.txt; the default PyPI wheel is CPU/MPS on
# macOS and CUDA-enabled on Linux. constraints.txt keeps the spaCy stack
# importable; overrides.txt settles the sentencepiece pin conflict between
# requirements.txt and setup.py (issue #5).
uv pip install -c .superset/constraints.txt --overrides .superset/overrides.txt \
  torch -r requirements.txt -e .

# --- datasets ----------------------------------------------------------------
DATA_STATUS="$(./.superset/link-data.sh)"

# --- smoke check -------------------------------------------------------------
.venv/bin/python - <<'PY'
import torch, pytorch_lightning as pl, spacy, xcore
spacy.load("en_core_web_sm")
print("torch", torch.__version__, "| pytorch_lightning", pl.__version__, "| xcore OK")
PY

echo "Workspace ready ($DATA_STATUS). Activate with: source .venv/bin/activate"
