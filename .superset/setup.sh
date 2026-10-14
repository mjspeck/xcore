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
if [ ! -x .venv/bin/python ]; then
  # Replace a broken .venv (e.g. its interpreter was uninstalled).
  rm -rf .venv
  uv venv --python "$PYTHON_VERSION" .venv
fi
export VIRTUAL_ENV="$PWD/.venv"

# --- python deps -------------------------------------------------------------
# torch is not listed in requirements.txt; the default PyPI wheel is CPU/MPS on
# macOS and CUDA-enabled on Linux. Install the package last so setup.py's
# sentencepiece==0.2.1 pin wins over requirements.txt's 0.2.0 (issue #5).
# .superset/constraints.txt keeps the spaCy stack importable and transformers<5.
CONSTRAINTS=".superset/constraints.txt"
uv pip install -c "$CONSTRAINTS" torch
uv pip install -c "$CONSTRAINTS" -r requirements.txt
uv pip install -c "$CONSTRAINTS" -e .

# --- datasets ----------------------------------------------------------------
DATA_STATUS="$(./.superset/link-data.sh)"

# --- smoke check -------------------------------------------------------------
.venv/bin/python - <<'PY'
import torch, pytorch_lightning as pl, spacy, xcore
spacy.load("en_core_web_sm")
print("torch", torch.__version__, "| pytorch_lightning", pl.__version__, "| xcore OK")
PY

echo "Workspace ready ($DATA_STATUS). Activate with: source .venv/bin/activate"
