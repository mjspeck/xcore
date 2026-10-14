#!/usr/bin/env bash
# Superset workspace setup: builds a uv-managed .venv and links shared datasets.
# Safe to re-run; every step skips work that is already done.
set -euo pipefail

PYTHON_VERSION="${XCORE_PYTHON_VERSION:-3.12}"
ROOT_PATH="${SUPERSET_ROOT_PATH:-}"

# --- sanity checks -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found in PATH. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

# --- virtualenv --------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  uv venv --python "$PYTHON_VERSION" .venv
fi
export VIRTUAL_ENV="$PWD/.venv"

# --- python deps -------------------------------------------------------------
# torch is not listed in requirements.txt; the default PyPI wheel is CPU/MPS on
# macOS and CUDA-enabled on Linux. Install the package last so setup.py's pins
# (transformers<5, sentencepiece==0.2.1) win over requirements.txt.
# .superset/constraints.txt keeps the spaCy stack importable.
CONSTRAINTS=".superset/constraints.txt"
uv pip install -c "$CONSTRAINTS" torch
uv pip install -c "$CONSTRAINTS" -r requirements.txt
uv pip install -c "$CONSTRAINTS" -e .

# --- datasets ----------------------------------------------------------------
# conf/data/*.yaml read from ./data, which is gitignored. Share the root
# checkout's copy instead of downloading ~200MB per workspace.
if [ -e data ] || [ -L data ]; then
  echo "data/ already present; leaving it as is."
elif [ -n "$ROOT_PATH" ] && [ "$ROOT_PATH" != "$PWD" ] && [ -d "$ROOT_PATH/data" ]; then
  ln -s "$ROOT_PATH/data" data
  echo "Linked data -> $ROOT_PATH/data"
else
  echo "Warning: no datasets found at \$SUPERSET_ROOT_PATH/data; run ./download_datasets.sh to fetch them." >&2
fi

# --- smoke check -------------------------------------------------------------
.venv/bin/python - <<'PY'
import torch, pytorch_lightning as pl, spacy, xcore
spacy.load("en_core_web_sm")
print("torch", torch.__version__, "| pytorch_lightning", pl.__version__, "| xcore OK")
PY

echo "Workspace ready. Activate with: source .venv/bin/activate"
