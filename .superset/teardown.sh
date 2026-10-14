#!/usr/bin/env bash
# Superset workspace teardown: unlinks the shared datasets before the worktree
# is removed. Only symlinks are removed, so the root checkout's data is never touched.
set -euo pipefail

if [ -L data ]; then
  rm data
elif [ -d data ]; then
  find data -maxdepth 1 -type l -delete
fi
echo "Removed dataset symlinks."
