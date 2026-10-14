#!/usr/bin/env bash
# Superset workspace teardown: unlinks the shared datasets before the worktree
# is removed. Only a symlink is removed, so the root checkout's data is never touched.
set -euo pipefail

if [ -L data ]; then
  rm data
  echo "Removed data symlink."
fi
