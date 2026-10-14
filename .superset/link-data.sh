#!/usr/bin/env bash
# Link the root checkout's datasets into this workspace's data/, or remove
# those links with --remove.
#
# conf/data/*.yaml read from ./data, which is gitignored. Each dataset is
# linked from the root checkout instead of downloading ~200MB per workspace,
# but data/ stays a real directory so processed caches (data/cache_*) remain
# per workspace: the cache key ignores code changes, so a shared cache would
# leak between branches. Linked datasets are shared; treat them as read-only.
#
# Prints a one-line status on stdout; warnings go to stderr.
set -euo pipefail

# Work from the workspace root no matter where the script is called from.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

ROOT_PATH="${SUPERSET_ROOT_PATH:-}"
if [ -z "$ROOT_PATH" ]; then
  # Outside Superset, the main checkout is the first entry git lists.
  ROOT_PATH="$(git worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p')"
fi
if [ -z "$ROOT_PATH" ] || [ ! -d "$ROOT_PATH" ]; then
  echo "using local data/ (no root checkout found)"
  exit 0
fi
RAW_ROOT_DATA="${ROOT_PATH%/}/data"
ROOT_PATH="$(cd "$ROOT_PATH" && pwd -P)"
ROOT_DATA="$ROOT_PATH/data"

if [ "$ROOT_PATH" = "$(pwd -P)" ]; then
  echo "using local data/ (root checkout)"
  exit 0
fi

# True when data is the older whole-directory link to the root's data/.
is_old_layout() {
  [ -L data ] && { [ "$(readlink data)" = "$ROOT_DATA" ] || [ "$(readlink data)" = "$RAW_ROOT_DATA" ]; }
}

# Removes only links this script created: data itself (older whole-directory
# layout) or data/<name> pointing at the root's data/<name>.
remove_links() {
  if is_old_layout; then
    rm data
    return
  fi
  if [ -L data ]; then
    return
  fi
  for link in data/*; do
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$ROOT_DATA/$(basename "$link")" ]; then
      rm "$link"
    fi
  done
}

if [ "${1:-}" = "--remove" ]; then
  remove_links
  echo "removed dataset links"
  exit 0
fi

if [ ! -d "$ROOT_DATA" ]; then
  echo "Warning: no datasets at $ROOT_DATA; run ./download_datasets.sh in $ROOT_PATH." >&2
  echo "no datasets linked"
  exit 0
fi

# Migrate the older layout. Any other data symlink belongs to the user, and
# writing through it could touch a directory other workspaces share.
if is_old_layout; then
  rm data
elif [ -L data ]; then
  echo "Warning: data is a symlink to $(readlink data); leaving it as is." >&2
  echo "using existing data symlink"
  exit 0
fi
mkdir -p data
# Drop links to root datasets that no longer exist; keep the user's own links.
for link in data/*; do
  if [ -L "$link" ] && [ ! -e "$link" ] && [ "$(readlink "$link")" = "$ROOT_DATA/$(basename "$link")" ]; then
    rm "$link"
  fi
done
for src in "$ROOT_DATA"/*; do
  name="$(basename "$src")"
  case "$name" in cache_*) continue ;; esac
  if [ -e "$src" ] && [ ! -e "data/$name" ]; then
    ln -s "$src" "data/$name"
  fi
done
echo "datasets linked from $ROOT_DATA"
