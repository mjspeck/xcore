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

ROOT_PATH="${SUPERSET_ROOT_PATH:-}"
if [ -z "$ROOT_PATH" ]; then
  # Outside Superset, the main checkout owns the shared .git directory.
  common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [ -n "$common_dir" ]; then
    ROOT_PATH="$(dirname "$common_dir")"
  fi
fi
ROOT_DATA="$ROOT_PATH/data"

if [ -z "$ROOT_PATH" ] || [ "$(cd "$ROOT_PATH" && pwd -P)" = "$(pwd -P)" ]; then
  echo "using local data/ (root checkout)"
  exit 0
fi

# Removes only links this script created: data itself (older whole-directory
# layout) or data/<name> pointing at the root's data/<name>.
remove_links() {
  if [ -L data ]; then
    if [ "$(readlink data)" = "$ROOT_DATA" ]; then
      rm data
    fi
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

# Migrate the older layout, where data was one symlink to the root's data/.
if [ -L data ] && [ "$(readlink data)" = "$ROOT_DATA" ]; then
  rm data
fi
mkdir -p data
for link in data/*; do
  if [ -L "$link" ] && [ ! -e "$link" ]; then
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
