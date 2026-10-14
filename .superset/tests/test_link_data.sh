#!/usr/bin/env bash
# Tests for .superset/link-data.sh against a throwaway root checkout and
# workspace. Run: ./.superset/tests/test_link_data.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/link-data.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0

check() {
  local desc="$1"; shift
  if "$@"; then
    echo "ok   - $desc"
  else
    echo "FAIL - $desc"
    FAILED=1
  fi
}

fresh() {
  rm -rf "$TMP/root" "$TMP/ws"
  mkdir -p "$TMP/root/data/LitBank" "$TMP/root/data/SciCo" "$TMP/root/data/cache_1770" "$TMP/ws"
  export SUPERSET_ROOT_PATH="$TMP/root"
  cd "$TMP/ws"
}

# Link: every dataset except cache_* is linked into a real data/ directory.
fresh
"$SCRIPT" >/dev/null
check "data/ is a real directory" test -d data -a ! -L data
check "LitBank linked to root" test "$(readlink data/LitBank)" = "$TMP/root/data/LitBank"
check "SciCo linked to root" test -L data/SciCo
check "cache_* not linked" test ! -e data/cache_1770

# Re-run: repairs missing and dangling links, leaves local files alone.
rm data/SciCo
ln -s /nonexistent/ghost data/ghost
mkdir data/cache_local
"$SCRIPT" >/dev/null
check "missing link restored" test -L data/SciCo
check "dangling link removed" test ! -L data/ghost
check "local cache kept" test -d data/cache_local

# Migration: the older whole-directory symlink becomes per-dataset links.
fresh
ln -s "$TMP/root/data" data
"$SCRIPT" >/dev/null
check "old data symlink replaced" test -d data -a ! -L data
check "root cache untouched by migration" test -d "$TMP/root/data/cache_1770"

# Remove: only links pointing at the root's data/<name> are deleted.
fresh
"$SCRIPT" >/dev/null
mkdir -p "$TMP/elsewhere/Extra"
ln -s "$TMP/elsewhere/Extra" data/Extra
mkdir data/cache_local
"$SCRIPT" --remove >/dev/null
check "root links removed" test ! -e data/LitBank -a ! -e data/SciCo
check "user's own link kept" test -L data/Extra
check "local cache kept on remove" test -d data/cache_local
check "root datasets intact" test -d "$TMP/root/data/LitBank" -a -d "$TMP/root/data/SciCo"

# Remove with the older layout deletes only the symlink.
fresh
ln -s "$TMP/root/data" data
"$SCRIPT" --remove >/dev/null
check "old data symlink removed" test ! -e data -a ! -L data
check "root data intact after old-layout remove" test -d "$TMP/root/data/LitBank"

# Root checkout: never links or removes anything.
fresh
cd "$TMP/root"
ln -s "$TMP/root/data/LitBank" data/Alias
"$SCRIPT" --remove >/dev/null
check "root's own links untouched" test -L data/Alias

# No root data: warns and reports, without failing.
fresh
rm -rf "$TMP/root/data"
status="$("$SCRIPT" 2>/dev/null)"
check "missing root data reported" test "$status" = "no datasets linked"

exit "$FAILED"
