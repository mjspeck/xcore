#!/usr/bin/env bash
# Tests for .superset/link-data.sh against a throwaway root checkout and
# workspace. Run: ./.superset/tests/test_link_data.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/link-data.sh"
# Canonical path, since link-data.sh resolves symlinks such as /var -> /private/var.
TMP="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "$TMP"' EXIT
# Keep git from finding a repository above $TMP and redirecting the script.
export GIT_CEILING_DIRECTORIES="$TMP"
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
ln -s "$TMP/root/data/Gone" data/Gone
mkdir data/cache_local
"$SCRIPT" >/dev/null
check "missing link restored" test -L data/SciCo
check "link to deleted root dataset removed" test ! -L data/Gone
check "local cache kept" test -d data/cache_local

# A data symlink the user made elsewhere is never written through.
fresh
mkdir -p "$TMP/elsewhere"
ln -s "$TMP/elsewhere" data
"$SCRIPT" >/dev/null 2>&1
check "foreign data symlink kept" test -L data
check "nothing linked into foreign dir" test -z "$(ls -A "$TMP/elsewhere")"
"$SCRIPT" --remove >/dev/null
check "foreign data symlink kept on remove" test -L data
rm -rf "$TMP/elsewhere"

# A dangling data symlink does not abort setup.
fresh
ln -s /nonexistent/data data
check "dangling data symlink tolerated" "$SCRIPT"

# The user's own dangling links survive a re-run.
fresh
mkdir data
ln -s /Volumes/unmounted/OntoNotes data/OntoNotes
"$SCRIPT" >/dev/null
check "user's dangling link kept" test -L data/OntoNotes

# A user's dangling link that shares a root dataset's name does not abort.
fresh
mkdir data
ln -s /Volumes/unmounted/LitBank data/LitBank
check "dangling name collision tolerated" "$SCRIPT"
check "user's colliding link kept" test "$(readlink data/LitBank)" = /Volumes/unmounted/LitBank

# Running from a subdirectory still targets the workspace root.
fresh
git init -q .
mkdir sub
(cd sub && "$SCRIPT" >/dev/null)
check "links created at workspace root" test -L data/LitBank
check "no data/ in subdirectory" test ! -e sub/data

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

# Root checkout: never links or removes anything.
fresh
cd "$TMP/root"
mkdir -p "$TMP/storage/LitBank"
rm -rf data/LitBank
ln -s "$TMP/storage/LitBank" data/LitBank
mv data/SciCo "$TMP/storage/SciCo"
ln -s "$TMP/root/data/SciCo" data/SciCo
"$SCRIPT" --remove >/dev/null
check "root's own links untouched" test -L data/LitBank -a -L data/SciCo
check "root reports itself" test "$("$SCRIPT")" = "using local data/ (root checkout)"

# Unknown arguments fail instead of falling through to link mode.
fresh
check "unknown argument rejected" bash -c '! "$0" --rm 2>/dev/null' "$SCRIPT"
check "nothing linked on bad argument" test ! -e data

# A missing root data/ (e.g. unmounted drive) keeps existing links.
fresh
"$SCRIPT" >/dev/null
mv "$TMP/root/data" "$TMP/root/data.unmounted"
"$SCRIPT" >/dev/null 2>&1
check "links kept while root data/ is missing" test -L data/LitBank
mv "$TMP/root/data.unmounted" "$TMP/root/data"
check "links work again once root data/ returns" test -d data/LitBank

# A regular file named data does not abort setup.
fresh
touch data
check "data file tolerated" "$SCRIPT"
check "data file kept" test -f data

# No root data: warns and reports, without failing.
fresh
rm -rf "$TMP/root/data"
status="$("$SCRIPT" 2>/dev/null)"
check "missing root data reported" test "$status" = "no datasets linked"

exit "$FAILED"
