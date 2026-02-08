#!/usr/bin/env bash
set -euo pipefail

PATCH="${1:-}"
if [[ -z "$PATCH" ]]; then
  echo "Usage: scripts/apply_patch_safe.sh <patchfile>"
  exit 2
fi

if [[ ! -f "$PATCH" ]]; then
  echo "No such patch file: $PATCH"
  exit 3
fi

echo "STATUS:"
git status -sb
git log -1 --oneline --decorate

if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "Already applied: $PATCH"
  exit 0
fi

echo "CHECK:"
if git apply --check "$PATCH" >/dev/null 2>&1; then
  echo "APPLY (clean):"
  git apply "$PATCH"
elif git apply --3way --check "$PATCH" >/dev/null 2>&1; then
  echo "APPLY (3way):"
  git apply --3way "$PATCH" || true
else
  echo "Patch does not apply: $PATCH"
  exit 4
fi

if git ls-files -u | grep -q .; then
  echo "Conflicts detected; resetting to clean state."
  git reset --hard HEAD
  find . \( -name '*.rej' -o -name '*.orig' \) -print
  find . \( -name '*.rej' -o -name '*.orig' \) -delete
  exit 5
fi

echo "DIFFSTAT:"
git diff --stat

echo "PY_COMPILE (best effort):"
python3 -m py_compile simulation.py motor.py open_loop.py world.py 2>/dev/null || true

echo "COMMIT+PUSH:"
git add -A
git commit -m "Chore: add scripts/apply_patch_safe.sh"
git push
