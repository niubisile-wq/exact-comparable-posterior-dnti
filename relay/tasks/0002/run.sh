#!/usr/bin/env bash
# Task 0002: health check of paper_archive (fixed version of 0001).
# Works on Linux and Windows Git Bash.
# Writes only to relay/results/0002/ and to a venv outside the repo.
# Exit code is non-zero if ANY step failed.
set -u
REPO=$(pwd)
OUT="relay/results/0002"
VENV="$HOME/relay-venv"
mkdir -p "$OUT"
FAILS=0

# 0. Find a working Python (skip Windows Store stubs)
PY_SYS=""
for c in python3 python py; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; assert sys.version_info >= (3, 9)" >/dev/null 2>&1; then
    PY_SYS="$c"; break
  fi
done
if [ -z "$PY_SYS" ]; then
  echo "NO_WORKING_PYTHON" | tee "$OUT/status.txt"
  exit 2
fi
echo "python_system=$PY_SYS" > "$OUT/pip.log"

# 1. Virtualenv outside the repo, with OS-specific paths
if [ ! -d "$VENV" ]; then
  "$PY_SYS" -m venv "$VENV" >> "$OUT/pip.log" 2>&1 || FAILS=$((FAILS+1))
fi
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"
else
  echo "VENV_PYTHON_MISSING" >> "$OUT/pip.log"; FAILS=$((FAILS+1)); PY=""
fi
if [ -n "$PY" ]; then
  "$PY" -m pip install -q -r paper_archive/requirements.txt >> "$OUT/pip.log" 2>&1
  rc=$?; echo "pip_exit=$rc" >> "$OUT/pip.log"; [ $rc -ne 0 ] && FAILS=$((FAILS+1))
fi

# 2. Inventory and checksums (read-only)
find paper_archive -type f | sort > "$OUT/inventory.txt"
find paper_archive -type f -exec sha256sum {} + | sort -k2 > "$OUT/checksums.txt"
echo "file_count=$(wc -l < "$OUT/inventory.txt")" > "$OUT/summary.txt"

# 3. Environment record
if [ -n "$PY" ]; then
  { echo "--- python"; "$PY" --version 2>&1
    echo "--- os"; uname -a
    echo "--- requirements.txt"; cat paper_archive/requirements.txt
    echo "--- installed"; "$PY" -m pip freeze 2>&1
  } > "$OUT/env.txt" 2>&1
fi

# 4. Figure scripts, each in a throwaway copy so the repo is never modified
if [ -n "$PY" ]; then
  WORK=$(mktemp -d)
  cp -r paper_archive "$WORK/paper_archive"
  for s in paper_archive/scripts/plot_fig*.py; do
    name=$(basename "$s" .py)
    ( cd "$WORK/paper_archive" && timeout 300 "$PY" -I "scripts/$name.py" ) > "$REPO/$OUT/$name.log" 2>&1
    rc=$?
    echo "$name exit=$rc" >> "$OUT/exit_codes.txt"
    [ $rc -ne 0 ] && FAILS=$((FAILS+1))
  done
  rm -rf "$WORK"
fi

if [ $FAILS -eq 0 ]; then echo "OK" > "$OUT/status.txt"; else echo "FAILED steps=$FAILS" > "$OUT/status.txt"; fi
echo "done" >> "$OUT/summary.txt"
exit $FAILS
