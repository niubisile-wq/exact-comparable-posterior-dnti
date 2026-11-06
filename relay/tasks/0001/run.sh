#!/usr/bin/env bash
# Task 0001: health check of paper_archive.
# Executor: run from the repository root. Only write inside relay/results/0001/ and
# the venv at $HOME/relay-venv (outside the repo).
set -u
OUT="relay/results/0001"
VENV="$HOME/relay-venv"
mkdir -p "$OUT"

# 0. Dependencies: isolated venv outside the repo
python3 -m venv "$VENV" > "$OUT/pip.log" 2>&1
"$VENV/bin/pip" install -q -r paper_archive/requirements.txt >> "$OUT/pip.log" 2>&1
echo "pip_exit=$?" >> "$OUT/pip.log"
PY="$VENV/bin/python"

# 1. File inventory and checksums (read-only)
find paper_archive -type f | sort > "$OUT/inventory.txt"
find paper_archive -type f -exec sha256sum {} + | sort -k2 > "$OUT/checksums.txt"
echo "file_count=$(wc -l < "$OUT/inventory.txt")" > "$OUT/summary.txt"

# 2. Environment (record only)
{
  echo "--- python"; "$PY" --version 2>&1
  echo "--- os"; uname -a
  echo "--- requirements.txt"; cat paper_archive/requirements.txt 2>&1
  echo "--- installed versions"
  "$PY" -m pip freeze 2>&1
} > "$OUT/env.txt" 2>&1

# 3. Figure scripts: run each in a throwaway copy so the repo is never modified
REPO=$(pwd)
WORK=$(mktemp -d)
cp -r paper_archive "$WORK/paper_archive"
cd "$WORK/paper_archive" || exit 1
for s in scripts/plot_fig*.py; do
  name=$(basename "$s" .py)
  timeout 300 "$PY" -I "$s" > "$REPO/$OUT/$name.log" 2>&1
  echo "$name exit=$?" >> "$REPO/$OUT/exit_codes.txt"
done
cd "$REPO" || exit 1
rm -rf "$WORK"

echo "done" >> "$OUT/summary.txt"
