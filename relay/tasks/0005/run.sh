#!/usr/bin/env bash
# Task 0005: regenerate the submission schematics in a throwaway copy and compare with committed figures.
# Writes only to relay/results/0005/ and the venv at $HOME/relay-venv.
# Exit code is non-zero if ANY step failed.
set -u
REPO=$(pwd)
OUT="relay/results/0005"
VENV="$HOME/relay-venv"
mkdir -p "$OUT"
FAILS=0

PY=""
for c in "$VENV/bin/python" "$VENV/Scripts/python.exe"; do
  if [ -x "$c" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "VENV_PYTHON_MISSING: run task 0002 first" | tee "$OUT/status.txt"
  exit 2
fi

WORK=$(mktemp -d)
cp -r paper_archive "$WORK/paper_archive"

# The script writes its outputs into its own scripts/ directory.
( cd "$WORK/paper_archive" && timeout 600 "$PY" -I "scripts/plot_submission_schematics.py" ) > "$OUT/plot_submission_schematics.log" 2>&1
rc=$?
echo "plot_submission_schematics exit=$rc" > "$OUT/exit_codes.txt"
[ $rc -ne 0 ] && FAILS=$((FAILS+1))

# Compare the regenerated files with the committed ones, same as task 0003.
mkdir -p "$WORK/regen"
cp "$WORK/paper_archive/scripts/"*.png "$WORK/regen/" 2>/dev/null
cp "$WORK/paper_archive/scripts/"*.pdf "$WORK/regen/" 2>/dev/null
"$PY" -I relay/tasks/0003/compare.py "$WORK/regen" "$REPO/paper_archive/figures" "$REPO/$OUT/compare.csv" > "$OUT/compare.log" 2>&1
rc=$?
[ $rc -ne 0 ] && FAILS=$((FAILS+1))

rm -rf "$WORK"

if [ $FAILS -eq 0 ]; then echo "OK" > "$OUT/status.txt"; else echo "FAILED steps=$FAILS" > "$OUT/status.txt"; fi
exit $FAILS
