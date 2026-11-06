#!/usr/bin/env bash
# Task 0003: regenerate all figures in a throwaway copy and compare with committed figures.
# Writes only to relay/results/0003/ and the venv at $HOME/relay-venv.
# Exit code is non-zero if ANY step failed.
set -u
REPO=$(pwd)
OUT="relay/results/0003"
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

# Regenerate. Scripts write their outputs into their own scripts/ directory (ROOT = scripts/).
for s in paper_archive/scripts/plot_fig*.py; do
  name=$(basename "$s" .py)
  ( cd "$WORK/paper_archive" && timeout 300 "$PY" -I "scripts/$name.py" ) > "$REPO/$OUT/$name.log" 2>&1
  rc=$?
  echo "$name exit=$rc" >> "$OUT/exit_codes.txt"
  [ $rc -ne 0 ] && FAILS=$((FAILS+1))
done

# Compare: regenerated PNG/PDF in the copy's scripts/ vs committed figures/
"$PY" -I relay/tasks/0003/compare.py "$WORK/paper_archive/scripts" "$REPO/paper_archive/figures" "$REPO/$OUT/compare.csv" > "$OUT/compare.log" 2>&1
rc=$?
[ $rc -ne 0 ] && FAILS=$((FAILS+1))

rm -rf "$WORK"

if [ $FAILS -eq 0 ]; then echo "OK" > "$OUT/status.txt"; else echo "FAILED steps=$FAILS" > "$OUT/status.txt"; fi
exit $FAILS
