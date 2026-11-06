# Task 0005 — Regenerate submission schematics and compare

status: pending
depends_on: 0002 (venv), 0003 (compare.py)
executor: Codex

## Why

Task 0003 regenerated all `plot_fig*` figures and they matched the committed versions
pixel for pixel. Five committed figures were not produced by any `plot_fig*` script.
This task runs the script that produces them: `paper_archive/scripts/plot_submission_schematics.py`.

## What to do

1. From the repository root, run:

   ```bash
   bash relay/tasks/0005/run.sh
   ```

   Record its exit code.

2. Commit everything under `relay/results/0005/` and nothing else.
   Commit message: `results: task 0005`

3. In `relay/queue.json`, set task 0005's `status` to:
   - `done` only if the exit code is 0 AND `relay/results/0005/status.txt` says `OK`
   - `failed` otherwise

4. Push to `main`. Do not open a PR.

## Hard rules

- Do not edit anything outside `relay/results/0005/` and `relay/queue.json`.
- Do not modify `paper_archive/`. The script works in a temporary copy.
- Do not interpret results. Keep the raw CSV and logs.
- If `PAUSE` exists in `relay/`, do nothing.

## Report back

In the commit message body, include the script exit code and the contents of `status.txt`.
