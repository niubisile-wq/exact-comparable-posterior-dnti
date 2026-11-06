# Task 0002 — Repository health check (fixed)

status: pending
replaces: 0001 (0001 failed: Python venv was never created on Windows Git Bash)
executor: Codex

## What to do

1. From the repository root, run:

   ```bash
   bash relay/tasks/0002/run.sh
   ```

   Record its exit code.

2. Commit everything under `relay/results/0002/` and nothing else.
   Commit message: `results: task 0002`

3. In `relay/queue.json`, set task 0002's `status` to:
   - `done` only if the script exit code is 0 AND `relay/results/0002/status.txt` says `OK`
   - `failed` otherwise

4. Push to `main`. Do not open a PR.

## Hard rules

- Do not edit any file outside `relay/results/0002/` and `relay/queue.json`.
- Do not install anything outside the venv at `$HOME/relay-venv`.
- Do not interpret results. Keep raw output only.
- If `NO_WORKING_PYTHON` appears, stop. Record it and push. Do not try to install Python.
- If `PAUSE` exists in `relay/`, do nothing.

## Report back

In the commit message body, include the script exit code and the contents of `status.txt`.
