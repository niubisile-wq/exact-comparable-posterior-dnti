# Task 0003 — Regenerate figures and compare with committed figures

status: pending
depends_on: 0002 (venv at $HOME/relay-venv must exist)
executor: Codex

## What to do

1. From the repository root, run:

   ```bash
   bash relay/tasks/0003/run.sh
   ```

   Record its exit code.

2. Commit everything under `relay/results/0003/` and nothing else.
   Commit message: `results: task 0003`

3. In `relay/queue.json`, set task 0003's `status` to:
   - `done` only if the exit code is 0 AND `relay/results/0003/status.txt` says `OK`
   - `failed` otherwise

4. Push to `main`. Do not open a PR.

## Hard rules

- Do not edit anything outside `relay/results/0003/` and `relay/queue.json`.
- Do not modify `paper_archive/`. The script works in a temporary copy.
- Do not interpret results. Keep the raw CSV and logs.
- If `PAUSE` exists in `relay/`, do nothing.

## Report back

In the commit message body, include the script exit code and the contents of `status.txt`.
