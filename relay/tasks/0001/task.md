# Task 0001 — Repository health check

status: pending
owner: planner (Claude)
executor: Codex

## What to do

1. From the repository root, run:

   ```bash
   bash relay/tasks/0001/run.sh
   ```

2. Commit everything written under `relay/results/0001/` and nothing else.
   Commit message: `results: task 0001`

3. In `relay/queue.json`, set this task's `status` to `done` if `run.sh` exited 0,
   otherwise to `failed`. Commit that change with the results.

4. Push to `main`.

## Hard rules

- Do not edit any file outside `relay/results/0001/` and `relay/queue.json`.
- Packages may be installed only into the venv at `$HOME/relay-venv` (done by `run.sh`). Do not install anything globally, do not change `requirements.txt`, do not delete anything.
- Do not interpret results. Keep raw output only. The planner decides what it means.
- If a step hangs past its timeout, let the script record it and continue.
- If `PAUSE` exists in `relay/`, stop and do nothing.

## Expected output

`relay/results/0001/` contains:

- `inventory.txt` — every file under `paper_archive/`
- `checksums.txt` — sha256 of every file
- `summary.txt` — file count and `done` marker
- `env.txt` — Python, OS, and installed versions of required packages
- `plot_fig*.log` — output of each figure script
- `exit_codes.txt` — one `name exit=N` line per figure script
