# Relay

Working bridge between the planner (Claude) and the executor (Codex).

- `tasks/` — task files written by the planner. Each task has a `status` (pending / done / failed).
- `results/` — raw outputs written by the executor: logs, exit codes, generated files. No conclusions.

Rules:
- The executor only runs tasks under `relay/tasks/`.
- The executor only writes to `relay/results/`.
- If a file named `PAUSE` exists in `relay/`, the executor stops.
