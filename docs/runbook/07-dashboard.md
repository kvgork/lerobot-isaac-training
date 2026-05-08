# Runbook 07 — View Metrics Dashboard

**Package:** `lerobot-isaac-dashboard`
**Env:** `dashboard` pixi environment
**Status:** All phases complete (live UI + static report + snapshots + compare).

---

## Prerequisites

- `pixi install -e dashboard` — installs the dashboard feature environment
- Some pipeline data exists in `outputs/` or `datasets/` (otherwise tabs show "No data yet")
- `LEROBOT_ISAAC_WORKSPACE` environment variable set (or pass `--workspace=$PWD` to CLI commands)

```bash
export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training
pixi install -e dashboard
```

---

## Live Dashboard

### Start

```bash
pixi run -e dashboard dashboard
# Opens http://localhost:8501
```

### Sidebar controls

| Control | Description |
|---------|-------------|
| Workspace path | Resolved from `LEROBOT_ISAAC_WORKSPACE` or current directory (read-only display) |
| Session selector | Drop-down of `.agent-state/<sessionId>/` subdirectories; "All sessions" scopes across all |
| Refresh interval | Slider 0–120 s; 0 = manual refresh only; >0 = auto-refresh via `st.rerun` |
| Watch files | Checkbox; when enabled, watchdog monitors `outputs/` and `.agent-state/` and triggers rerun on any file change |
| Mode radio | **Live** (default) / **Compare (2-way)** / **Compare (N-way)** — changes the main area layout |
| Save snapshot | Text field for label + "Save snapshot" button; writes to `outputs/snapshots/<ts>-<label>/` |
| Export static report | Button; triggers `export_report` and opens a download link for `report.html` |

### Tab guide

**Tab 1 — Data Collection**
Shows a summary table of all LeRobot Parquet datasets under `datasets/`. Each row is one
`<task>/<repo_id>` pair with episode count, FPS, total steps, and disk size. A bar chart
breaks down episodes by source tag (`real`, `sim_dr`, `mimicgen`). Reads from
`datasets/<task>/<repo_id>/meta/info.json` and `meta/episodes.parquet`.

**Tab 2 — Synthetic Data**
Shows the composition of merged datasets: pie chart of real vs sim_dr vs mimicgen episodes,
and a table per merged dataset listing the source breakdown. Reads the `source` column from
`datasets/<merged>/meta/episodes.parquet`. Empty when no merged datasets exist.

**Tab 3 — Policy Training**
Shows loss curves for all policy training runs (`smolvla`, `act`, `diffusion`) and a
checkpoint inventory table with step, timestamp, and size. Loss curves are parsed from
`outputs/checkpoints/<arch>/<run_id>/log.txt` (one `metric_name=value` line per step).

**Tab 4 — World Model Training**
Shows `recon_loss` (DreamerV3) and `pred_loss` (LeWorldModel) curves for all world-model
runs. Source is the same log format as Tab 3, filtered to `arch in {dreamerv3, le_world_model}`.

**Tab 5 — Evaluation**
Shows `pc_success` over evaluation runs as a line chart with a curriculum-stage overlay.
Reads `outputs/eval/*.json`. Each JSON file must contain `{run_id, ts, arch, pc_success, mean_ep_len, n_episodes}`.
Note: schema is contract-pending with `lerobot-evaluation-agent`; a schema warning banner
appears if fields are missing.

**Tab 6 — Autoresearch**
Shows the HP search trial history as a scatter plot (trial index vs metric value) with a
best-config highlight. Also shows plateau detector state (n_no_improve / limit) as a gauge.
Reads `.agent-state/<session>/autoresearch/<slug>/history.jsonl`. Empty when no autoresearch
run has been invoked.

**Tab 7 — Curriculum**
Shows the current curriculum stage as a KPI card and the stage advancement timeline as a
waterfall chart. Reads `outputs/curriculum_stage.json` (current stage) and
`outputs/curriculum_history.jsonl` (advancement events). Note: `curriculum_stage.json` is
written by `lerobot-curriculum-agent`; it is not emitted by any other agent — this is a
known gap until the agent is wired end-to-end.

**Tab 8 — Pipeline Health**
Shows the agent event log as a scrollable table (newest first), an error summary table, and
a workspace checklist (datasets exist? outputs exist? Isaac Lab installed? etc.). Reads
`.agent-state/<session>/events.jsonl`. Useful for diagnosing why other tabs are empty.

---

## Static Report

Export a self-contained HTML report of the current workspace state:

```bash
# Default: inline plotly.js (~5 MB self-contained), with auto-snapshot side effect
pixi run -e dashboard report --workspace=$PWD

# CDN plotly.js: ~50 KB report (requires internet for offline viewing)
pixi run -e dashboard report --workspace=$PWD --cdn

# Disable auto-snapshot side effect
pixi run -e dashboard report --workspace=$PWD --no-snapshot

# Export companion CSVs alongside the HTML
pixi run -e dashboard report --workspace=$PWD --with-csv
```

Output: `outputs/reports/<run_id>/report.html`

The `<run_id>` is a timestamp-based identifier (e.g. `2026-05-08T072115`).
Open the HTML in any browser — no server required in inline mode.

When `--no-snapshot` is not passed, a snapshot is automatically saved to
`outputs/snapshots/<run_id>/` so the report state can be replayed later.

---

## Snapshots

A snapshot captures the full loader state (all DataFrames + metadata) at a point in time.
Snapshots are stored under `outputs/snapshots/` (gitignored) and are never written during
the live dashboard unless you click "Save snapshot" or run the CLI command.

### Save a snapshot

```bash
# CLI: save with label
pixi run -e dashboard snapshot --workspace=$PWD --label=baseline

# CLI: save for a specific session
pixi run -e dashboard snapshot --workspace=$PWD --label=epoch100 --session-id=20260508-064654

# Python module
python -m lerobot_isaac_dashboard.snapshots --workspace=$PWD --label=baseline
```

Output: `outputs/snapshots/<timestamp>-<label>/`

```
outputs/snapshots/2026-05-08T072115-baseline/
├── meta.json          — workspace, git SHA, ts, label, schema_version=1
└── loaders/
    ├── parquet_dataset.parquet
    ├── eval_results.parquet
    ├── ... (one file per loader)
    ├── autoresearch__history.parquet    — hierarchical loaders use __ separator
    ├── autoresearch__program.json
    └── curriculum__current.json
```

### List existing snapshots

```bash
pixi run -e dashboard snapshot --workspace=$PWD list
# Output (newest first):
#   2026-05-08T072115-baseline   2026-05-08 07:21:15 UTC  [baseline]
#   2026-05-07T143000-unlabeled  2026-05-07 14:30:00 UTC
```

### Schema versioning

Snapshot format version is `schema_version: 1`. If a snapshot from a future dashboard
version is loaded, `load_snapshot` raises `ValueError: schema_version=2 > 1`. Fix:
upgrade the dashboard package.

---

## Compare Modes

### 2-way (A vs B)

Compare two snapshots side-by-side:

```bash
# CLI: by snapshot ID
pixi run -e dashboard compare --workspace=$PWD --snapshots baseline after-dr

# CLI: by absolute path
pixi run -e dashboard compare --workspace=$PWD \
  --snapshots outputs/snapshots/2026-05-08T072115-baseline \
              outputs/snapshots/2026-05-08T143000-after-dr

# With CDN plotly
pixi run -e dashboard compare --workspace=$PWD --snapshots baseline after-dr --cdn
```

Output: `outputs/reports/compare-baseline-vs-after-dr/report.html`

Layout: each of the 8 tabs is split into two columns with a **delta KPI strip** above:
- `pc_success (mean)` — A value, B value, delta (green if positive)
- `train_loss (latest)` — A value, B value, delta (green if negative)

**Via UI:** Sidebar → Mode → Compare (2-way) → pick snapshot A and B → renders inline.

### N-way overlay

Compare 3 or more snapshots with overlaid traces:

```bash
# CLI: 4 snapshots overlaid
pixi run -e dashboard compare --workspace=$PWD \
  --snapshots baseline exp-lr1e3 exp-lr5e4 exp-dr5x \
  --mode nway
```

Output: `outputs/reports/compare-baseline-vs-exp-lr1e3-.../report.html`

Layout: time-series charts overlay traces with `<snapshot_label> – <metric>` as legend.
Table-heavy tabs (Pipeline Health) may be empty in the static N-way report as they do
not produce time-series figures suitable for overlay.

**Via UI:** Sidebar → Mode → Compare (N-way) → multiselect 2+ snapshots → renders inline.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "All tabs empty" | `LEROBOT_ISAAC_WORKSPACE` not set or `outputs/` does not exist | Set the env var; run at least one training step to populate `outputs/` |
| "Autoresearch tab empty" | No autoresearch run has been invoked | Confirm `.agent-state/<session>/autoresearch/<slug>/history.jsonl` exists |
| "Curriculum tab empty" | `curriculum_stage.json` not written | `lerobot-curriculum-agent` must write `outputs/curriculum_stage.json`; not emitted until the agent is wired end-to-end — **known gap** |
| "Eval tab schema warning" | `outputs/eval/*.json` has unexpected fields | Schema is contract-pending with `lerobot-evaluation-agent`; check banner warnings and the agent contract doc |
| "Snapshot reload fails: future schema_version" | Snapshot from newer dashboard version | Upgrade the dashboard (`pixi update -e dashboard`) or downgrade the snapshot by re-saving |
| "Compare mode: no snapshots found" | `outputs/snapshots/` is empty | Save at least one snapshot first via the button or `pixi run -e dashboard snapshot` |
| "Port 8501 already in use" | Another Streamlit process running | `pkill -f streamlit` or change port: `pixi run -e dashboard dashboard -- --server.port=8502` |
| Dashboard loads but all charts blank | plotly not installed in dashboard env | `pixi install -e dashboard` and verify: `pixi run -e dashboard python -c "import plotly"` |
