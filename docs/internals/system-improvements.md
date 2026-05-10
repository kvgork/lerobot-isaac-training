# System Improvements Log

Tracks systemic gaps discovered during orchestration runs. Project-specific patterns
go to `CLAUDE.md`. Pipeline improvements edit agent files directly. Only true
infrastructure/missing-agent gaps land here.

## Format

```
### YYYY-MM-DD — {short title}
- **Type:** missing-agent | missing-skill | infrastructure
- **Discovered in:** {sessionId}
- **Gap:** {what was missing}
- **Workaround applied:** {what we did instead}
- **Suggested fix:** {what to build}
```

## Entries

### 2026-05-08 — `outputs/eval/*.json` schema not specified
- **Type:** infrastructure (cross-package contract)
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** `lerobot-evaluation-agent` does not document or enforce the on-disk JSON shape it writes to `outputs/eval/*.json`. The dashboard's `eval_results` loader had to guess keys (`run_id`, `task`, `ts`, `pc_success`, `n_episodes`, `intervention_rate`, `mean_ep_len`).
- **Workaround applied:** Loader uses `df.get(col, default)` per field with NA fill + per-field warnings; documented as "contract pending" in module docstring + dashboard runbook 07.
- **Suggested fix:** Add a `docs/contracts/eval-results.md` (or co-locate with the evaluation agent) that pins the schema. Update `lerobot-evaluation-agent.md` to write that exact shape. Then the dashboard loader can drop its lenient mode.

### 2026-05-08 — `outputs/curriculum_stage.json` never written
- **Type:** missing-skill / agent-output gap
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** `lerobot-curriculum-agent` advances stages but does not persist the current stage state to disk. The dashboard's `curriculum` loader returns empty + UI banner, but the Curriculum tab can never populate without this file.
- **Workaround applied:** Loader returns empty canonical-shape DataFrame; runbook 07 troubleshooting points users at the agent contract.
- **Suggested fix:** Update `lerobot-curriculum-agent` to write `outputs/curriculum_stage.json` (`{stage, task_config, advancement_reason, ts}`) on every advancement, plus append a row to `outputs/curriculum_history.jsonl`.

### 2026-05-08 — No standardized run/snapshot registry across packages
- **Type:** infrastructure
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** Every package writes to `outputs/` with its own ad-hoc layout (eval, checkpoints, autoresearch, snapshots). Cross-package consumers (the dashboard, future agents) must hardcode glob patterns. A registry (e.g. `outputs/runs/<run_id>/manifest.json`) would let consumers iterate runs cleanly.
- **Workaround applied:** Dashboard loaders glob each output type independently; `outputs/snapshots/<id>/meta.json` introduces a partial pattern but only for snapshot state.
- **Suggested fix:** Define a `RunManifest` contract (probably in `lerobot-isaac-meta`) that every training/eval run writes. Migrate existing producers + the dashboard loaders to consume it.
