# Runbook 08 — Batch train and auto-compare

Train multiple `target_arch`s sequentially on the same dataset and render a
single side-by-side / overlay HTML report — no manual snapshot juggling.

**Entrypoint:** `lerobot-isaac-batch` (registered by `lerobot-isaac-meta`).
**Module:** `lerobot_isaac_meta.batch` + `lerobot_isaac_meta.batch_config`.
**Compare backend:** `lerobot_isaac_dashboard.snapshots.save_snapshot` plus
`lerobot_isaac_dashboard.compare.export_compare_report`.

---

## When to use

- A/B-comparing two backends on identical data (e.g. `smolvla` vs
  `le_world_model`).
- Sweeping seeds or learning rates with the same arch.
- Running an overnight queue of three or four runs and waking up to one HTML
  report instead of N tabs.

For a single-run training session use `lerobot-isaac-train` directly
(see runbook 03 / 04).
For automated hyperparameter search use the autoresearch loop
(see `USAGE.md §F`).

---

## YAML schema

```yaml
batch_id: <string>            # filesystem-safe; used as snapshot prefix
dataset:  <path-or-repo-id>   # default for every run
output_root: outputs/runs     # default; per-run output_dir overrides this
on_failure: continue          # continue | abort
compare:
  enabled: true
  mode: nway                  # 2way | nway
  output_dir: null            # default: outputs/reports/compare-<batch_id>/
runs:
  - id: <unique>
    target_arch: smolvla|act|diffusion|dreamerv3|le_world_model
    config: <path>            # optional — falls back to lerobot-isaac-configs default
    dataset: <override>       # optional — overrides batch-level dataset
    steps: 50000              # optional knobs mirror lerobot-isaac-train
    batch_size: 32
    lr: 1.0e-4
    seed: 42
    output_dir: <path>        # optional — defaults to {output_root}/{batch_id}/{id}
    label: SmolVLA baseline   # optional — appears in compare legend
    extra_args: []            # forwarded to backend after `--`
```

Validation (raised as `BatchConfigError`):

| Rule | Trigger |
|------|---------|
| `runs` non-empty | empty list |
| `run.id` unique within a batch | duplicate ids |
| `target_arch` ∈ supported list | unknown arch |
| `compare.mode` ∈ `{"2way", "nway"}` | `2way` requires exactly 2 runs |
| `on_failure` ∈ `{"continue", "abort"}` | other strings |

---

## Execution flow

```
for run in cfg.runs:
    cmd = python -m lerobot_isaac_adapters.train \
              --target_arch <arch> \
              --dataset <resolved> \
              --config <run.config> \
              --output_dir {output_root}/{batch_id}/{run.id} \
              [--steps --batch_size --lr --seed] \
              [-- run.extra_args ...]

    rc = subprocess.run(cmd)
    if rc == 0 and not dry_run:
        snapshot_id = save_snapshot(label=run.label, snapshot_id=f"{batch_id}-{run.id}")
    elif on_failure == "abort":
        break

if compare.enabled and ≥ 2 successful snapshots:
    export_compare_report(snapshot_ids=[...], mode=compare.mode,
                          output_dir=outputs/reports/compare-<batch_id>/)
```

Each run is its own subprocess — soft-import discipline (ADR-0003) preserved;
heavy deps stay isolated per backend.

---

## Quickstart

### 1. Drop a batch YAML

A worked example ships with the configs package:

```bash
cat packages/lerobot-isaac-configs/src/lerobot_isaac_configs/configs/batches/example.yaml
```

Edit it (or copy under your own batch id) — point `dataset:` at a real
LeRobotDataset directory.

### 2. Verify dispatch with `--dry_run`

```bash
pixi run -e default lerobot-isaac-batch \
    --config packages/lerobot-isaac-configs/src/lerobot_isaac_configs/configs/batches/example.yaml \
    --workspace . \
    --dry_run
```

Each run prints its resolved subprocess command and exits 0 — no checkpoints
written, no snapshots taken.

### 3. Run for real

```bash
pixi run -e full lerobot-isaac-batch \
    --config <your-batch.yaml> \
    --workspace .
```

`-e full` activates the env that has every backend importable. Use a narrower
env (`train-policy`, `train-lewm`, …) when the batch only touches one family.

The pixi shortcut runs the example config directly:

```bash
pixi run train-and-compare
```

### 4. Open the compare report

```bash
xdg-open outputs/reports/compare-<batch_id>/report.html
# or copy to a workstation
```

Snapshots are kept under `outputs/snapshots/<batch_id>-<run_id>/` — re-render
the report any time with the dashboard CLI:

```bash
lerobot-isaac-compare --workspace . \
    --snapshots <batch_id>-smolvla-baseline <batch_id>-lewm-baseline \
    --mode nway
```

---

## Failure handling

`on_failure: continue` (default):

- A failed run produces `RunResult(exit_code != 0, snapshot_id=None)`.
- Subsequent runs proceed.
- The compare step uses only the successful snapshots; if fewer than two
  succeed compare is skipped with an info-level log line.
- Process exit code is `1` when at least one run failed.

`on_failure: abort`:

- First non-zero exit halts the batch immediately.
- `result.aborted = True`; no snapshot, no compare.
- Process exit code is `1`.

`subprocess.run` raising (e.g. `python` missing) is caught — the run is
recorded with `exit_code = -1` and `error = str(exc)`.

---

## Compare semantics

`save_snapshot` captures the **entire workspace state** at call time. So with
runs `A → B`:

| Snapshot | Contents |
|----------|----------|
| `<batch>-A` | artefacts from run A only |
| `<batch>-B` | artefacts from runs A **and** B |

In the N-way overlay this still surfaces per-arch deltas — the
`load_checkpoints` and `load_training_logs` loaders group by `arch / run_id`,
so each arch shows up as its own trace. Per-run isolated comparison (filtering
the workspace to a single arch) is a planned future enhancement.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `error: 'dataset' is required` | Top-level `dataset:` missing | Add at batch level or per run |
| `compare.mode='2way' requires exactly 2 runs` | Schema | Use `mode: nway` for ≥ 3 runs |
| Compare step skipped | < 2 successful runs | Inspect run summary; check failed runs first |
| `Cannot snapshot — lerobot-isaac-dashboard missing` | Dashboard pkg not installed | Use `pixi run -e dashboard …` or install `lerobot-isaac-dashboard` |
| Subprocess `exit_code = 127` | `lerobot-isaac-train` not on PATH | `pixi shell -e <env>` or run via `pixi run -e <env> …` |

---

## Related docs

- [`docs/runbook/03-train-policy.md`](03-train-policy.md) — single-run policy training.
- [`docs/runbook/04-train-world-model.md`](04-train-world-model.md) — single-run world-model training.
- [`docs/runbook/07-dashboard.md`](07-dashboard.md) — dashboard tabs, snapshots, compare deep-dive.
- `packages/lerobot-isaac-meta/src/lerobot_isaac_meta/batch_config.py` — full schema source.
- `packages/lerobot-isaac-meta/src/lerobot_isaac_meta/batch.py` — runner source.
