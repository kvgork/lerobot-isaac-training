# lerobot-isaac-autoresearch

Holds the `program.md` configs that drive the autoresearch ML loop for LeRobot
policies and world models. Wraps the adapters' `train.py` so
`autoresearch-ml-executor-worker` can run it via a single stable entrypoint.

---

## Purpose

This package provides two things:

1. **`programs/`** — Three `program.md` files (one per training target family),
   each consumed by `autoresearch-loop-orchestrator` to run autonomous
   hyperparameter search.
2. **`src/lerobot_isaac_autoresearch/train_wrapper.py`** — A thin shim that
   forwards CLI args to `lerobot_isaac_adapters.train` via subprocess, captures
   stdout, guarantees a regex-parseable metric line as the final output line,
   and handles CUDA OOM by halving `--batch_size` and retrying once.

---

## Status

**Phase 3 — implemented.** `train_wrapper.py` is functional. Programs are configured
and ready for autoresearch runs.

| Component | Status |
|-----------|--------|
| `train_wrapper.py` | Implemented — OOM recovery, timeout, metric guarantee |
| `programs/lerobot-policy.md` | Configured — SmolVLA/ACT/Diffusion |
| `programs/dreamerv3.md` | Configured — DreamerV3 world model |
| `programs/leworldmodel.md` | Configured — HF LeWorldModel |

---

## Installation

### Monorepo mode (pixi)

```bash
pixi install   # from workspace root
```

### Standalone mode

```bash
cd packages/lerobot-isaac-autoresearch
pixi install
```

### Direct pip install

```bash
pip install -e packages/lerobot-isaac-autoresearch/
```

This also installs `lerobot-isaac-adapters` (listed as a dependency in `pyproject.toml`).

---

## Quick Example

### Invoke train_wrapper directly (manual test)

```bash
python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch smolvla \
  --dataset /data/real \
  --output_dir /tmp/test_run \
  --steps 100 \
  --dry_run
```

Expected output:
```
[train_wrapper] running: python -m lerobot_isaac_adapters.train --target_arch smolvla ...
[dry_run] target_arch=smolvla ...
pc_success=0.0
```

### Launch autoresearch loop

```bash
cd ~/tools/claude_code

# LeRobot policy search
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model

# DreamerV3 world model search
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md --type ml_model
```

---

## Programs

| File | Target | Metric | Direction | Budget |
|------|--------|--------|-----------|--------|
| `programs/lerobot-policy.md` | SmolVLA / ACT / Diffusion | `pc_success` | maximize | 10 runs × 2h |
| `programs/dreamerv3.md` | DreamerV3 world model | `recon_loss` | minimize | — |
| `programs/leworldmodel.md` | HF LeWorldModel | `pred_loss` | minimize | — |

### `program.md` schema (key fields)

```yaml
## Research Goal
Maximize pc_success for SO-101 manipulation ...

## Training Script
path: src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch smolvla --dataset {dataset} --output_dir {out} --steps {steps}"

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.]+)'

## Budget
seconds_per_experiment: 7200
max_experiments: 10
plateau_limit: 3
```

---

## Public API

- **`lerobot_isaac_autoresearch.train_wrapper.main()`** — CLI entrypoint.
  Registered as `lerobot-isaac-train-wrapper` console script.
- **`lerobot_isaac_autoresearch.train_wrapper.run(args)`** — executes the training
  subprocess with OOM recovery and metric guarantee.
- **`lerobot_isaac_autoresearch.train_wrapper.parse_args(argv=None)`** — parses
  CLI args; returns `(namespace, extra_args)`.
- **`TRAIN_TIMEOUT_SECONDS`** — hard timeout ceiling (default: 4 hours). Override
  via `LEROBOT_TRAIN_TIMEOUT` env var.

---

## train_wrapper.py Contract

`autoresearch-ml-executor-worker` invokes `train_wrapper.py` as:

```bash
python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch smolvla \
  --dataset /path/to/dataset \
  --output_dir /tmp/run_001 \
  --steps 20000
```

The wrapper:

1. Builds a `python -m lerobot_isaac_adapters.train` subprocess command from args.
2. Streams stdout in real time (mirrors to wrapper's own stdout).
3. Enforces `TRAIN_TIMEOUT_SECONDS` hard ceiling.
4. On CUDA OOM: halves `--batch_size` and retries once.
5. After subprocess completes: emits `<metric>=<float>` as the final stdout line.
   - If the subprocess already emitted a metric line, re-emits the last one.
   - Otherwise: emits `<metric>=0.0` sentinel.

---

## Dependencies

### Python (pyproject.toml)

```
lerobot-isaac-adapters    (sibling — invoked as subprocess)
```

Dev extras:
```
pytest, pyyaml
```

### Heavy/external dependencies

None directly. `lerobot_isaac_adapters` is only invoked as a subprocess, so all
heavy deps (lerobot, sheeprl, etc.) are in that package's environment.

---

## Configuration

### `LEROBOT_TRAIN_TIMEOUT` env var

Override the 4-hour hard timeout:

```bash
export LEROBOT_TRAIN_TIMEOUT=3600   # 1-hour timeout
```

### OOM recovery

Automatic. When CUDA OOM is detected in subprocess stdout, `batch_size` is halved
and the run retries once. After one retry the process exits with the subprocess
return code regardless.

---

## Running Tests

```bash
cd packages/lerobot-isaac-autoresearch
pytest tests/ -v
```

All tests pass without Isaac Lab or lerobot installed.

---

## Agent Dependencies (read-only)

| Agent | Path |
|-------|------|
| `autoresearch-loop-orchestrator` | `~/.claude/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| `autoresearch-ml-executor-worker` | `~/.claude/agents/workers/autoresearch-ml-executor-worker.md` |
| `autoresearch-ml-proposer-worker` | `~/.claude/agents/workers/autoresearch-ml-proposer-worker.md` |

Source of truth: `${CLAUDE_CODE_ROOT}/agents/`

Do NOT modify agents here — edit source files in `~/tools/claude_code/agents/` and
re-run `install.sh`.

---

## Standalone Spinout

```bash
git subtree split -P packages/lerobot-isaac-autoresearch -b spinout-autoresearch
git checkout spinout-autoresearch
git remote add origin git@github.com:user/lerobot-isaac-autoresearch.git
git push -u origin main
```

Cross-package dependency: only `lerobot-isaac-adapters` (invoked as subprocess).

---

## See Also

- Build plan: `${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 3
- Template reference: `${CLAUDE_CODE_ROOT}/templates/ml-program.md`
