# lerobot-isaac-autoresearch

Holds the `program.md` configs that drive the autoresearch ML loop for LeRobot
policies and world models. Wraps the adapters' `train.py` so
`autoresearch-ml-executor-worker` can run it via a single stable entrypoint.

## Purpose

This package provides:

1. **`programs/`** — Three `program.md` files (one per training target family),
   each consumed by `autoresearch-loop-orchestrator` to run autonomous
   hyperparameter search.
2. **`src/lerobot_isaac_autoresearch/train_wrapper.py`** — A thin shim that
   forwards CLI args to `lerobot_isaac_adapters.train` so the executor worker
   always points at one stable path regardless of which program is running.

## Programs

| File | Target | Metric | Direction |
|------|--------|--------|-----------|
| `programs/lerobot-policy.md` | SmolVLA / ACT / Diffusion | `pc_success` | maximize |
| `programs/dreamerv3.md` | DreamerV3 world model | `recon_loss` | minimize |
| `programs/leworldmodel.md` | HF LeWorldModel | `pred_loss` | minimize |

## How to Run

Invoke `autoresearch-loop-orchestrator` from the workspace root
(`~/workspaces/lerobot-isaac-training`) or from inside
`~/tools/claude_code` using the `/autoresearch` slash command:

```bash
cd ~/tools/claude_code

# LeRobot policy search (SmolVLA / ACT / Diffusion)
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model

# DreamerV3 world model search
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md --type ml_model

# HF LeWorldModel search
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/leworldmodel.md --type ml_model
```

The orchestrator auto-spawns `autoresearch-ml-executor-worker` (runs training)
and `autoresearch-ml-proposer-worker` (proposes mutations).

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
1. Forwards all args to `python -m lerobot_isaac_adapters.train`
2. Captures stdout and ensures a metric line (`pc_success=0.XXXX`) is the
   final line emitted (compatible with the executor's default regex)
3. Handles CUDA OOM by halving `--batch_size` and retrying once
4. Enforces a 4-hour timeout hard ceiling

## Agent Dependencies (do NOT modify these)

| Agent | Path |
|-------|------|
| `autoresearch-loop-orchestrator` | `~/.claude/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| `autoresearch-ml-executor-worker` | `~/.claude/agents/workers/autoresearch-ml-executor-worker.md` |
| `autoresearch-ml-proposer-worker` | `~/.claude/agents/workers/autoresearch-ml-proposer-worker.md` |

Source of truth: `/home/koen/tools/claude_code/agents/`

## Standalone Spinout

```bash
git subtree split -P packages/lerobot-isaac-autoresearch -b spinout-autoresearch
```

Cross-package dependency: only `lerobot-isaac-adapters` (for `train.py`).

## See Also

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 3 / Section 11.8
- Template reference: `/home/koen/tools/claude_code/templates/ml-program.md`
- LeRobot program template: `/home/koen/tools/claude_code/templates/lerobot-program.md`
