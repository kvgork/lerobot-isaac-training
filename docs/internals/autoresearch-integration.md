# Autoresearch Integration — Internals

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [training-dispatch.md](./training-dispatch.md)
**Agent source:** `/home/koen/tools/claude_code/agents/orchestrators/autoresearch-loop-orchestrator.md`
**Skill source:** `/home/koen/tools/claude_code/skills/autoresearch/`

---

## Overview

The autoresearch loop drives automated hyperparameter search without human intervention.
It follows a simple pattern: propose a mutation, run training, observe the metric, keep if better.
The infrastructure is entirely provided by the `claude_code` repo; this workspace only provides
`program.md` configuration files and a thin `train_wrapper.py` shim.

---

## program.md Schema

Each `packages/lerobot-isaac-autoresearch/programs/*.md` file defines a search problem.
Required fields:

```markdown
# Program: <name>

## Metric
- name: pc_success           # or recon_loss, pred_loss
- direction: maximize        # or minimize
- regex: "pc_success=([0-9.eE+-]+)"   # stdout extraction pattern

## Budget
- seconds_per_experiment: 7200   # wall-clock budget per training run
- max_experiments: 10
- plateau_limit: 3               # stop if N consecutive runs don't improve

## Baseline
- script_path: packages/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
- args:
    target_arch: smolvla
    config: packages/lerobot-isaac-configs/configs/policy_smolvla.yaml
    dataset_path: datasets/so101_pick_v1_filtered

## Hyperparameters
batch_size:
  type: int
  range: [16, 128]
  scale: log2
learning_rate:
  type: float
  range: [1e-5, 1e-3]
  scale: log
num_steps:
  type: int
  range: [50000, 500000]
  scale: log10

## Constraints
allow_architecture_change: false
allow_dataset_change: false
```

Three programs are provided:

| File | Target | Metric | Direction | Budget |
|------|--------|--------|-----------|--------|
| `lerobot-policy.md` | smolvla / act / diffusion | `pc_success` | maximize | 7200 s / run |
| `dreamerv3.md` | DreamerV3 | `recon_loss` | minimize | 7200 s / run |
| `leworldmodel.md` | LeWorldModel | `pred_loss` | minimize | 5400 s / run |

---

## Mutation Operators

`autoresearch-ml-proposer-worker` applies one of 6 operators per experiment:

| Operator | Description |
|----------|-------------|
| `tweak_lr` | Multiply learning rate by random factor in [0.5, 2.0] |
| `tweak_batch` | Change batch_size to adjacent power-of-2 |
| `tweak_steps` | Scale max_steps by 0.5 or 2.0 |
| `tweak_arch_param` | Modify an architecture hyperparameter (e.g. RSSM dim for DreamerV3) |
| `tweak_data_aug` | Toggle or modify a data augmentation parameter |
| `random_restart` | Sample all hyperparameters fresh from their ranges |

For world-model programs, `allow_architecture_change: false` restricts mutations to
optimizer and data-pipeline parameters only. `tweak_arch_param` is disabled.

---

## Metric History

The `autoresearch` skill maintains a metric history file per program:
```
.agent-state/<session>/autoresearch/<program_slug>/
  history.jsonl         # one line per experiment: {config, metric_name, value, timestamp}
  best_config.yaml      # current best hyperparameter config
  plateau_tracker.json  # consecutive non-improving runs count
```

Each entry in `history.jsonl`:
```json
{
  "experiment_id": 3,
  "metric_name": "pc_success",
  "metric_value": 0.71,
  "config": {"batch_size": 64, "learning_rate": 0.0003, "num_steps": 200000},
  "timestamp": "2026-05-06T20:15:33Z",
  "wall_seconds": 6821
}
```

---

## Plateau Detection

After each experiment, the executor worker checks:
```python
recent_values = [h["metric_value"] for h in history[-plateau_limit:]]
if direction == "maximize":
    is_plateau = max(recent_values) <= best_ever_value
else:
    is_plateau = min(recent_values) >= best_ever_value
plateau_count += 1 if is_plateau else 0
if plateau_count >= plateau_limit:
    # terminate loop
```

Plateau limits:
- Policy programs: 3 (3 consecutive non-improving runs)
- World-model programs: 2 (more expensive; tighter budget)

---

## Invoking the Loop

```bash
# From any Claude session with agents installed:
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model
```

Or directly:
```python
# Task(autoresearch-loop-orchestrator, {
#   program_path: "packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md",
#   workspace_root: "~/workspaces/lerobot-isaac-training"
# })
```

The orchestrator is responsible for:
1. Parsing `program.md`
2. Running the proposer worker to generate mutations
3. Running the executor worker to call `train_wrapper.py`
4. Logging results and checking plateau
5. Selecting the next mutation based on history

---

## Metric Emission Contract

Every training script called by `train_wrapper.py` MUST emit a line of the form:
```
<metric_name>=<float>
```
on stdout at each evaluation step. The executor worker reads the **last** matching line.

Example valid emissions:
```
pc_success=0.73
recon_loss=0.0317
pred_loss=0.0421
```

Invalid (will not be parsed):
```
pc_success: 0.73           # colon instead of equals
Final pc_success = 0.73    # leading text with spaces around equals
```

The `MetricEmitter` class in `metric_extractor.py` guarantees the correct format.
Always use it — never print metrics manually.
