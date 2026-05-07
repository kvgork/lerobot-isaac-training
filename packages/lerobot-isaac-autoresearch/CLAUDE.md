# lerobot-isaac-autoresearch — Package Orientation

**Role:** Autoresearch ML loop wiring. Holds `program.md` configs consumed by
`autoresearch-loop-orchestrator` and a thin `train_wrapper.py` shim.
**Phase:** 3 — implemented.
**Status:** Functional. train_wrapper handles OOM recovery and metric guarantee.

---

## What This Package Does

Two responsibilities:
1. **`programs/`** — Three `program.md` files consumed by `autoresearch-loop-orchestrator`.
   Each defines the search goal, training script path, metric name/regex, budget, and
   hyperparameter search space for one training target family.
2. **`train_wrapper.py`** — Thin shim invoked by `autoresearch-ml-executor-worker`.
   Forwards args to `lerobot_isaac_adapters.train` via subprocess, captures stdout,
   guarantees the last stdout line is `<metric>=<float>`, and handles CUDA OOM by
   halving `batch_size` once.

---

## Public API

- `train_wrapper.main()` — CLI entrypoint registered as `lerobot-isaac-train-wrapper`
- `train_wrapper.run(args: argparse.Namespace) -> int` — executes with OOM retry
- `train_wrapper.parse_args(argv=None) -> (namespace, extra_list)` — arg parsing
- `TRAIN_TIMEOUT_SECONDS` — module-level constant (default 14400s / 4h)
  Override via `LEROBOT_TRAIN_TIMEOUT` env var.

---

## Package Map

```
packages/lerobot-isaac-autoresearch/
├── programs/
│   ├── lerobot-policy.md    # SmolVLA/ACT/Diffusion — pc_success maximize
│   ├── dreamerv3.md         # DreamerV3 world model — recon_loss minimize
│   └── leworldmodel.md      # HF LeWorldModel      — pred_loss minimize
├── src/lerobot_isaac_autoresearch/
│   ├── __init__.py
│   └── train_wrapper.py     # shim → lerobot_isaac_adapters.train subprocess
└── tests/
    ├── test_programs_parse.py   # validates program.md YAML keys
    └── test_train_wrapper.py    # argparse smoke test, OOM detection
```

---

## Dependency on Sibling Packages (plan §11.6)

- Only depends on `lerobot-isaac-adapters` (invoked as subprocess).
- Does NOT import `lerobot-isaac-adapters` at module load — subprocess only.
- No circular deps.

---

## train_wrapper OOM Recovery Logic

```python
while True:
    returncode, stdout_lines = _run_subprocess(cmd, timeout)
    if returncode != 0 and _detect_oom(stdout_lines) and retry_count < 1:
        batch_size = max(1, batch_size // 2)
        retry_count += 1
        continue
    break
```

OOM detection checks for any of these substrings in stdout (case-insensitive):
- `"cuda out of memory"`
- `"out of memory"`
- `"cudaoutofmemoryerror"`
- `"runtimeerror: cuda"`

---

## Metric Guarantee Logic

After subprocess completes, `train_wrapper` searches stdout (in reverse) for the last
line containing `<metric_name>=`. If found, re-emits it as the final stdout line.
If not found, emits `<metric_name>=0.0` sentinel so the executor doesn't crash.

---

## Agent Source of Truth (read-only references)

- `~/.claude/agents/orchestrators/autoresearch-loop-orchestrator.md`
- `~/.claude/agents/workers/autoresearch-ml-executor-worker.md`
- `~/.claude/agents/workers/autoresearch-ml-proposer-worker.md`
- `/home/koen/tools/claude_code/skills/autoresearch/`

Installed copies live at `~/.claude/agents/`. Do NOT edit agents here.

---

## How to Run Autoresearch

```bash
cd ~/tools/claude_code
/autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model
```

---

## How to Add a New Program

1. Create `programs/<name>.md` following the schema in `programs/lerobot-policy.md`.
2. Set `path:` to `src/lerobot_isaac_autoresearch/train_wrapper.py`.
3. Set `metric.name`, `metric.direction`, `metric.regex` to match the emitted metric.
4. Add a test in `tests/test_programs_parse.py` that validates the required YAML keys.

---

## Testing Notes

- `test_programs_parse.py` — parses each `.md` file for required YAML keys
  (`## Training Script`, `path:`, `## Metric`, `name:`, `regex:`).
- `test_train_wrapper.py` — `parse_args()` smoke test; OOM detection logic.

All tests pass without any external deps.

---

## Spinout Note

```bash
git subtree split -P packages/lerobot-isaac-autoresearch -b spinout-autoresearch
```

See `../../docs/ARCHITECTURE.md` (spinout section).

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 3 / §11.8
- Template reference: `/home/koen/tools/claude_code/templates/ml-program.md`
- LeRobot program template: `/home/koen/tools/claude_code/templates/lerobot-program.md`
