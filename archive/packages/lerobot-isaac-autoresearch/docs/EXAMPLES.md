# lerobot-isaac-autoresearch — Usage Examples

---

## Example 1 — Import the package

```python
import lerobot_isaac_autoresearch
from lerobot_isaac_autoresearch import train_wrapper
print("import ok")
```

Expected output:
```
import ok
```

---

## Example 2 — Parse args and inspect

```python
from lerobot_isaac_autoresearch.train_wrapper import parse_args

args, extra = parse_args([
    "--target_arch", "smolvla",
    "--dataset", "/data/real",
    "--steps", "20000",
    "--batch_size", "8",
])

print(args.target_arch)   # smolvla
print(args.dataset)       # /data/real
print(args.steps)         # 20000
print(args.batch_size)    # 8
print(extra)              # []
```

---

## Example 3 — Dry-run via CLI

```bash
python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch smolvla \
  --dataset /data/real \
  --output_dir /tmp/run \
  --steps 100 \
  --dry_run
```

Expected stdout:
```
[train_wrapper] running: python -m lerobot_isaac_adapters.train --target_arch smolvla --dataset /data/real --output_dir /tmp/run --steps 100 --dry_run
[dry_run] target_arch=smolvla ...
pc_success=0.0
```

---

## Example 4 — Run with OOM simulation (understanding OOM recovery)

This example illustrates the OOM retry logic conceptually. In practice, OOM is
detected when the subprocess stdout contains "cuda out of memory".

```python
from lerobot_isaac_autoresearch.train_wrapper import _detect_oom

lines_no_oom = ["step=100 loss=0.5", "pc_success=0.3"]
lines_oom = ["Traceback...", "RuntimeError: CUDA out of memory"]

print(_detect_oom(lines_no_oom))  # False
print(_detect_oom(lines_oom))     # True
```

---

## Example 5 — Launch autoresearch for LeRobot policy

Full autoresearch run from the `claude_code` repo.

```bash
cd ~/tools/claude_code

/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md \
  --type ml_model
```

The orchestrator will:
1. Parse `programs/lerobot-policy.md` for the search goal and constraints.
2. Spawn `autoresearch-ml-proposer-worker` to suggest initial hyperparameters.
3. Spawn `autoresearch-ml-executor-worker` to run `train_wrapper.py`.
4. After each run, re-propose based on the emitted metric.
5. Stop after `max_experiments=10` or `plateau_limit=3`.

---

## Example 6 — Launch autoresearch for DreamerV3 world model

```bash
cd ~/tools/claude_code

/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md \
  --type ml_model
```

Note: `wm_dreamerv3.run()` is still a stub — see `../../lerobot-isaac-adapters/docs/INTERNALS.md`
for wiring instructions.

---

## Example 7 — Pass extra backend args through wrapper

Extra args after `--` on the CLI are forwarded verbatim to the adapter:

```bash
python -m lerobot_isaac_autoresearch.train_wrapper \
  --target_arch act \
  --dataset /data/real \
  --steps 30000 \
  -- --policy.n_action_steps=100 --policy.chunk_size=50
```

The `-- --policy.n_action_steps=100 --policy.chunk_size=50` portion is passed
through to `lerobot_isaac_adapters.train` unchanged.

---

## Example 8 — Inspect program.md metric config

```python
import re

with open("programs/lerobot-policy.md") as f:
    content = f.read()

# Extract metric regex
m = re.search(r"regex:\s*'([^']+)'", content)
if m:
    print(f"Metric regex: {m.group(1)}")
    # Metric regex: pc_success[=:\s]+([0-9.]+)

# Verify it matches expected metric line
metric_line = "pc_success=0.73"
match = re.search(m.group(1), metric_line)
print(f"Extracted value: {match.group(1)}")  # 0.73
```
