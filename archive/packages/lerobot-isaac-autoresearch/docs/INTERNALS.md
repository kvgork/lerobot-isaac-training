# lerobot-isaac-autoresearch — Internals

---

## File Structure Walk-through

```
packages/lerobot-isaac-autoresearch/
├── pyproject.toml              — dep: lerobot-isaac-adapters; dev: pytest, pyyaml
├── pixi.toml
├── README.md / CLAUDE.md / docs/
├── programs/
│   ├── lerobot-policy.md       — SmolVLA/ACT/Diffusion program config
│   ├── dreamerv3.md            — DreamerV3 world model program config
│   └── leworldmodel.md         — HF LeWorldModel program config
├── src/
│   └── lerobot_isaac_autoresearch/
│       ├── __init__.py
│       └── train_wrapper.py    — subprocess shim with OOM recovery + metric guarantee
└── tests/
    ├── test_programs_parse.py   — validates program.md required YAML keys
    └── test_train_wrapper.py    — argparse, OOM detection, metric extraction
```

---

## Key Data Structures

### Subprocess command construction (`_build_cmd`)

```python
cmd = [
    sys.executable, "-m", "lerobot_isaac_adapters.train",
    "--target_arch", args.target_arch,
]
# + optional flags: --dataset, --output_dir, --steps, --config, --batch_size, --dry_run
# + args.extra (unrecognised args forwarded verbatim)
```

The wrapper always invokes `lerobot_isaac_adapters.train` as a subprocess. This
isolation boundary means the adapter's heavy deps (lerobot, sheeprl) are loaded in
the subprocess environment, not the wrapper's.

### OOM detection strings

```python
oom_signals = [
    "cuda out of memory",
    "out of memory",
    "cudaoutofmemoryerror",
    "runtimeerror: cuda",
]
```

All compared case-insensitively against the joined stdout lines.

### Metric extraction (`_last_metric_line`)

Searches `stdout_lines` in reverse for a line containing `<metric_name>=`.
Splits each line into tokens and returns the first token starting with the
pattern. Returns `None` if not found, causing a sentinel to be emitted.

### Metric name mapping

```python
metric_map = {
    "smolvla": "pc_success",
    "act": "pc_success",
    "diffusion": "pc_success",
    "dreamerv3": "recon_loss",
    "le_world_model": "pred_loss",
}
```

This maps `--target_arch` to the expected metric name for sentinel generation.

---

## Soft-Import Strategy

No soft imports in `train_wrapper.py`. It only uses:
- Python stdlib (`argparse`, `os`, `subprocess`, `sys`, `time`)
- No imports from `lerobot_isaac_adapters` at module load

The subprocess isolation is the soft-import strategy here: heavy deps are fully
contained in the subprocess environment.

---

## program.md Schema

Each `program.md` file follows the autoresearch program format:

```markdown
# <Title>

## Research Goal
<text describing the optimization objective>

## Training Script
path: <relative path from package root>
entry_args: "--target_arch ... --dataset {dataset} --output_dir {out} --steps {steps}"

## Metric
name: <metric_name>
direction: maximize | minimize
source: stdout
regex: '<regex capturing the float value>'

## Budget
seconds_per_experiment: <int>
max_experiments: <int>
plateau_limit: <int>

## Hyperparameter Search Space
<yaml block>
```

`{dataset}`, `{out}`, `{steps}` are template variables filled by the proposer worker.

---

## Test Architecture

Two test files, no external deps:

- `test_programs_parse.py` — reads each `.md` file, checks for required section
  headers and key names using regex. Validates:
  - `## Training Script` + `path:` present
  - `## Metric` + `name:` + `regex:` present
  - `direction:` is one of `maximize`/`minimize`
- `test_train_wrapper.py` — verifies:
  - `parse_args()` accepts all 5 archs
  - `_detect_oom()` returns True/False correctly for sample lines
  - `_last_metric_line()` extracts correct token from sample lines
  - `_build_cmd()` produces a list starting with `sys.executable`

---

## Subprocess Timeout Mechanism

`_run_subprocess` uses `subprocess.Popen` with `stdout=subprocess.PIPE`. It reads
stdout line by line in a loop and checks `time.monotonic()` elapsed against
`TRAIN_TIMEOUT_SECONDS` on each line. If timeout exceeded:

1. `proc.kill()` is called.
2. A `[train_wrapper] TIMEOUT after N s — killed` line is written to stdout.
3. Returns `(-1, stdout_lines_so_far)`.

This is a cooperative timeout on the read loop, not a thread-based timeout, so it
only fires between lines. Long-running subprocesses that produce no output are
handled by the line-level check.

---

## Known Limitations

1. **Timeout fires on line boundaries** — a subprocess that hangs without writing
   any stdout will not be killed until it resumes writing. For complete hang protection,
   the autoresearch executor's own `budget_seconds` watchdog provides an outer boundary.

2. **OOM retry is once only** — `max_retries = 1`. If the halved batch size still
   causes OOM, the run exits with non-zero code.

3. **`parse_known_args` for extra args** — unrecognised args are captured as `args.extra`
   via `parse_known_args`. This means invalid flags are silently forwarded to the adapter
   rather than rejected. The adapter's argparse will reject them.

---

## Future Work

| Item | Plan |
|------|------|
| `--lr` forwarding | Currently not forwarded by `_build_cmd()`; add when hyperparameter search needs it |
| Multi-GPU aware OOM | Current OOM halves to `max(1, batch_size//2)`; could also try `--num_gpus` reduction |
| Program validation | Add `test_programs_parse.py` check that `entry_args` template vars match proposer contract |
