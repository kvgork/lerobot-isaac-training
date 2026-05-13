# lerobot-isaac-autoresearch — Public API Reference

---

## Module: `lerobot_isaac_autoresearch.train_wrapper`

Thin shim invoked by `autoresearch-ml-executor-worker`.

---

### `TRAIN_TIMEOUT_SECONDS: int`

Hard timeout ceiling in seconds. Default: `14400` (4 hours).
Override via `LEROBOT_TRAIN_TIMEOUT` environment variable.

```python
import os
os.environ["LEROBOT_TRAIN_TIMEOUT"] = "3600"   # 1-hour limit
```

---

### `FALLBACK_METRIC_LINE: str`

Sentinel emitted when no metric line is found in subprocess stdout.
Value: `"pc_success=0.0"`.

---

### `parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]`

Parse train_wrapper CLI arguments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `argv` | `list[str] \| None` | Argument list. If `None`, reads `sys.argv[1:]`. |

**Returns:** `(namespace, extra_args)` where `namespace.extra` holds unrecognised args.

**Namespace attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_arch` | `str` | required | One of `smolvla`, `act`, `diffusion`, `dreamerv3`, `le_world_model`. |
| `dataset` | `str \| None` | `None` | Dataset path or HuggingFace repo id. |
| `output_dir` | `str \| None` | `None` | Output directory for checkpoints. |
| `steps` | `int \| None` | `None` | Training steps. |
| `config` | `str \| None` | `None` | Config YAML path. |
| `batch_size` | `int \| None` | `None` | Batch size (may be halved on OOM). |
| `dry_run` | `bool` | `False` | Pass `--dry_run` to adapter. |
| `extra` | `list[str]` | `[]` | Unknown args forwarded verbatim. |

---

### `run(args: argparse.Namespace) -> int`

Execute the training run with optional OOM retry.

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `argparse.Namespace` | Parsed args from `parse_args()`. |

**Returns:** `int` — process exit code (0 = success).

**Behaviour:**
1. Builds a `python -m lerobot_isaac_adapters.train` subprocess command from `args`.
2. Streams stdout in real time; mirrors to wrapper's own stdout.
3. On CUDA OOM and `retry_count < 1`: halves `batch_size`, retries.
4. After subprocess: re-emits last `<metric>=<float>` line as final stdout.
   Emits `<metric>=0.0` sentinel if no metric found.

**Side effects:** Writes to `sys.stdout` (metric line is always emitted as last line).

**Example:**
```python
from lerobot_isaac_autoresearch.train_wrapper import parse_args, run

args, _ = parse_args([
    "--target_arch", "smolvla",
    "--dataset", "/data",
    "--steps", "100",
    "--dry_run",
])
rc = run(args)
print(f"exit code: {rc}")  # 0
```

---

### `main() -> None`

Console script entrypoint. Calls `sys.exit(run(args))`.
Registered as `lerobot-isaac-train-wrapper` in `pyproject.toml`.

---

### Internal helpers (not public API)

| Function | Description |
|----------|-------------|
| `_build_cmd(args)` | Constructs subprocess command list from namespace |
| `_run_subprocess(cmd, timeout)` | Runs command, mirrors stdout, enforces timeout |
| `_detect_oom(stdout_lines)` | Returns True if any line indicates CUDA OOM |
| `_last_metric_line(stdout_lines, metric_name)` | Returns last `<name>=<float>` token or None |

---

## Programs Schema Reference

Each `programs/*.md` file is parsed by `autoresearch-loop-orchestrator`. Required sections:

| Section header | Required keys | Description |
|---------------|---------------|-------------|
| `## Training Script` | `path:`, `entry_args:` | Script path relative to package root; arg template |
| `## Metric` | `name:`, `direction:`, `regex:` | Metric name, `maximize`/`minimize`, stdout regex |
| `## Budget` | `seconds_per_experiment:`, `max_experiments:` | Time/count constraints |

Optional sections: `## Research Goal`, `## Constraints`, `## Hyperparameter Search Space`,
`## Operators Priority`.

---

## Cross-Package References

- Metric format matches `../../lerobot-isaac-adapters/docs/API.md` — `metric_extractor.emit()`
- Subprocess invokes `../../lerobot-isaac-adapters/docs/API.md` — `train.main()`
