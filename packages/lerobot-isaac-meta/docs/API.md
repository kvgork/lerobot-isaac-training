# lerobot-isaac-meta — Public API Reference

This document covers every exported symbol in `lerobot_isaac_meta`.

---

## Module: `lerobot_isaac_meta`

Top-level package. Imports `cli` and `workspace_paths` sub-modules.

```python
import lerobot_isaac_meta
print(lerobot_isaac_meta.__version__)  # "0.1.0"
```

**Exports:** `cli`, `workspace_paths`, `__version__`

---

## Module: `lerobot_isaac_meta.cli`

CLI entrypoint module.

### `build_parser() -> argparse.ArgumentParser`

Returns the top-level argument parser with all subcommands registered.

| Parameter | Type | Description |
|-----------|------|-------------|
| *(none)* | — | — |

**Returns:** `argparse.ArgumentParser` with `train`, `record`, `dr-replay`,
`mimicgen-augment` subparsers registered.

**Example:**

```python
from lerobot_isaac_meta.cli import build_parser

parser = build_parser()
args = parser.parse_args(["train"])
print(args.subcommand)  # "train"
```

---

### `main(argv: list[str] | None = None) -> int`

CLI entrypoint. Registered as the `lerobot-isaac` console script.

| Parameter | Type | Description |
|-----------|------|-------------|
| `argv` | `list[str] \| None` | Argument list. If `None`, reads `sys.argv[1:]`. |

**Returns:** `int` exit code (0 for success, non-zero for error).

**Raises:** Nothing — all errors are printed to stderr and return non-zero.

**Example:**

```python
from lerobot_isaac_meta.cli import main

main(["--version"])      # prints version and exits 0
main(["--help"])         # prints help and exits 0
main(["train"])          # prints stub message, exits 0
main(["unknown-cmd"])    # prints error, exits 2
```

---

## Module: `lerobot_isaac_meta.workspace_paths`

Canonical workspace path resolver.

### Constants

All constants are `pathlib.Path` objects resolved at module import time.

| Constant | Value | Description |
|----------|-------|-------------|
| `WORKSPACE_ROOT` | resolved at import | Workspace root directory. |
| `DATASETS_DIR` | `WORKSPACE_ROOT / "datasets"` | Dataset storage directory. |
| `OUTPUTS_DIR` | `WORKSPACE_ROOT / "outputs"` | Training output directory. |
| `CONFIGS_DIR` | `WORKSPACE_ROOT / "packages/lerobot-isaac-configs/configs"` | YAML config files. |
| `AGENT_STATE_DIR` | `WORKSPACE_ROOT / ".agent-state"` | Autoresearch agent state storage. |

**Resolution order for `WORKSPACE_ROOT`:**

1. `LEROBOT_ISAAC_WORKSPACE` environment variable (if set and valid directory)
2. `__file__`-relative: walk 4 levels up from `workspace_paths.py`

---

### `ensure_dirs() -> None`

Idempotently create gitignored workspace runtime directories.

| Parameter | Type | Description |
|-----------|------|-------------|
| *(none)* | — | — |

**Returns:** `None`

**Raises:**
- `OSError` if directory creation fails (e.g., permissions).

**Side effects:** Creates `DATASETS_DIR`, `OUTPUTS_DIR`, `AGENT_STATE_DIR` with
`parents=True, exist_ok=True`. Does NOT create `CONFIGS_DIR` (managed by the package).

**Example:**

```python
from lerobot_isaac_meta.workspace_paths import ensure_dirs, DATASETS_DIR

ensure_dirs()
assert DATASETS_DIR.is_dir()
```

---

## Cross-Package References

- `CONFIGS_DIR` points into `../../lerobot-isaac-configs/docs/API.md`
- `DATASETS_DIR` is the dataset root consumed by `../../lerobot-isaac-synthetic/docs/API.md`
  merge utilities
- `OUTPUTS_DIR` is the training output root used by `../../lerobot-isaac-adapters/docs/API.md`
