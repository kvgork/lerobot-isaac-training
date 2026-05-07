# lerobot-isaac-meta — Usage Examples

Examples increase in complexity. All examples in sections 1–4 require no
external dependencies beyond the workspace packages.

---

## Example 1 — Import and verify scaffold

Verifies the package is installed and the workspace root is found.

```python
import lerobot_isaac_meta

print(lerobot_isaac_meta.__version__)  # 0.1.0
print("ok")
```

Expected output:
```
0.1.0
ok
```

No dependencies required.

---

## Example 2 — Resolve workspace paths

Access all workspace path constants.

```python
from lerobot_isaac_meta.workspace_paths import (
    WORKSPACE_ROOT,
    DATASETS_DIR,
    OUTPUTS_DIR,
    CONFIGS_DIR,
    AGENT_STATE_DIR,
)

print(WORKSPACE_ROOT)
# /home/user/workspaces/lerobot-isaac-training

print(DATASETS_DIR)
# /home/user/workspaces/lerobot-isaac-training/datasets

print(CONFIGS_DIR)
# /home/user/workspaces/lerobot-isaac-training/packages/lerobot-isaac-configs/configs
```

Expected output: absolute paths under the workspace root. All paths are `Path`
objects; call `.exists()` to check if directories are present.

---

## Example 3 — Initialise workspace dirs before a training run

Ensure all runtime directories exist before launching any training or data-collection job.

```python
from lerobot_isaac_meta.workspace_paths import ensure_dirs, DATASETS_DIR, OUTPUTS_DIR

ensure_dirs()
print(DATASETS_DIR.is_dir())   # True
print(OUTPUTS_DIR.is_dir())    # True
```

Expected output:
```
True
True
```

Call this at the start of any training script or agent run.

---

## Example 4 — Override workspace root with env var

Useful in CI pipelines or Docker containers where the workspace is mounted at
a non-standard path.

```python
import os
import tempfile
from pathlib import Path

# Create a fake workspace directory
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")

    os.environ["LEROBOT_ISAAC_WORKSPACE"] = str(tmp_path)

    import importlib
    import lerobot_isaac_meta.workspace_paths as wp
    importlib.reload(wp)

    print(wp.WORKSPACE_ROOT == tmp_path.resolve())  # True

    del os.environ["LEROBOT_ISAAC_WORKSPACE"]
    importlib.reload(wp)   # restore
```

Expected output:
```
True
```

---

## Example 5 — Use CLI from shell

Invoke the CLI and verify it responds correctly.

```bash
# Verify installation
lerobot-isaac --version
# lerobot-isaac 0.1.0 (Phase 0 scaffold)

# Show top-level help
lerobot-isaac --help

# Show subcommand help (stubs — print guidance on what to implement)
lerobot-isaac train --help
lerobot-isaac dr-replay --help

# Invoke a stub subcommand — prints "not yet wired" with implementation hint
lerobot-isaac train
```

Expected output for `lerobot-isaac train`:
```
lerobot-isaac train: not yet wired — see Phase 2 (packages/lerobot-isaac-adapters).
When implemented: python -m lerobot_isaac_adapters.train --target_arch <arch> ...
```

---

## Example 6 — Build the parser programmatically

Useful for testing or for inspecting available subcommands.

```python
from lerobot_isaac_meta.cli import build_parser

parser = build_parser()
subparsers_action = None
for action in parser._actions:
    if hasattr(action, '_name_parser_map'):
        subparsers_action = action
        break

if subparsers_action:
    print(list(subparsers_action._name_parser_map.keys()))
```

Expected output:
```
['train', 'record', 'dr-replay', 'mimicgen-augment']
```

---

## Example 7 — Full integration: resolve path and load config (with lerobot-isaac-configs)

Demonstrates how meta + configs packages work together.

```python
# Requires: lerobot-isaac-configs (pip install -e packages/lerobot-isaac-configs)
from lerobot_isaac_meta.workspace_paths import CONFIGS_DIR
from lerobot_isaac_configs import load_config

# List available configs
import os
available = [f.stem for f in CONFIGS_DIR.glob("*.yaml")]
print("Available configs:", available)

# Load the DreamerV3 config
cfg = load_config("wm_dreamerv3")
print(cfg)
```

Expected output (stub configs):
```
Available configs: ['isaac_so101_pickplace', 'policy_act', 'policy_diffusion',
                    'policy_smolvla', 'wm_dreamerv3', 'wm_leworldmodel']
{'image_size': 64, ...}
```

Note: `lerobot-isaac-configs` must be installed. See
`../../lerobot-isaac-configs/README.md` for installation.
