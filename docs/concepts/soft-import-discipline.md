# Soft Import Discipline — Design Concept

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [modular-training-adapter.md](./modular-training-adapter.md)

---

## Why Heavy Deps Are Lazy

This workspace has six heavy dependency groups:
- `isaaclab` — requires Isaac Sim, NVIDIA GPU, ~15 GB disk
- `lerobot` — requires PyTorch + several robotics libraries
- `sheeprl` — requires Lightning + DreamerV3 dependencies
- `transformers` (LeWorldModel) — large Hugging Face stack
- `robosuite` + `mimicgen` — MuJoCo-dependent simulation stack

If any of these were top-level imports, then:
- `pytest` would fail on a machine without a GPU
- CI would require all heavy deps installed
- Developers working on docs or configs would need the full stack
- Import-time errors would prevent even `--help` from working

The solution: **every heavy dep import is deferred to the function body where it is first used**.

---

## The Pattern

```python
# WRONG — module-level import fails at import time if dep is absent
import isaaclab
from isaaclab.envs import ManagerBasedRLEnv

class MyEnv:
    def reset(self):
        return self._env.reset()

# CORRECT — import deferred inside the function/class that needs it
class MyEnv:
    def reset(self):
        try:
            from isaaclab.envs import ManagerBasedRLEnv  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "isaaclab not installed. Run: pixi run install-isaac-lab"
            ) from e
        return self._env.reset()
```

For class-level instantiation, use a lazy property:
```python
class ReplayRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self._env = None  # deferred

    @property
    def env(self):
        if self._env is None:
            try:
                from lerobot_isaac_env import make_env
            except ImportError as e:
                raise ImportError("lerobot_isaac_env not importable") from e
            self._env = make_env(self.cfg.env_id)
        return self._env
```

---

## Enforcement Rule

The pattern is enforced by a ruff rule in `pyproject.toml`:

```toml
[tool.ruff.lint]
# PLC0415 = import-outside-toplevel
# We ALLOW this (it is our intentional pattern for heavy deps)
# The "allowed" list is the approved heavy deps:
per-file-ignores = {"*/targets/*.py" = ["PLC0415"], "*/isaac_dr/*.py" = ["PLC0415"]}
```

The linter still flags any heavy import at module level in non-approved files
(e.g. `lerobot_isaac_configs`, `lerobot_isaac_autoresearch`).

---

## How to Add a New Heavy Dep

1. Identify the feature group: `isaaclab`, `lerobot`, `dreamerv3`, `leworldmodel`, or new.

2. Add to `pixi.toml` under the appropriate feature:
   ```toml
   [feature.myfeature.dependencies]
   my-package = ">=1.0"
   ```

3. Add the feature to the appropriate environment in `pixi.toml`:
   ```toml
   [environments]
   train-myfeature = {features = ["dev", "lerobot", "myfeature"]}
   ```

4. In the code, always soft-import with a helpful error message:
   ```python
   def do_thing():
       try:
           import my_package
       except ImportError as e:
           raise ImportError(
               "my-package not installed. "
               "Run: pixi install -e train-myfeature  or  pip install my-package>=1.0"
           ) from e
       my_package.run(...)
   ```

5. In `docs/api-reference.md`, document the dep requirement in the function docstring.

---

## Test Strategy for Soft Imports

Tests in `packages/*/tests/` are designed to pass WITHOUT any heavy deps installed.
They use either:

1. **Stubs** — `lerobot_isaac_env` exports no-op stubs when Isaac Lab is absent:
   ```python
   try:
       from isaaclab.envs import ManagerBasedRLEnv as _RealEnv
       _STUB_MODE = False
   except ImportError:
       class _RealEnv: pass
       _STUB_MODE = True
   ```

2. **`unittest.mock.patch`** — tests patch the heavy import:
   ```python
   from unittest.mock import patch, MagicMock
   with patch.dict("sys.modules", {"isaaclab": MagicMock()}):
       from lerobot_isaac_env import make_env
       env = make_env("Isaac-SO101-Pick-v0")  # uses mock
   ```

3. **`pytest.importorskip`** — integration tests skip if dep is absent:
   ```python
   isaaclab = pytest.importorskip("isaaclab", reason="Isaac Lab not installed")
   ```

All three strategies ensure `pixi run test` passes in the default (dev) environment.
