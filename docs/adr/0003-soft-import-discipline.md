# ADR-0003: Soft-Import Discipline for Heavy Dependencies

**Status:** Accepted
**Date:** 2026-05-06
**Deciders:** Project team

---

## Context

This workspace depends on several heavy optional libraries:

| Library | Install size | Reason optional |
|---------|-------------|----------------|
| `isaaclab` + Isaac Sim | ~30 GB | Requires GPU + NVIDIA drivers |
| `lerobot` + torch | ~5 GB | Large but no GPU required for CPU inference |
| `dreamerv3` / sheeprl | ~3 GB | Only needed for world-model training |
| `leworldmodel` | ~2 GB | Only needed for LeWM path |

If any of these are imported at module top-level, the following problems arise:

1. **`import lerobot_isaac_adapters` fails** on a clean dev machine that only has the
   `default` pixi environment (no `lerobot`, no `isaaclab`).
2. **CI default job fails** because the ubuntu runner does not have CUDA or Isaac Sim.
3. **Docs generation fails** (Sphinx / mkdocs auto-imports modules).
4. **Other packages that import this package as a dep also fail**, transitively.

---

## Decision

Heavy optional dependencies are **never imported at module top-level**. All imports of
`isaaclab`, `lerobot`, `dreamerv3`, `leworldmodel`, and `torch` happen inside the
functions that need them, wrapped in a `try/except ImportError` that raises a clear,
actionable error message.

### Pattern

```python
# src/lerobot_isaac_adapters/backends/dreamerv3_backend.py

def train(config):
    try:
        import sheeprl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DreamerV3 backend requires sheeprl. "
            "Install with: pixi install -e train-dreamer"
        ) from exc
    # ... rest of function
```

### Test markers

Tests requiring heavy deps are marked with pytest markers defined in `conftest.py`:

```python
# conftest.py (per package)
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_isaaclab: needs Isaac Lab + GPU")
    config.addinivalue_line("markers", "requires_lerobot: needs lerobot + torch")
    config.addinivalue_line("markers", "requires_dreamerv3: needs dreamerv3/sheeprl")
```

CI default job skips all such markers:

```bash
pytest -m 'not requires_isaaclab and not requires_lerobot and not requires_dreamerv3'
```

### Enforcement

Ruff rule `TID252` (banned module-level imports) is configured in `pyproject.toml` to
flag any top-level import of the heavy deps. This gives a lint error before the test
even runs.

---

## Rationale

### Importability without installs

The primary benefit is that `python -c "import lerobot_isaac_configs"` works on any Python
3.10+ environment with no GPU and no heavy deps. This is required for:

- Documentation generation (Sphinx reads module docstrings)
- IDE autocompletion (language servers import the package)
- Lightweight tools that only need config parsing

### Test architecture impact

Soft imports enable a two-tier test structure:

| Tier | Marker | Runs in CI | Env |
|------|--------|------------|-----|
| Unit | (none) | Always | `default` |
| Integration | `requires_lerobot` etc. | Manual / GPU runner | `train-*` / `sim` |

This mirrors how ROS2 separates unit tests (ament_cmake_gtest) from launch tests.

### Actionable error messages

When a heavy dep is missing, the raised `ImportError` tells the user exactly which
`pixi install -e <env>` command to run. This reduces friction for contributors who
work in only one environment.

---

## Consequences

**Positive:**
- Package importable with no heavy deps
- CI default job is fast (no large installs)
- IDE tooling works everywhere
- Clear error messages guide users to the right install command

**Negative:**
- Import errors surface at function call time, not at module load time (harder to
  detect misconfiguration early)
- Developers must remember to wrap every new heavy-dep import
- Ruff rule requires manual configuration per `pyproject.toml`

**Mitigation:**
- `pre-commit` ruff hook catches violations before commit
- Code review checklist item: "heavy deps wrapped?"
- `conftest.py` templates in each package provide the marker boilerplate

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| `importlib.util.find_spec` guards at top-level | Still runs at import time; same problem |
| Single monolithic environment with all deps | Defeats purpose; 30+ GB install for all contributors |
| Optional extras in `pyproject.toml` only | Does not handle conda/GPU deps; no lazy loading |
| Separate repos per heavy dep | Excessive fragmentation; spinout is the right boundary |
