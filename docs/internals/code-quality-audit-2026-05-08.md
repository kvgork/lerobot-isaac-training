# Code Quality Audit — 2026-05-08

**Session:** `20260508-090936-code-quality-audit`
**Scope:** all 8 packages under `packages/`, `scripts/`, top-level docs, tests.
**Rule sources:** vault concepts (`Python-Best-Practices`, `Code-Smells`,
`Multi-Package-Python-Monorepo`) + ruff defaults (F, E, B, UP, SIM rule families).
**Mode:** report + auto-applied safe fixes.

---

## Baseline

`ruff check packages/ scripts/` (default rule set) reported **133 violations** on
session start; `ruff check ... --select F,E,B,UP,SIM` reported **354** with the
expanded rule set.

## Outcome

- `ruff check packages/ scripts/` → **0 errors**
- `ruff format --check packages/ scripts/` → 131 files already formatted
- `pytest -m "not requires_*"` → **689 passed, 1 skipped, 0 failed**
  (no regressions from baseline 689 passed in the previous session)

---

## Findings & Fixes

### 1. Unused imports / vars / redefinitions

| Rule | Count | Disposition |
|------|-------|-------------|
| `F401` unused-import | 103 | Auto-fixed by `ruff --fix` |
| `F841` unused-variable | 6 | Manually removed |
| `F811` redefinition | 2 | Auto-fixed |
| `F541` f-string-no-placeholder | 7 | Auto-fixed |

**Notable manual fixes:**

- `merge_utilities.py:225` — `path_to_sim` dict was built then never read.
  Replaced with explicit `del sim_paths` + comment so the parameter is still
  accepted for caller-API symmetry.
- `merge_utilities.py:103` — `import shutil` and `import pyarrow.parquet as pq`
  imported lazily but never used. Dropped.
- `parquet_writer.py:241` — `task_index = dataset.meta.tasks` assigned but
  never used. Replaced with comment explaining `add_frame()` reads it itself.
- `isaac_data_recorder.py:170,176` — `import time` + `t0 = time.monotonic()`
  loop start; throughput logging was never wired up. Removed both.
- `tests/test_dry_run.py:166` — `original_import` captured but
  `monkeypatch.setattr` was used instead. Dropped.
- `tests/test_snapshots.py:196,201` — `snap_dir` / `ws` assigned then
  unused after the call. Replaced with `_` discard names.

### 2. Undefined names (F821) — `pd` in string annotations

`merge_utilities.py` annotated parameters with `"pd.DataFrame"` strings but
imported pandas only inside the function body. ruff couldn't see the import.

**Fix:** added a `TYPE_CHECKING` block:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
```

That gives the static checker visibility without paying the import cost at
module load.

### 3. F821 `ManagerBasedRLEnv`

`lerobot-isaac-env/__init__.py` had `make_env()` annotated with the
quoted Isaac Lab type but the type was never imported (even as type-only).
Added a `TYPE_CHECKING`-guarded import that tries both the new
(`isaaclab.envs`) and legacy (`omni.isaac.lab.envs`) namespaces.

### 4. Module-level imports below code (E402)

- `dashboard/app.py` — `logger = logging.getLogger(__name__)` was set BEFORE
  the `from lerobot_isaac_dashboard.* import` block. Moved logger after all
  imports per PEP 8.
- `tests/test_train_wrapper.py` — `sys.path.insert(...)` happens before the
  package import (intentional pattern to make the test runnable without
  install). Added `# noqa: E402  — sys.path inserted above` so the violation
  documents intent rather than being silenced blindly.

### 5. Streamlit availability check (F401)

`dashboard/tabs/_compare.py` did `import streamlit as st` purely to detect
availability. Replaced with `importlib.util.find_spec("streamlit") is not None`
— no symbol bound, no F401 violation.

### 6. Modernization (UP family)

`ruff --select UP006,UP007,UP035,UP037,UP045,SIM300 --fix` applied **113 fixes**:

- `Optional[X]` → `X | None` (UP045 ×34)
- `List[X]` → `list[X]` (UP006 ×27)
- Quoted annotations stripped where unnecessary (UP037 ×45)
- `from typing import Iterable` → `from collections.abc import Iterable`
  (UP035 — manual, deprecated import)
- A pair of yoda conditions (`SIM300`) flipped to natural form

### 7. Duplication / DRY (the biggest finding)

The three training-target modules (`policy_lerobot`, `wm_dreamerv3`,
`wm_leworldmodel`) each contained ~30 lines of identical logic:

```text
subprocess.Popen(...) → handle FileNotFoundError → for line in proc.stdout:
  mirror to stdout + regex.search + emit() → proc.wait() → handle nonzero
```

**Refactor:** extracted to
`packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/_subprocess.py`
exposing `stream_training_subprocess(cmd, *, metric_re, metric_name, label,
install_hint) -> int`. Each target now has a single 7-line tail call:

```python
return stream_training_subprocess(
    cmd, metric_re=_PC_SUCCESS_RE, metric_name="pc_success",
    label="policy_lerobot", install_hint="Install LeRobot: pip install lerobot",
)
```

Net change: ~75 lines deduplicated, single canonical place for stdout handling
+ FileNotFoundError → 127 mapping + ANSI red error tail. Existing 107 adapter
tests still pass without modification (they patch `subprocess.Popen` globally
so the helper inherits the patch).

### 8. Duplicated train_cmd construction within each WM target

Already fixed earlier in `feat(pipelines): finalize Phase 1-4 dry-run smoke`
(commit `3949edf`) by lifting `_build_train_cmd` into a closure used by both
the dry-run banner path and the live subprocess path. No further work.

---

## What was deliberately NOT changed

- **`except Exception:` blocks (15 sites).** All sit in init / config-build
  paths where multiple unrelated failures (USD missing, network error, schema
  mismatch) need to fall back gracefully. Tightening to specific exceptions
  would either lose coverage or grow into long tuples that obscure intent.
  Ruff `BLE001` is not in the enabled rule set, so no markers needed.
- **Soft-import pattern across `lerobot-isaac-env`** (5+ files).
  Each file imports a different subset of Isaac Lab symbols with file-specific
  fallbacks. Extracting a shared `_isaaclab_imports.py` would couple files
  that should remain independently spinout-able.
- **`_convert_dataset` between `wm_dreamerv3` and `wm_leworldmodel`.**
  Two sites with different image-size, window-size, output filename, and error
  message. Per *Code Smells* "rule of three" — borderline; keeping separate.
- **`E501` line-too-long (188 reports).** ruff format already wraps cleanly
  where possible; remaining long lines are mostly in docstring-embedded
  shell commands or URLs where breaking would harm readability.
- **Stale `Common Pitfalls` etc.** Already covered by the previous
  pipeline-finalization commit.

---

## Vault rules cross-reference

| Vault rule | Source | Status |
|------------|--------|--------|
| Type hints + `T \| None` over `Optional` | Python-Best-Practices §1 | enforced via UP045 |
| `list[X]` over `List[X]` | Python-Best-Practices §1 | enforced via UP006 |
| `from collections.abc import …` over `from typing import …` | Python-Best-Practices §1 | enforced via UP035 |
| No bare `except:` | Python-Best-Practices §4 | clean (0 sites) |
| `is None` not `== None` | Python-Best-Practices §4 | clean (0 sites) |
| No mutable default args | Python-Best-Practices anti-patterns | clean (0 sites) |
| No `from x import *` | Python-Best-Practices anti-patterns | clean (0 sites) |
| `_private` prefix for module helpers | PEP 8 / general | clean — all module-private functions audited; no rename needed |
| `pyproject.toml` single source of truth | Python-Best-Practices §3 | already in place per package |
| No sibling cross-imports | Multi-Package-Python-Monorepo | clean (verified during dedup audit) |
| DRY: rule of three before extracting | Code-Smells / Refactoring | applied — extracted `stream_training_subprocess` (3 sites) |

---

## Files touched

102 files changed, 1328 insertions(+), 853 deletions(-) — most of it formatter
output (84 files reformatted). The substantive logic changes live in:

- `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/_subprocess.py` (new)
- `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py`
- `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_dreamerv3.py`
- `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_leworldmodel.py`
- `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/isaac_data_recorder.py`
- `packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/merge_utilities.py`
- `packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/isaac_dr/parquet_writer.py`
- `packages/lerobot-isaac-env/src/lerobot_isaac_env/__init__.py`
- `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/app.py`
- `packages/lerobot-isaac-dashboard/src/lerobot_isaac_dashboard/tabs/_compare.py`

---

## How to re-run this audit

```bash
pixi run -e default ruff check packages/ scripts/
pixi run -e default ruff check packages/ scripts/ --select F,E,B,UP,SIM --statistics
pixi run -e default ruff format --check packages/ scripts/
pixi run -e default pytest packages/ -m "not requires_isaaclab and not requires_lerobot and not requires_dreamerv3 and not integration"
```

The pre-commit config already runs these on every commit via `pre-commit
install`. CI runs the same checks per `.github/workflows/lint.yml`.
