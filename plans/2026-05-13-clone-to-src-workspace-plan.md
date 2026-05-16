# Plan: Clone-to-`src/` Editable Workspace (Phase C)

**Date:** 2026-05-13
**Status:** Plan only. No destructive ops.
**Predecessors:**
- `plans/2026-05-12-package-spinout-plan.md` (Phase A — extract to bare repos)
- `plans/2026-05-13-thin-meta-repo-plan.md` (Phase B — meta-only monorepo)
**Tag pin:** `post-spinout-2026-05-13` (HEAD `bfa0ff6`)

---

## TL;DR

You already have separate repos (7 bare repos at `~/workspaces/spinouts/<name>.git`).
**You do NOT need to "make them separate first" — they are.**

This plan switches `pixi install` from "fetch sibling repos as **non-editable** `git+file://`
pip installs into `.pixi/envs/`" → to "**clone** sibling repos into `src/<name>/` and install
them **editable**." Edits in `src/<name>/` hot-reload; commits/pushes go back to each repo's
own remote.

End-state layout:
```
lerobot-isaac-training/                 ← THIS monorepo (meta only)
├── packages/lerobot-isaac-meta/        ← workspace member, editable
├── src/                                ← NEW. Each subdir is its own git clone.
│   ├── lerobot-isaac-configs/          ← clone of ~/workspaces/spinouts/lerobot-isaac-configs.git
│   ├── lerobot-isaac-dashboard/
│   ├── lerobot-isaac-autoresearch/
│   ├── lerobot-isaac-env/
│   ├── lerobot-isaac-adapters/
│   └── lerobot-isaac-synthetic/
├── deps.repos                          ← NEW. YAML manifest (vcstool format).
├── pixi.toml                           ← UPDATED. Editable path deps to src/<name>/.
├── scripts/install.sh                  ← UPDATED. Calls `vcs import` before pixi install.
└── archive/packages/                   ← keep one more session, then delete.
```

`pixi install` flow becomes:
1. `pixi run sync` → `vcs import src/ < deps.repos` (clones missing repos, no-op for existing).
2. `pixi install` → resolves `[pypi-dependencies]` editable path deps.

---

## Design Decisions (with recommendations)

### D1. Tooling: vcstool

**Recommended.** ROS-world standard. Idempotent re-runs. YAML pin per repo. Clean CLI.

Install via pixi (`vcstool` is on conda-forge as `python-vcstool`):
```toml
[feature.dev.dependencies]
python-vcstool = "*"
```

Alternatives considered:
- **Bash loop in `scripts/install.sh`** — works, but no version pinning, no `.repos` manifest.
- **Git submodules** — rejected. Pinned to a single commit; ergonomically awful for daily dev.

### D2. Editable install: pixi `[pypi-dependencies]` with `path` + `editable = true`

**Recommended.** Pixi-native, no `pyproject.toml` direct-URL hackery.

```toml
[pypi-dependencies]
lerobot-isaac-meta = { path = "packages/lerobot-isaac-meta", editable = true }
lerobot-isaac-configs = { path = "src/lerobot-isaac-configs", editable = true }
lerobot-isaac-dashboard = { path = "src/lerobot-isaac-dashboard", editable = true }
lerobot-isaac-autoresearch = { path = "src/lerobot-isaac-autoresearch", editable = true }
lerobot-isaac-env = { path = "src/lerobot-isaac-env", editable = true }
lerobot-isaac-adapters = { path = "src/lerobot-isaac-adapters", editable = true }
lerobot-isaac-synthetic = { path = "src/lerobot-isaac-synthetic", editable = true }
```

The existing `[project.optional-dependencies].post-spinout` block in
`packages/lerobot-isaac-meta/pyproject.toml` stays as-is — it's the production install
path used by `scripts/install.sh` for standalone (non-monorepo) installs.

### D3. Which packages go into `src/`

**6 packages** (the meta deps). Recorder is standalone — clone manually if needed.

| Package | In src/ by default? | Why |
|---------|---------------------|-----|
| lerobot-isaac-configs | yes | meta dep |
| lerobot-isaac-dashboard | yes | meta dep |
| lerobot-isaac-autoresearch | yes | meta dep |
| lerobot-isaac-env | yes | meta dep |
| lerobot-isaac-adapters | yes | meta dep |
| lerobot-isaac-synthetic | yes | meta dep |
| robot-data-recorder | no | standalone hardware-tier, not meta dep |

Document recorder opt-in (one-liner clone) in the runbook.

### D4. Source URL parameterization

**Recommended.** Use `${LEROBOT_SPINOUTS_BASE}` env var in `deps.repos`. Defaults to
`file:///home/koen/workspaces/spinouts`. Flip to `https://github.com/kvgork` once published.

vcstool expands env vars in URLs natively (verified — supports `${VAR}` interpolation).

### D5. Fate of `archive/packages/`

**Recommended: keep for one more session.** Currently 27 references across `pixi.toml`,
runbooks, `pyproject.toml`. Delete in a follow-up session once `src/` proves out and refs
are migrated to `src/`.

### D6. `.gitignore` for `src/`

Each `src/<name>/` is its own git repo. Monorepo must NOT track them. Add:
```
# Cloned sibling repos (each is its own git repo with its own remote)
src/
```

---

## Concrete File Changes

### New file: `deps.repos`

```yaml
# vcstool manifest. Used by `pixi run sync` to clone sibling repos into src/.
#
# Override the URL base with $LEROBOT_SPINOUTS_BASE (defaults below to local bare repos):
#   export LEROBOT_SPINOUTS_BASE=https://github.com/kvgork
#   pixi run sync
#
# Idempotent: re-running skips repos that already exist in src/.
repositories:
  src/lerobot-isaac-configs:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-configs.git
    version: main
  src/lerobot-isaac-dashboard:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-dashboard.git
    version: main
  src/lerobot-isaac-autoresearch:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-autoresearch.git
    version: main
  src/lerobot-isaac-env:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-env.git
    version: main
  src/lerobot-isaac-adapters:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-adapters.git
    version: main
  src/lerobot-isaac-synthetic:
    type: git
    url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/lerobot-isaac-synthetic.git
    version: main
  # Recorder is opt-in. Uncomment to develop it alongside:
  # src/robot-data-recorder:
  #   type: git
  #   url: ${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}/robot-data-recorder.git
  #   version: main
```

Note: vcstool's `${VAR:-default}` works because vcstool runs URLs through shell expansion;
if it does NOT (some versions only support bare `${VAR}`), the `scripts/install.sh` will
export the var with a default before calling `vcs import`.

### Updated: `pixi.toml`

Replace the entire `[pypi-dependencies]` block (lines 16-28) and add `python-vcstool` to
`[feature.dev.dependencies]`. Add `sync` task. Update `test` / `lint` / `fmt` to point at
`src/` instead of `archive/packages/`.

```toml
[pypi-dependencies]
# Editable path deps. Meta is in packages/; siblings are git clones in src/ (populated
# by `pixi run sync`). All editable — edits hot-reload, commits push to each repo's
# own remote.
lerobot-isaac-meta = { path = "packages/lerobot-isaac-meta", editable = true }
lerobot-isaac-configs = { path = "src/lerobot-isaac-configs", editable = true }
lerobot-isaac-dashboard = { path = "src/lerobot-isaac-dashboard", editable = true }
lerobot-isaac-autoresearch = { path = "src/lerobot-isaac-autoresearch", editable = true }
lerobot-isaac-env = { path = "src/lerobot-isaac-env", editable = true }
lerobot-isaac-adapters = { path = "src/lerobot-isaac-adapters", editable = true }
lerobot-isaac-synthetic = { path = "src/lerobot-isaac-synthetic", editable = true }

[feature.dev.dependencies]
pytest = "*"
ruff = "*"
hatchling = "*"
python-vcstool = "*"   # used by `pixi run sync`

# ... (other features unchanged) ...

[tasks]
sync = "bash scripts/sync_src.sh"            # clones missing repos into src/
test = "pytest packages/lerobot-isaac-meta/tests/ src/*/tests/"
lint = "ruff check packages/ src/"
fmt = "ruff format packages/ src/"
# ... (rest unchanged, but swap any `archive/packages/` paths → `src/`) ...
```

Tasks that currently reference `archive/packages/lerobot-isaac-configs/...` (e.g.
`train-and-compare`) and `archive/packages/lerobot-isaac-env/...` (e.g. `download-usd`)
must change to `src/lerobot-isaac-configs/...` and `src/lerobot-isaac-env/...`.

### New file: `scripts/sync_src.sh`

Thin wrapper that ensures `LEROBOT_SPINOUTS_BASE` is set, then runs vcstool.

```bash
#!/usr/bin/env bash
# Clone (or update) the 6 sibling repos into src/ via vcstool.
#
# Usage:
#   bash scripts/sync_src.sh                    # uses ~/workspaces/spinouts/ bare repos
#   LEROBOT_SPINOUTS_BASE=https://github.com/kvgork bash scripts/sync_src.sh
#
# Idempotent: skips repos that already exist as git clones at src/<name>/.

set -euo pipefail

export LEROBOT_SPINOUTS_BASE="${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/deps.repos"

mkdir -p "$REPO_ROOT/src"

if ! command -v vcs >/dev/null 2>&1; then
  echo "ERROR: vcstool not found on PATH. Are you inside a pixi env?" >&2
  echo "  Run: pixi shell    # then re-run this script" >&2
  exit 1
fi

# `vcs import` is idempotent — skips repos that already exist.
echo "Syncing src/ from $LEROBOT_SPINOUTS_BASE..."
envsubst < "$MANIFEST" | vcs import "$REPO_ROOT/src" --skip-existing

echo ""
echo "Sync complete. Current src/ contents:"
ls -la "$REPO_ROOT/src/"
```

Note on `envsubst`: vcstool versions vary on whether they expand `${VAR}` themselves.
Piping through `envsubst` (coreutils, always available) is the portable way. If vcstool
expands natively, `envsubst` is a no-op pass-through.

### Updated: `scripts/install.sh`

Add a pre-step that runs sync. The post-spinout standalone path (lines 33-38) stays
unchanged — that's for users WITHOUT the monorepo.

```bash
#!/usr/bin/env bash
# Install lerobot-isaac-meta + 6 dep repos.
#
# In MONOREPO mode: clones siblings into src/ via vcstool, then `pixi install` resolves
# everything as editable path deps. Run `pixi run sync` directly if you only want the
# git clone step.
#
# In POST-SPINOUT standalone mode (no monorepo): falls back to git+file:// install of
# meta with [post-spinout] extra.
#
# Usage:
#   bash scripts/install.sh                                                 # monorepo
#   LEROBOT_SPINOUTS_BASE=https://github.com/kvgork bash scripts/install.sh # GitHub
#   STANDALONE=1 bash scripts/install.sh                                    # no monorepo

set -euo pipefail

export LEROBOT_SPINOUTS_BASE="${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${STANDALONE:-0}" == "1" ]]; then
  # Old path. Pip-only, no monorepo.
  pip install "git+file://${LEROBOT_SPINOUTS_BASE#file://}/lerobot-isaac-meta.git@main[post-spinout]"
  exit 0
fi

# Monorepo path.
echo "[1/2] Syncing src/ from $LEROBOT_SPINOUTS_BASE..."
bash "$REPO_ROOT/scripts/sync_src.sh"

echo "[2/2] Running pixi install..."
cd "$REPO_ROOT"
pixi install

echo ""
echo "Done. Workspace layout:"
echo "  packages/lerobot-isaac-meta/     (editable)"
echo "  src/lerobot-isaac-*/             (cloned + editable, 6 repos)"
echo ""
echo "Recorder is standalone — to develop it alongside, run:"
echo "  git clone $LEROBOT_SPINOUTS_BASE/robot-data-recorder.git src/robot-data-recorder"
```

### Updated: `.gitignore`

Append:
```
# Cloned sibling repos (each is its own git repo with its own remote).
# Monorepo only tracks packages/lerobot-isaac-meta/ + workspace config.
src/
```

### Updated: `packages/lerobot-isaac-meta/pyproject.toml`

**No change required.** The `[project.optional-dependencies].post-spinout` block is the
production install path for standalone usage (`pip install meta[post-spinout]`). It coexists
fine with the pixi editable layout.

### Updated: `docs/runbook/00-install.md`

Add Section "Step 2.5: Workspace dev mode (clone-to-src)" between current Step 2 and Step 3.
Refer users to `scripts/sync_src.sh` and `pixi run sync`.

---

## First-Time Setup Sequence

```bash
cd ~/workspaces/lerobot-isaac-training

# Option A: one-shot
bash scripts/install.sh

# Option B: step-by-step
pixi install                  # installs vcstool into the default env
pixi run sync                 # clones 6 repos into src/
pixi install                  # re-resolves now that src/<name>/ exists

# Verify all 7 packages are editable
pixi run python -c "import lerobot_isaac_meta, lerobot_isaac_configs, lerobot_isaac_dashboard; print('OK')"
pixi run pytest packages/lerobot-isaac-meta/tests/ src/*/tests/
```

## Re-Sync (after pulling latest from a sibling remote)

```bash
# Pull latest in each src/ repo (vcs pull does git pull --ff-only per repo)
pixi run -- vcs pull src/

# Or per-repo manually:
cd src/lerobot-isaac-env && git pull && cd -

# No need to re-run pixi install — editable installs pick up the new code immediately.
```

## Switching to GitHub URLs (when published)

```bash
export LEROBOT_SPINOUTS_BASE=https://github.com/kvgork
# Then either re-run sync (if src/ is empty) or update each repo's remote:
cd src/lerobot-isaac-env
git remote set-url origin https://github.com/kvgork/lerobot-isaac-env.git
git pull
```

---

## Migration Path (from current `git+file://` non-editable setup)

Users who already ran `pixi install` under the current `pixi.toml` have the 6 sibling
packages installed non-editable inside `.pixi/envs/default/lib/python*/site-packages/`.

```bash
# 1. Pull latest monorepo (gets new pixi.toml + scripts + deps.repos)
git pull

# 2. Nuke stale pixi env and lockfile (siblings are about to switch from git+file:// → path)
rm -rf .pixi/envs/ pixi.lock

# 3. Fresh install — this clones siblings into src/ and installs everything editable
bash scripts/install.sh

# 4. Verify editable
pip show lerobot-isaac-env | grep -i editable      # expect: Editable project location: .../src/lerobot-isaac-env

# 5. Sanity test
pixi run pytest packages/lerobot-isaac-meta/tests/ src/*/tests/
```

---

## Verification Checklist

After implementation:

- [ ] `src/` exists and contains 6 git clones (run `ls -la src/`; each subdir has `.git/`)
- [ ] `pip show lerobot-isaac-env` reports `Editable project location: .../src/lerobot-isaac-env`
- [ ] Editing `src/lerobot-isaac-env/src/lerobot_isaac_env/__init__.py` and re-importing
      inside `pixi shell` reflects the change without reinstall (hot reload)
- [ ] `cd src/lerobot-isaac-env && git remote -v` shows the bare repo (or GitHub) URL
- [ ] `cd src/lerobot-isaac-env && git status` works — each is its own repo
- [ ] Monorepo `git status` does NOT list `src/lerobot-isaac-*` (ignored)
- [ ] `pixi run test` passes against `packages/lerobot-isaac-meta/tests/ src/*/tests/`
- [ ] `pixi run lint` and `pixi run fmt` cover `packages/` and `src/`

---

## Open Questions for User (3)

**Q1.** Tooling choice: **vcstool** (recommended), bash loop in `scripts/install.sh`,
or git submodules?

**Q2.** Include `robot-data-recorder` in `src/` by default (un-comment its block in
`deps.repos`)? **Recommended: NO** — recorder is standalone hardware-tier, separate concern.

**Q3.** Delete `archive/packages/` immediately as part of this phase, or keep one more
session for safety? **Recommended: KEEP one more session.** Tasks in `pixi.toml`
currently reference `archive/packages/...`; we'll migrate them to `src/...` in this phase,
then nuke `archive/` in the next.

---

## What This Plan Does NOT Do

- Does not modify any sibling repo (the 6 bare repos at `~/workspaces/spinouts/` are read-only here).
- Does not publish to GitHub. URL flip is a separate phase (see `docs/runbook/09-publish-to-github.md`).
- Does not delete `archive/packages/`. Deferred per Q3.
- Does not change `packages/lerobot-isaac-meta/pyproject.toml`'s `[post-spinout]` extra — standalone install still works.
