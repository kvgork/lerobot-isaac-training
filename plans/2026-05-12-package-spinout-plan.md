# Package Spinout Plan — 2026-05-12

**Goal:** Stage every package under `packages/` as an independent, spin-out-ready repo
*inside the monorepo only* (no remotes, no pushes). Generate `spinout/<pkg>` branches
via `git subtree split`. Keep the monorepo authoritative; document bi-directional
`git subtree push/pull` for future round-trip syncs.

**User decisions locked in (this turn):**
- Q1 Remote: stage in monorepo only; remotes decided in a later session.
- Q2 Monorepo fate: keep monorepo authoritative + plan bi-directional sync.
  Do NOT replace `packages/<pkg>/` with stubs. Do NOT delete `packages/`.
- Q3 Naming: drop the `lerobot-isaac-` prefix where it adds no identity. Mapping below.
- Q4+Q5 Pre-split: commit recorder rename + audit sizes first; tag pre-spinout anchor.

**Out of scope (defer to follow-up sessions):**
- Pushing to GitHub remotes.
- Replacing `packages/<pkg>/` with README stubs.
- Spinning out the recorder to a path *outside* `packages/` (it remains at `packages/lerobot-isaac-recorder/` even though its project name is now `robot-data-recorder`).

---

## §0 Audit Findings (read-only, no commits made yet)

### Per-package tracked size

| Package | Tracked size | Files | Largest file |
|---------|-------------:|------:|--------------|
| lerobot-isaac-adapters    | 0.13 MB | 25 | `pixi.lock` (19 KB) |
| lerobot-isaac-autoresearch| 0.06 MB | 18 | `tests/test_programs_parse.py` (8 KB) |
| lerobot-isaac-configs     | 0.07 MB | 22 | `pixi.lock` (19 KB) |
| lerobot-isaac-dashboard   | 0.39 MB | 71 | `src/.../compare.py` (23 KB) |
| lerobot-isaac-env         | 0.12 MB | 28 | `src/.../so101_env_cfg.py` (15 KB) |
| lerobot-isaac-meta        | 0.09 MB | 20 | `src/.../batch.py` (10 KB) |
| lerobot-isaac-recorder    | 0.07 MB | 15 | `README.md` (10 KB) |
| lerobot-isaac-synthetic   | 0.14 MB | 24 | `pixi.lock` (25 KB) |

**Total tracked across packages: ~1.07 MB. No package is bloated.**

### Suspicious-binary scan (regex over `.pt|.ckpt|.h5|.hdf5|.parquet|.onnx|.npz|.npy|.usd*|.png|.jpg|.jpeg|.mp4|.avi|.mkv|.webm|.tar|.zip|.gz|.bz2|.pkl|.pickle|.safetensors`)

**Result: ZERO matches across all 8 packages.** No tracked binaries / artifacts.

### Gitignored-dir leak check

- `datasets/` 8 KB, `outputs/` 4.7 MB, `.agent-state/` 28 KB — all on disk
- `git ls-files datasets/ outputs/` → empty. Nothing leaked into git.
- `.gitignore` already covers `datasets/ outputs/ .agent-state/ __pycache__/`.

### Suspicious untracked file inside recorder old dir

`packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/so101_teleop.py` is
**untracked** (git status shows `__init__.py … schema.py` all `D` deleted, but
`so101_teleop.py` was never tracked under the old path — it lives under the new
`robot_data_recorder/` dir already and the old file is a stale leftover).
Also: `packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/__pycache__/`
contains compiled `.pyc` files (untracked, gitignored — harmless, will get cleaned
when the old src dir is removed in §2).

**Audit verdict:** No cleanup commits needed before split. Steps §1-c (artifact
cleanup) drops out — proceed straight from recorder rename commit (§2) to tag (§3)
to subtree split (§4).

---

## §1 Naming Mapping Table — REQUIRES USER CONFIRMATION

This is the proposal. **Do not execute §4-onwards until the user signs off on this
table** (a single message saying "approve table" or specifying edits is sufficient).

| Current dir under `packages/` | Proposed standalone repo name | Python module (`src/<name>/`) | Rename rationale |
|------------------------------|------------------------------|-------------------------------|------------------|
| `lerobot-isaac-meta`         | `lerobot-isaac-meta`         | `lerobot_isaac_meta`          | Meta-package — prefix IS the identity. Keep. |
| `lerobot-isaac-env`          | `isaac-so101-env`            | `isaac_so101_env`             | It is the Isaac Lab SO-101 env; "lerobot" is misleading (env is policy-agnostic). |
| `lerobot-isaac-adapters`     | `lerobot-training-adapters`  | `lerobot_training_adapters`   | Training-arch dispatcher; "isaac" is misleading (works for any backend). |
| `lerobot-isaac-autoresearch` | `autoresearch-ml-loop`       | `autoresearch_ml_loop`        | General-purpose ML autoresearch wrapper; not lerobot- or isaac-specific. |
| `lerobot-isaac-synthetic`    | `isaac-dr-synthetic`         | `isaac_dr_synthetic`          | Isaac DR replay data gen; the "lerobot" half is consumer, not producer. |
| `lerobot-isaac-configs`      | `lerobot-isaac-configs`      | `lerobot_isaac_configs`       | Shared YAML config registry — used by recorder + adapters + env. Keep. |
| `lerobot-isaac-recorder`     | `robot-data-recorder`        | `robot_data_recorder`         | Already renamed in `pyproject.toml`. Hardware-side, not Isaac. |
| `lerobot-isaac-dashboard`    | `lerobot-metrics-dashboard`  | `lerobot_metrics_dashboard`   | Streamlit dashboard. Drop "isaac" — works on any LeRobot run dir. |

**Renames in scope right now (already in-flight or accepted):**
- `lerobot-isaac-recorder` → `robot-data-recorder` (already partially done; finish + commit in §2)

**Renames deferred (decided here but executed in a future session):**
- `lerobot-isaac-env` → `isaac-so101-env`
- `lerobot-isaac-adapters` → `lerobot-training-adapters`
- `lerobot-isaac-autoresearch` → `autoresearch-ml-loop`
- `lerobot-isaac-synthetic` → `isaac-dr-synthetic`
- `lerobot-isaac-dashboard` → `lerobot-metrics-dashboard`

Reason for deferral: each non-recorder rename requires sweeping changes across
imports, configs, tests, and docs. Doing all 5 in one shot risks merge conflicts
with the ongoing recorder rename work. Plan handles them in §6 as **per-package
follow-up tasks**, each gated independently.

For §4 subtree split we use the **current directory names** as branch names
(`spinout/lerobot-isaac-env` etc.). The standalone-repo *project name* mismatch
inside `pyproject.toml` is fine — pip resolves by `[project].name`, not dir name.

---

## §2 Finish + Commit the Recorder Rename — REQUIRES USER CONFIRMATION

Current state (verified):
- New src dir `packages/lerobot-isaac-recorder/src/robot_data_recorder/` exists (10 files).
- Old src dir `packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/` still
  contains untracked `so101_teleop.py` + `__pycache__/` (will be removed).
- `pyproject.toml` already declares `name = "robot-data-recorder"` and
  `packages = ["src/robot_data_recorder"]`.
- `pixi.toml` already declares `name = "robot-data-recorder"` and
  `robot-data-recorder = { path = ".", editable = true }`.
- 4 new test files untracked: `test_check_hardware.py`, `test_config_env.py`,
  `test_keylistener.py`, `test_recorder_keys.py`.
- Root `pyproject.toml`, `pixi.toml`, `README.md`, `CHANGELOG.md`, and
  `packages/lerobot-isaac-meta/{pyproject.toml, src/.../cli.py, tests/test_cli_record.py}`
  still reference the OLD module name `lerobot_isaac_recorder` and dir name
  `lerobot-isaac-recorder`.

**Decision needed from user before §2 executes:** Do we update the meta-package
+ workspace root to point at the new `robot_data_recorder` module *as part of this
rename commit*, or leave that for the deferred rename batch?

**Recommendation:** Update everything in one atomic commit. The current state is
broken (meta CLI imports `lerobot_isaac_recorder.cli` which no longer exists in
the new src dir). Leaving it for "later" means CI stays red.

**Caveat for user (do NOT skip reading):** the *directory* `packages/lerobot-isaac-recorder/`
is kept as-is (rename only the Python package name + standalone project name).
Renaming the directory now would require updating every cross-reference twice
(once here, once again when we batch the other 5 renames in §6). Defer the dir
rename to §6.

### §2.a — Clean stale old src dir (non-destructive: only untracked files)

```bash
# Verify no tracked files remain in old src dir before removing
git ls-files packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/ | wc -l
# Expected output: 0  (all 7 originally-tracked .py files already shown as 'D' deleted)

# requires user confirmation
rm -rf packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/
```

### §2.b — Fix meta-package + workspace root references to the new module name

Files to edit (exhaustive list from grep):

1. `packages/lerobot-isaac-meta/src/lerobot_isaac_meta/cli.py`
   lines 38, 40, 43, 44, 45, 154:
   - `from lerobot_isaac_recorder.cli import main` → `from robot_data_recorder.cli import main`
   - error string `lerobot_isaac_recorder` → `robot_data_recorder`
   - help-string `lerobot-isaac-recorder` → `robot-data-recorder`
   - install hint `packages/lerobot-isaac-recorder` stays (dir unchanged)
2. `packages/lerobot-isaac-meta/tests/test_cli_record.py`
   lines 13, 15, 68, 70, 75, 81, 94: rename `lerobot_isaac_recorder` →
   `robot_data_recorder` in sys.path injection, fake-finder name match, and
   error-string assertions. Leave the directory path
   `parents[3] / "lerobot-isaac-recorder" / "src"` unchanged (dir name unchanged).
3. `packages/lerobot-isaac-meta/pyproject.toml` line 23:
   `"lerobot-isaac-recorder"` → `"robot-data-recorder"` (PyPI dep name in `monorepo` extra).
4. Root `pyproject.toml` line 20: `lerobot-isaac-recorder = { workspace = true }`
   → `robot-data-recorder = { workspace = true }`.
5. Root `pixi.toml` line 24:
   `lerobot-isaac-recorder = { path = "packages/lerobot-isaac-recorder", editable = true }`
   → `robot-data-recorder = { path = "packages/lerobot-isaac-recorder", editable = true }`.
6. Root `README.md` lines 32, 111: rename `lerobot-isaac-recorder` →
   `robot-data-recorder` in the package-map table and ASCII diagram (path stays
   `packages/lerobot-isaac-recorder/` since dir unchanged).
7. Root `CHANGELOG.md` line 50: rename `lerobot-isaac-recorder` → `robot-data-recorder`.
8. Workspace-root `CLAUDE.md`: scan for any remaining `lerobot-isaac-recorder`
   strings and update — search command provided below.

```bash
# Re-run grep right before commit to catch anything missed:
grep -rn "lerobot_isaac_recorder\|lerobot-isaac-recorder" \
  --include="*.py" --include="*.toml" --include="*.md" \
  --include="*.yaml" --include="*.yml" --include="*.sh" \
  . 2>/dev/null
# Expected after edits: zero matches outside .git/ and historical CHANGELOG entries.
```

### §2.c — Re-run recorder tests to verify the new src layout works standalone

```bash
cd /home/koen/workspaces/lerobot-isaac-training/packages/lerobot-isaac-recorder
python3 -m pytest tests/ -q
# Expected: all 54 tests pass (per CLAUDE.md status line).
```

### §2.d — Verify pixi workspace + lockfile still resolves

```bash
cd /home/koen/workspaces/lerobot-isaac-training
pixi install --dry-run
# Expected: clean resolve, no missing-package errors after pixi.toml edit.
# If lockfile changed, this is the moment to regenerate it.
```

### §2.e — Stage + commit the rename atomically

```bash
cd /home/koen/workspaces/lerobot-isaac-training

# Stage everything related to the recorder rename + the cross-package fixups.
git add \
  packages/lerobot-isaac-recorder/ \
  packages/lerobot-isaac-meta/src/lerobot_isaac_meta/cli.py \
  packages/lerobot-isaac-meta/tests/test_cli_record.py \
  packages/lerobot-isaac-meta/pyproject.toml \
  pyproject.toml \
  pixi.toml \
  README.md \
  CHANGELOG.md \
  CLAUDE.md   # only if §2.b step 8 found edits

# Sanity-check staging set before commit
git status --short

# requires user confirmation
git commit -m "$(cat <<'EOF'
refactor(recorder): rename package to robot-data-recorder

Python module: lerobot_isaac_recorder -> robot_data_recorder
Project name : lerobot-isaac-recorder -> robot-data-recorder

Directory packages/lerobot-isaac-recorder/ is unchanged (dir-name rename
deferred to package-spinout batch — see plans/2026-05-12-package-spinout-plan.md §6).

Cross-package fixups:
- lerobot-isaac-meta CLI delegation + tests retargeted to robot_data_recorder.cli
- workspace root pyproject.toml + pixi.toml + README + CHANGELOG updated

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## §3 Tag Recovery Anchor — REQUIRES USER CONFIRMATION

```bash
cd /home/koen/workspaces/lerobot-isaac-training

# Sanity: working tree must be clean post-§2 commit.
git status --short
# Expected: empty.

# requires user confirmation
git tag -a pre-spinout-2026-05-12 -m "Recovery anchor before package subtree splits"

# Verify
git tag -l 'pre-spinout-*' --format='%(refname:short)  %(creatordate:short)  %(subject)'
```

If anything goes wrong in §4-onwards, reset is:
`git reset --hard pre-spinout-2026-05-12` (requires user confirmation when used).

---

## §4 Subtree Split — One Branch Per Package — REQUIRES USER CONFIRMATION

For each of the 8 packages, generate a `spinout/<pkg>` branch containing only
that package's history flattened to repo root.

**Pre-flight:**
```bash
cd /home/koen/workspaces/lerobot-isaac-training

# Tree must be clean.
git status --short  # expect empty
# On main (or whatever the spinout source branch should be):
git branch --show-current  # expect: main
```

### §4.a — Pilot split (lerobot-isaac-configs, smallest, lowest blast radius)

```bash
cd /home/koen/workspaces/lerobot-isaac-training

PKG=lerobot-isaac-configs

# Delete any stale spinout branch first (no-op if absent).
git branch -D "spinout/$PKG" 2>/dev/null || true

# requires user confirmation
git subtree split --prefix="packages/$PKG" -b "spinout/$PKG"

# Verify branch exists and points at a commit
git log -1 --oneline "spinout/$PKG"

# Quick structural check on the branch (does NOT check out)
git ls-tree --name-only "spinout/$PKG"
# Expected at branch root: pyproject.toml, pixi.toml, README.md, src/, tests/, docs/, ...
# NO 'packages/' dir.
```

### §4.b — Run the spinout smoke test against the pilot

⚠ **The smoke test currently has hardcoded assumptions that will break for any
package whose `src/<name>/` does not equal `${PKG//-/_}`.** See §5 for the patch.
For the §4.a pilot (`lerobot-isaac-configs`) the convention holds, so the
unmodified script works.

```bash
cd /home/koen/workspaces/lerobot-isaac-training
bash scripts/spinout_smoke_test.sh lerobot-isaac-configs
# Expected: "PASS: lerobot-isaac-configs spinout smoke test"
```

### §4.c — Split the remaining 7 packages

**Stop and ask user before proceeding if §4.b printed FAIL.**

```bash
cd /home/koen/workspaces/lerobot-isaac-training

for PKG in lerobot-isaac-meta \
           lerobot-isaac-env \
           lerobot-isaac-adapters \
           lerobot-isaac-autoresearch \
           lerobot-isaac-synthetic \
           lerobot-isaac-recorder \
           lerobot-isaac-dashboard; do
  echo "==== splitting $PKG ===="
  git branch -D "spinout/$PKG" 2>/dev/null || true
  # requires user confirmation
  git subtree split --prefix="packages/$PKG" -b "spinout/$PKG"
  git log -1 --oneline "spinout/$PKG"
done

# List all spinout branches
git branch --list 'spinout/*'
```

### §4.d — Smoke-test each split branch

**`lerobot-isaac-recorder` will FAIL the unmodified smoke test** because its
`src/` dir is `robot_data_recorder/`, not `lerobot_isaac_recorder/`. This is the
exact case §5 fixes. Run §5 BEFORE this loop, or expect 1 failure (recorder)
and address it via §5.

```bash
cd /home/koen/workspaces/lerobot-isaac-training
for PKG in lerobot-isaac-meta \
           lerobot-isaac-env \
           lerobot-isaac-adapters \
           lerobot-isaac-autoresearch \
           lerobot-isaac-synthetic \
           lerobot-isaac-recorder \
           lerobot-isaac-dashboard; do
  echo "==== smoke $PKG ===="
  bash scripts/spinout_smoke_test.sh "$PKG" || echo "FAIL: $PKG"
done
```

---

## §5 Patch `scripts/spinout_smoke_test.sh` — REQUIRES USER CONFIRMATION

**Problem:** Line 22 hardcodes `[ -d "src/${PKG//-/_}" ]`. This fails for:
- `lerobot-isaac-recorder` (project name now `robot-data-recorder`, src dir `robot_data_recorder`)
- All future dir-renames in §6

**Fix:** Read the actual src package name from `pyproject.toml` instead of
inferring it from the dir name. Two viable approaches:

**Option A (preferred, robust):** Parse `[tool.hatch.build.targets.wheel] packages = ["src/<name>"]`
out of `pyproject.toml`. Falls back to `PKG//-/_` if not found.

**Option B (simpler, less robust):** Add a `--src-dir <name>` CLI flag to the
script. Caller must remember the override per package.

Going with **Option A**. Apply this diff via `Edit`:

```bash
# Current line 22:
[ -d "src/${PKG//-/_}" ] || { echo "FAIL: no src/$(echo $PKG | tr - _)"; exit 1; }
```

Replace with:

```bash
# Determine src package dir name from pyproject.toml [tool.hatch.build.targets.wheel].packages,
# falling back to PKG//-/_ if not declared.
SRC_PKG=$(python3 -c "
import sys, tomllib
with open('pyproject.toml','rb') as f:
    d = tomllib.load(f)
pkgs = d.get('tool',{}).get('hatch',{}).get('build',{}).get('targets',{}).get('wheel',{}).get('packages')
if pkgs and pkgs[0].startswith('src/'):
    print(pkgs[0][4:])
else:
    print('$PKG'.replace('-','_'))
" 2>/dev/null || echo "${PKG//-/_}")

[ -d "src/$SRC_PKG" ] || { echo "FAIL: no src/$SRC_PKG (pyproject-declared or fallback)"; exit 1; }
```

Also adjust the success line for clarity:

```bash
echo "PASS: $PKG spinout smoke test (src/$SRC_PKG)"
```

After patching, re-run §4.d. Recorder should now pass.

```bash
cd /home/koen/workspaces/lerobot-isaac-training
# requires user confirmation
git add scripts/spinout_smoke_test.sh
git commit -m "$(cat <<'EOF'
fix(spinout): read src package name from pyproject instead of guessing from PKG

Hardcoded `src/${PKG//-/_}` broke whenever the project name diverged from the
directory name (e.g. robot-data-recorder lives in packages/lerobot-isaac-recorder/).
Now parses [tool.hatch.build.targets.wheel].packages and falls back to the old
convention. Also reports the resolved src dir in the PASS line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Ordering note:** §5 should ideally run *before* §4.c so the smoke loop in §4.d
just works. Updated execution order at bottom reflects this.

---

## §6 Deferred Per-Package Renames (Future Sessions)

Each rename below is its own follow-up task. **None executed this session.**
Listed here so we have a written record.

Pattern (apply once per package):
1. Rename `packages/<old-name>/` → `packages/<new-name>/`.
2. Rename `src/<old_module>/` → `src/<new_module>/`.
3. Update `pyproject.toml` `[project].name` + `[tool.hatch.build.targets.wheel].packages`.
4. Update `pixi.toml` `[workspace].name` + `[pypi-dependencies].<name>`.
5. Update all imports (`grep -rn "import <old_module>\|from <old_module>"`).
6. Update workspace root `pyproject.toml`, `pixi.toml`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`.
7. Update any package `CLAUDE.md` cross-references.
8. Re-run that package's tests + the workspace integration tests.
9. Re-run subtree split: `git subtree split --prefix=packages/<new-name> -b spinout/<new-name>`.
10. Re-run smoke test.

Renames pending (per §1 table):
- [ ] `lerobot-isaac-env` → `isaac-so101-env`
- [ ] `lerobot-isaac-adapters` → `lerobot-training-adapters`
- [ ] `lerobot-isaac-autoresearch` → `autoresearch-ml-loop`
- [ ] `lerobot-isaac-synthetic` → `isaac-dr-synthetic`
- [ ] `lerobot-isaac-dashboard` → `lerobot-metrics-dashboard`

---

## §7 Bi-Directional Sync Documentation (No Execution This Turn)

Add a runbook documenting the `git subtree push/pull` workflow once remotes
exist. Skeleton goes in `docs/runbook/09-package-spinout-sync.md` — to be
written *after* the first remote is created in a later session.

Workflow sketch:
- Monorepo → spinout repo: `git subtree push --prefix=packages/<pkg> <remote> main`
- Spinout repo → monorepo: `git subtree pull --prefix=packages/<pkg> <remote> main --squash`
- Conventions: monorepo stays canonical; remote PRs are pulled in monthly via squash-pull.

---

## Execution Order Summary (after user approves §1 mapping table)

1. **§2.a** — `rm -rf packages/lerobot-isaac-recorder/src/lerobot_isaac_recorder/`
2. **§2.b** — Apply 8 sets of file edits to fix cross-package + workspace root refs.
3. **§2.c** — `pytest packages/lerobot-isaac-recorder/tests/ -q` (expect 54 pass).
4. **§2.d** — `pixi install --dry-run`.
5. **§2.e** — `git add` + `git commit` recorder rename.
6. **§5** — Patch + commit `scripts/spinout_smoke_test.sh`.
7. **§3** — `git tag -a pre-spinout-2026-05-12 …`.
8. **§4.a** — Pilot subtree split of `lerobot-isaac-configs`.
9. **§4.b** — Smoke-test the pilot. STOP if FAIL.
10. **§4.c** — Loop subtree split for the other 7 packages.
11. **§4.d** — Smoke-test all 7. Recorder requires §5 patch (already applied at step 6).
12. **§6** — Deferred. Not executed.
13. **§7** — Deferred. Not executed.

All destructive operations marked `# requires user confirmation` inline. No
`git push`, no `git tag -d`, no `git reset --hard` in this plan.
