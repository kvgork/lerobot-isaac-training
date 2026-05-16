# Thin-Meta-Repo Spinout Plan — 2026-05-13

**Author session:** orchestrator
**Date:** 2026-05-13
**GitHub user:** `kvgork`
**Supersedes:** the "keep monorepo authoritative + bi-directional sync" decision
in `plans/2026-05-12-package-spinout-plan.md` §0 / Q2. **This plan REVERSES that
decision.** The monorepo becomes a thin meta-repo; the other 7 packages move to
standalone GitHub repos under `kvgork/`.

---

## Requirements Recap (verbatim from user)

- **R1.** Workspace becomes a thin meta-repo: only `packages/lerobot-isaac-meta/`
  stays in source. The other 7 packages get pulled in by an install script.
- **R2.** All 7 dep packages must also be installable as plain Python deps
  (without source checkout in this workspace) — i.e. each spinout is a standalone
  PyPI-installable or git-URL-installable Python package.
- **R3.** `lerobot-isaac-recorder` is NOT a dependency of meta. It is standalone —
  its own repo that runs hardware data-gathering independently. Meta does NOT
  pull it.

So the dependency graph after spinout is:

```
lerobot-isaac-meta   (THIS workspace)
        │ depends on (6 standalone repos)
        ├── lerobot-isaac-env-or-renamed
        ├── lerobot-isaac-adapters-or-renamed
        ├── lerobot-isaac-autoresearch-or-renamed
        ├── lerobot-isaac-synthetic-or-renamed
        ├── lerobot-isaac-configs
        └── lerobot-isaac-dashboard-or-renamed

lerobot-isaac-recorder-or-renamed   (SEPARATE; not a meta dep; runs on hardware host)
```

7 total standalone repos under `kvgork/` (6 meta-deps + 1 standalone recorder).
The meta package itself stays in THIS workspace's `packages/lerobot-isaac-meta/`.

---

## §0 Pre-flight Snapshot

| Item | State (verified 2026-05-13) |
|------|------------------------------|
| Branch | `main` |
| Tip | `f34caf5` — meta workspace-discovery decoupling |
| Recovery tag | `pre-spinout-2026-05-12` → `5db70ca` |
| Spinout branches | All 8 present (`spinout/lerobot-isaac-{configs,meta,dashboard,autoresearch,env,adapters,synthetic,recorder}`); each smoke-passed at its tip |
| Recorder rename | DONE in source (`robot-data-recorder` / `robot_data_recorder`); dir name unchanged (`packages/lerobot-isaac-recorder/`) |
| Meta deps (pyproject `[monorepo]` extra) | `lerobot-isaac-{env,adapters,autoresearch,synthetic,configs}` + `robot-data-recorder`. **Note: recorder is currently listed in this extra — R3 says it must be removed.** |
| Untracked plan/audit files | `plans/2026-05-12-package-spinout-plan.md`, `plans/2026-05-13-thin-meta-repo-plan.md` (this file), `audits/2026-05-13-sb-vs-workspace.md`, `docs/internals/system-improvements.md`, `outputs/pusht_diffusion_smoke/`, `datasets/pusht/` |

No destructive op has been performed yet for this plan. The 8 spinout branches
existing locally are inert until pushed.

---

## §1 Open Decisions — REQUIRES USER ANSWER

The 4 questions below MUST be answered before Phase B starts. Each shows the
default the orchestrator will use if the user says "go with the defaults", but
the orchestrator will NOT execute until the user responds explicitly.

### Q1 — Repo naming under `kvgork/<name>`

Two options:

**Q1.a — Keep the rename mapping from the 2026-05-12 plan (DEFAULT-A):**

| Current dir | Standalone repo name under `kvgork/` |
|-------------|--------------------------------------|
| `lerobot-isaac-env`          | `kvgork/isaac-so101-env` |
| `lerobot-isaac-adapters`     | `kvgork/lerobot-training-adapters` |
| `lerobot-isaac-autoresearch` | `kvgork/autoresearch-ml-loop` |
| `lerobot-isaac-synthetic`    | `kvgork/isaac-dr-synthetic` |
| `lerobot-isaac-configs`      | `kvgork/lerobot-isaac-configs` |
| `lerobot-isaac-recorder`     | `kvgork/robot-data-recorder` |
| `lerobot-isaac-dashboard`    | `kvgork/lerobot-metrics-dashboard` |

Pros: each name reflects what the package actually does; better for
external discoverability outside this workspace; recorder + autoresearch read
clean as general-purpose tools.

Cons: 5 of 7 require code+config+import renames before split (high churn);
breaks every existing import path; the `[monorepo]` extra in meta + workspace
root `pyproject.toml`/`pixi.toml` all need rewrites.

**Q1.b — Keep the `lerobot-isaac-*` prefix everywhere (DEFAULT-B):**

| Current dir | Standalone repo name under `kvgork/` |
|-------------|--------------------------------------|
| `lerobot-isaac-env`          | `kvgork/lerobot-isaac-env` |
| `lerobot-isaac-adapters`     | `kvgork/lerobot-isaac-adapters` |
| `lerobot-isaac-autoresearch` | `kvgork/lerobot-isaac-autoresearch` |
| `lerobot-isaac-synthetic`    | `kvgork/lerobot-isaac-synthetic` |
| `lerobot-isaac-configs`      | `kvgork/lerobot-isaac-configs` |
| `lerobot-isaac-recorder`     | `kvgork/lerobot-isaac-recorder` *(but Python project name stays `robot-data-recorder` — already committed)* |
| `lerobot-isaac-dashboard`    | `kvgork/lerobot-isaac-dashboard` |

Pros: zero rename churn in code; GitHub repo namespace acts as a discoverable
"lerobot-isaac" suite; spinout branches push as-is; commit history clean; reverse
later if needed (rename via GitHub UI).

Cons: 4 of the names slightly mislead (env / adapters / autoresearch /
synthetic are not lerobot-bound or isaac-bound); harder to position individual
packages as general-purpose tools later.

**Hybrid option Q1.c:** keep `lerobot-isaac-*` for repos that are genuinely
LeRobot+Isaac-coupled (`env`, `synthetic`, `configs`, `dashboard`); rename the
two truly-decoupled ones (`adapters` → `lerobot-training-adapters`,
`autoresearch` → `autoresearch-ml-loop`); recorder already renamed to
`robot-data-recorder` so repo follows the project name. 4 keep, 3 rename.

**Recommendation:** Q1.b (keep prefix). The cost of renames is high (file edits,
import sweeps, broken cross-refs in 8+ docs), and "for-publish discoverability"
is the use case Q1.b is actually best for — a coherent suite under one prefix is
more discoverable than 7 independently-named packages. Renaming individual
packages later via `gh repo rename` is one command + a redirect; reverse-renaming
is also cheap. Start cheap, rename later if it matters.

> **Q1 — Answer needed:** A (rename per table), B (keep `lerobot-isaac-*`), or
> C (hybrid: rename only adapters + autoresearch)? Default if you say "go": **B**.

### Q2 — Dependency mechanism: meta → 6 standalone repos

How does the meta package, post-spinout, fetch the 6 dep packages it now depends
on (excluding recorder per R3)?

**Q2.a — Git URLs in `pyproject.toml`:**

```toml
[project]
dependencies = [
  "pyyaml>=6.0",
  "lerobot-isaac-env @ git+https://github.com/kvgork/lerobot-isaac-env.git@main",
  "lerobot-isaac-adapters @ git+https://github.com/kvgork/lerobot-isaac-adapters.git@main",
  # ... 4 more
]
```

Pros: zero PyPI infra needed; works for private repos via `git+ssh://`; pinnable
to commit SHA (`@<sha>`) for reproducibility; no version bumps.

Cons: not installable via plain `pip install lerobot-isaac-meta` from PyPI
without `--index-url` tricks (R2 is partially satisfied: each repo is
git-installable but not PyPI-installable until you publish).

**Q2.b — PyPI-publish all 7, then meta declares plain version specifiers:**

```toml
[project]
dependencies = [
  "pyyaml>=6.0",
  "lerobot-isaac-env>=0.1.0",
  "lerobot-isaac-adapters>=0.1.0",
  # ... 4 more
]
```

Pros: cleanest UX (`pip install lerobot-isaac-meta` works); R2 fully satisfied;
discoverable on PyPI.

Cons: requires PyPI account (you have one for `kvgork`?); requires release-tag
+ version-bump discipline per package; need CI hooks for publish.

**Q2.c — pixi `[pypi-dependencies]` with git sources (dev) + plain (end-user):**

```toml
# pixi.toml for meta package (dev mode):
[pypi-dependencies]
lerobot-isaac-env = { git = "https://github.com/kvgork/lerobot-isaac-env.git", branch = "main" }
# ...

# pyproject.toml [project].dependencies stays plain "lerobot-isaac-env>=0.1.0"
# for pip/PyPI consumers.
```

Pros: pixi devs get auto-updating git deps; pip users still install via PyPI
once published; covers both UX paths.

Cons: dual configuration (pixi.toml + pyproject.toml diverge); requires
publishing to PyPI eventually anyway for `pip` users.

**Recommendation:** Q2.a as the first cut. PyPI publishing is a separate
project; getting the meta repo working with git URLs unblocks today. Migrate to
Q2.b in a later session once PyPI release process is set up. Pinning is via
branch name (`@main`); switch to commit SHA pins (`@<sha>`) for reproducibility
before any real release.

> **Q2 — Answer needed:** A (git URLs), B (PyPI), C (pixi dev + pip prod
> hybrid). Default if you say "go": **A**.

### Q3 — Install-script form

How do users (and you, after re-cloning the meta repo) install the workspace?

**Q3.a — Shell script `scripts/install.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Install meta package + transitive 6 deps from GitHub (per Q2.a).
pip install -e packages/lerobot-isaac-meta
# Recorder lives in its own repo: install separately on the hardware host:
#   pip install git+https://github.com/kvgork/lerobot-isaac-recorder.git
echo "Done. To run a training adapter:  python -m lerobot_isaac_adapters.train --help"
```

Pros: simple; works without pixi; one entrypoint; transitive deps resolved by
pip automatically given Q2.a/b.

Cons: no isolation (pollutes system Python unless you set up a venv first);
relies on the user already having Python 3.10+ available.

**Q3.b — Pixi feature `[feature.deps]` listing all 6 as git pypi-deps:**

```toml
# pixi.toml at repo root:
[feature.deps.pypi-dependencies]
lerobot-isaac-env = { git = "https://github.com/kvgork/lerobot-isaac-env.git", branch = "main" }
# ... 5 more

[environments]
default = ["deps", "dev"]
```

Then `pixi install` does everything.

Pros: isolated env; one tool covers system deps (conda) + python deps (pip+git);
matches the existing pixi workflow this repo already uses.

Cons: requires pixi as a hard dependency for end users (pip-only users can't
install); duplicates the dep list (also lives in `meta/pyproject.toml`).

**Q3.c — Both — pixi for devs, pip-script for end-users:**

Ship both `scripts/install.sh` (pip path) and the pixi `[feature.deps]` block.
Document both in README.

Pros: maximum reach; matches each user's preferred tool.

Cons: two paths to keep in sync; doubles maintenance.

**Recommendation:** Q3.c. The existing workspace ALREADY uses pixi heavily
(7 environments, `pixi run` for everything). Removing pixi support would be a
regression for the dev workflow. But the published meta package should also be
plain-pip-installable for users who don't want pixi. Both paths sharing the same
git-URL source via Q2.a keeps drift minimal.

> **Q3 — Answer needed:** A (shell script), B (pixi feature), C (both). Default
> if you say "go": **C**.

### Q4 — Fate of `packages/<7 deps>/` directories in monorepo

After spinout, what happens to the 7 directories that no longer have a reason to
live in this workspace's source tree?

**Q4.a — `git rm -r` (DEFAULT-A; aggressive):**

Delete the 7 directories entirely. Single commit:
`chore(repo): remove 7 packages now spun out to standalone repos`.

Pros: workspace tree becomes truly thin (only `packages/lerobot-isaac-meta/`);
clarity of "this is the meta repo, deps live elsewhere"; no confusion about
which copy is authoritative.

Cons: working tree loses ~1 MB of code (history still retains it). If a future
session needs to re-inspect a package's earlier history without cloning the
spun-out repo, you have to `git log -- packages/<pkg>/` against the pre-spinout
commit — still possible but less convenient. The spinout branches preserve
post-split history.

**Q4.b — Move to `archive/packages/`:**

`git mv packages/<pkg> archive/packages/<pkg>` for each. Add an
`archive/README.md` with a deprecation pointer to the standalone repos.

Pros: code stays browsable in-place; obvious from the path that it's frozen;
no risk of someone editing it expecting changes to flow into the live package.

Cons: bloats the meta workspace tree (~1 MB you have to clone to use the meta
package); confusing — looks like the packages still live here.

**Q4.c — Leave in-place + deprecation README only:**

Add `packages/<pkg>/DEPRECATED.md` to each, pointing at the standalone repo.
Don't move or delete the source.

Pros: zero deletion risk; smoke tests / sibling tests in `[monorepo]` extra
still work locally for now.

Cons: violates R1 (workspace is not "thin"); easy for future-you to edit a
package here forgetting it's no longer authoritative; eventual rot.

**Recommendation:** Q4.a (delete). R1 explicitly says "thin meta-repo, only
meta in source". The spinout branches preserve every package's history at the
pre-split tip, and the standalone GitHub repos preserve forward history. There
is no need to keep dead code in the meta workspace.

⚠ **HARD GUARD:** Q4.a only proceeds after:
1. All 7 spinout branches pushed to GitHub `main` (A3) AND
2. `gh repo view kvgork/<name>` confirms each repo has commits AND
3. Meta tests pass with the 6 deps installed from GitHub (A4 + A5 + A8).

If any of those fail, Q4 stays at Q4.c (in-place) until resolved. This is
non-negotiable to prevent data loss if the GitHub side has issues.

> **Q4 — Answer needed:** A (delete), B (archive), C (leave in-place). Default
> if you say "go": **A**, gated by the hard guard above.

---

## §2 Phase B Execution Plan (only runs after user answers §1)

Order matters. Each step has its own commit + verification.

### A1 — Apply package renames (if Q1.a or Q1.c)

**Skip if Q1.b.** Otherwise per rename:

1. `git mv packages/<old-dir> packages/<new-dir>`
2. `git mv packages/<new-dir>/src/<old_module> packages/<new-dir>/src/<new_module>`
3. Edit `pyproject.toml`: `[project].name`, `[tool.hatch.build.targets.wheel].packages`
4. Edit `pixi.toml`: `[workspace].name`, `[pypi-dependencies].<name>`
5. Sweep imports: `grep -rln "import <old_module>\|from <old_module>"` → edit
6. Sweep cross-refs in root `pyproject.toml`, `pixi.toml`, `README.md`,
   `ARCHITECTURE.md`, `USAGE.md`, `CLAUDE.md`, `docs/runbook/*`, every
   package's `CLAUDE.md`
7. Run that package's tests + workspace integration tests
8. Commit: `refactor(<pkg>): rename to <new-name>`
9. Delete old spinout branch + re-split:
   ```
   git branch -D spinout/<old-dir>
   git subtree split --prefix=packages/<new-dir> -b spinout/<new-dir>
   ```
10. Smoke-test: `bash scripts/spinout_smoke_test.sh <new-dir>` → must PASS

Order: do the lowest-coupling rename first (probably `dashboard` or
`autoresearch`), highest-coupling last (`env`, since adapters + synthetic import
from it).

### A2 — Create 7 GitHub repos under `kvgork/`

One repo at a time (so failures don't cascade):

```bash
for NAME in <Q1-resolved-names>; do
  gh repo create "kvgork/$NAME" --public \
    --description "Component of the LeRobot + Isaac Lab SO-101 training stack."
  gh repo view "kvgork/$NAME" --json url,name,visibility
done
```

Stop and ask the user if any `gh repo create` returns "Name already exists on
this account".

### A3 — Push each spinout branch as `main` of its new repo

```bash
for PKG in <7 dirs>; do
  NAME=<resolve to repo name>
  git push "https://github.com/kvgork/$NAME.git" "spinout/$PKG:main"
  gh repo view "kvgork/$NAME" --json defaultBranchRef,pushedAt
done
```

⚠ This is the FIRST push to each new repo. No `--force`, no overwriting.
If the push is rejected because the repo already has commits, STOP and report.

### A4 — Update meta deps to reference 6 standalone repos (per Q2)

Edit `packages/lerobot-isaac-meta/pyproject.toml`:

- Move the 6 dep names out of `[project.optional-dependencies].monorepo` and
  into `[project].dependencies` (Q2.a/b form) — except `robot-data-recorder`
  which is REMOVED entirely per R3.
- Verify by grep: `grep -n "robot-data-recorder\|lerobot-isaac-recorder" packages/lerobot-isaac-meta/pyproject.toml`
  → must return zero matches.
- The `[monorepo]` extra is either dropped or repurposed to mean
  "install all 6 from local checkouts when monorepo source is present".

Edit root `pixi.toml`:
- Remove `[pypi-dependencies]` lines for the 6 deps + recorder (workspace
  members no longer exist locally after A6).
- Keep workspace-level features (`dev`, `lerobot`, etc.) since they're meta's
  features now.

Commit: `feat(meta): depend on 6 standalone repos via <Q2-mechanism>`.

Test: `cd packages/lerobot-isaac-meta && python -c "from lerobot_isaac_meta import cli"` →
still importable. Then `pixi install` (with the new deps) → must succeed.

### A5 — Create install script per Q3

If Q3.a or Q3.c: write `scripts/install.sh`. If Q3.b or Q3.c: add
`[feature.deps]` block to root `pixi.toml`.

Commit: `feat: add install script for dep packages`.

Verify: from a clean clone-like state (`pixi clean` or fresh tmpdir), running
the install path should reproduce a working env.

### A6 — Apply Q4 monorepo-fate decision

After the GUARD conditions in Q4 are met:

- Q4.a: `git rm -r packages/lerobot-isaac-{env,adapters,autoresearch,synthetic,configs,recorder,dashboard}/`.
  One commit: `chore(repo): remove 7 packages now spun out to standalone repos`.
- Q4.b: `git mv packages/<7> archive/packages/`. Add deprecation README. One
  commit: `chore(repo): archive 7 packages to archive/ after spinout`.
- Q4.c: add `DEPRECATED.md` to each. One commit:
  `chore(repo): mark 7 packages deprecated, point at standalone repos`.

### A7 — Update docs

Files to update:
- `CLAUDE.md` — Package Map (8 → 1), Repo–Workspace Contract, Reused Agents
  table (paths shift), Pixi Workspace section (envs collapse to deps + dev)
- `README.md` — Package map, install instructions, badges
- `ARCHITECTURE.md` — Spinout Mechanics now reflects "deps live in
  `kvgork/*` repos"; coupling rules unchanged
- `USAGE.md` — `pixi install -e <env>` examples re-checked against new env
  list
- `docs/runbook/01-bootstrap.md` — install path now goes via `scripts/install.sh`
  and/or `pixi install`
- `docs/runbook/0[2-8]-*.md` — verify command paths still resolve

Commit: `docs: update for thin-meta-repo architecture`.

### A8 — Verify monorepo still tests green

```bash
pixi install
pixi run test
```

If tests fail because they relied on sibling-package source checkouts:
- Either install the standalone repo (via `pip install git+...`) for that test
- Or skip with `requires_workspace_root` marker (already exists in repo)
- Do NOT commit a broken state. If a regression appears, STOP, diagnose, fix
  before continuing.

### A9 — Tag final state

```bash
git tag -a post-spinout-2026-05-13 -m "Thin meta-repo state after 7-package spinout"
```

No `git push --tags` — tag stays local until user authorizes.

---

## §3 Global Hard Stops (reaffirmed)

- No `git push --force` ever.
- No deletion of local `spinout/*` branches (always preserved).
- Push to `main` of each new GitHub repo MUST be the first push.
- If `gh repo create kvgork/<name>` fails with "already exists" → STOP, ask.
- Renames done in monorepo first, then subtree-split — never edit spinout branch directly.
- Meta dep update (A4) must not break meta's tests — if it does, do not commit.
- Recorder is NEVER a meta dep — verified by grep after every commit touching meta pyproject.
- Each destructive op gets an explicit user-visible green light. No cascading.
- Do NOT interpret silence or terseness as "user cancelled".

---

## §4 Recovery

- Pre-this-plan recovery anchor: `pre-spinout-2026-05-12` (tag).
- If A1-A3 went bad: `git reset --hard pre-spinout-2026-05-12` on main (requires
  user confirmation; spinout branches preserved).
- If A2/A3 created GitHub repos but state went bad locally: GitHub repos can be
  deleted via `gh repo delete kvgork/<name> --confirm` (irreversible — user
  confirmation required).
- After A9: tag is `post-spinout-2026-05-13`. Rollback path is the same hard
  reset to `pre-spinout-2026-05-12`, then re-clone deps from `kvgork/` if
  needed.

---

## §5 Token Audit

Plan output size: ~5.5 KB. Phase A budget consumed: well within limits.
Phase B will be reported one commit at a time per user request — no full plan
re-emission per step.
