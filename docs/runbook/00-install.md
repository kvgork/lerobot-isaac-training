# Runbook 00: Install (Thin-Meta-Repo, Local Bare Repo Phase)

**Audience:** Anyone setting up the workspace from scratch after the 2026-05-13 spinout.
**Prerequisites:** git, pixi
**Expected outcome:** `~/workspaces/spinouts/` exists with the 6 sibling bare repos; `pixi install` from the monorepo resolves all 7 packages (meta + 6 siblings).

> **TODO (future session):** swap the `file://` URLs in this runbook + `pixi.toml`
> + `packages/lerobot-isaac-meta/pyproject.toml` for `https://github.com/kvgork/<name>.git`
> once the repos are published. See `09-publish-to-github.md`.

---

## Architecture Recap

After Phase B (2026-05-13) the workspace uses a **thin-meta-repo** layout:

- **Live workspace member (1):** `packages/lerobot-isaac-meta/` — the umbrella CLI + workspace paths
- **Spun-out sibling packages (6), bare repos at `~/workspaces/spinouts/`:**
  - `lerobot-isaac-configs`
  - `lerobot-isaac-dashboard`
  - `lerobot-isaac-autoresearch`
  - `lerobot-isaac-env`
  - `lerobot-isaac-adapters`
  - `lerobot-isaac-synthetic`
- **Standalone hardware package (NOT a meta dep):**
  - `~/workspaces/spinouts/robot_data_recorder/` — install separately via `pixi run sync-recorder`
- **History-only mirrors:** `archive/packages/<name>/` (kept for git blame / archaeology; do NOT edit)

> **URL note (2026-05-13):** the on-disk bare repos do NOT carry a `.git` suffix on
> the directory name (they are just bare directories whose `config` declares
> `bare = true`). `pixi.toml` and `meta/pyproject.toml` reference them without `.git`.
> The recorder repo is currently an underscore-named working tree, not a bare repo.

`pixi.toml` installs meta as an editable workspace member and the 6 siblings
from `git+file://` URLs (default env), OR from local clones at `src/<name>/`
(opt-in `editable` env — see "Editable dev mode" below).

---

## Step 1: Create the bare repos (one-time)

If `~/workspaces/spinouts/` does not exist (or you are on a fresh machine), recreate the bare repos from the monorepo's `spinout/*` branches.

```bash
mkdir -p ~/workspaces/spinouts

# The 6 spinout/* branches exist in the monorepo and are the canonical source.
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  git init --bare ~/workspaces/spinouts/$name
  git push ~/workspaces/spinouts/$name spinout/$name:main
done

# Recorder uses its renamed branch (currently created as a working tree, not bare)
git clone . ~/workspaces/spinouts/robot_data_recorder
(cd ~/workspaces/spinouts/robot_data_recorder && git checkout spinout/lerobot-isaac-recorder)
```

Verify:
```bash
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  echo "=== $name ==="
  git --git-dir=$HOME/workspaces/spinouts/$name log --oneline main | head -3
done
```

You should see 3 recent commits per repo.

---

## Step 2: Install from the monorepo (default mode — git+file://)

```bash
cd ~/workspaces/lerobot-isaac-training
pixi install                    # default env: meta editable + 6 git+file:// installs
pixi run -e default test
```

All ~659 tests in `packages/lerobot-isaac-meta/tests/` + `archive/packages/*/tests/` should pass.
(Recorder tests under `archive/packages/lerobot-isaac-recorder/` are excluded by `--ignore-glob`
because `robot_data_recorder` is not a workspace dep — see Step 4 for opt-in recorder dev.)

To verify the 6 siblings actually came from git+file:// URLs:
```bash
cat .pixi/envs/default/lib/python3.12/site-packages/lerobot_isaac_configs-0.1.0.dist-info/direct_url.json
```
Output should reference `file:///home/koen/workspaces/spinouts/lerobot-isaac-configs`.

---

## Step 3: Editable dev mode (opt-in, no regression to default)

**Use this mode when you want to edit a sibling package's source code and see changes
reflected in the workspace without reinstalling.** Default `pixi install` continues
to work unchanged; switching modes is just a matter of installing a different pixi env.

### Step 3a: Sync the sibling clones

```bash
pixi run sync       # clones the 6 spinouts into src/<name>/  (idempotent)
ls src/             # expect 6 directories
```

`pixi run sync` is a thin wrapper around `scripts/sync/sync_siblings.sh`. It:
- Skips any clone that already exists (re-runnable safely).
- Reads `LEROBOT_SPINOUTS_BASE` (default `file:///home/koen/workspaces/spinouts`) for the source.
- Creates `src/<name>/` as an independent git checkout of each spinout repo.

The `src/<pkg>/` directories are ignored by the workspace `.gitignore` (`/src/*/`) and
are NOT tracked as submodules — they are independent repos. Develop on them, commit,
and push directly to their bare-repo origins.

### Step 3b: Install the `editable` pixi env

```bash
pixi install -e editable      # resolves the 6 siblings as path deps + editable=true
pixi run -e editable test     # ~659 tests passing inside the editable env
```

The `editable` env uses the `editable-siblings` feature, which provides path-dep
overrides for all 6 siblings. The default env's git+file:// installs are NOT activated
inside `editable` (they live in a separate `git-siblings` feature), so there is no
URL conflict.

Verify any sibling actually resolves under `src/`:
```bash
pixi run -e editable python -c "import lerobot_isaac_configs; print(lerobot_isaac_configs.__file__)"
# expected: .../src/lerobot-isaac-configs/src/lerobot_isaac_configs/__init__.py
```

Verify editable install metadata:
```bash
cat .pixi/envs/editable/lib/python3.12/site-packages/lerobot_isaac_configs-0.1.0.dist-info/direct_url.json
# expected: {"url":"file:///.../src/lerobot-isaac-configs","dir_info":{"editable":true}}
```

### Step 3c: Update sibling clones (pull from bare repos)

```bash
pixi run sync-update    # git fetch && git pull --ff-only on each src/lerobot-isaac-*
```

Diverged clones are reported but not rewritten — resolve manually inside `src/<pkg>/`.

### Step 3d: Switching between modes

The two modes are independent pixi environments living side-by-side:

| Command | Activates env | Sibling source |
|---------|---------------|----------------|
| `pixi install`              | `default` | git+file:// from `~/workspaces/spinouts/` |
| `pixi install -e editable`  | `editable` | local editable clones in `src/` |
| `pixi shell -e editable`    | `editable` | (same as above) |

No need to uninstall one before installing the other — pixi keeps each env's `.pixi/envs/<name>/`
isolated.

---

## Step 4: Install heavyweight training deps (lerobot + sheeprl)

`pixi install` resolves the workspace's pixi-managed deps but **does not**
install the two big training libraries. They are pinned/structured in a way
that prevents pixi from co-resolving them with the rest of the workspace:

- **lerobot** (used by `train-policy` and `train-lewm`) — gymnasium version pin
  conflicts with sheeprl's pin.
- **sheeprl** (used by `train-dreamer`) — published wheel metadata pins
  `python<3.12`, but the train-dreamer pixi env runs Python 3.12.

Run the helper script once after `pixi install`:

```bash
bash scripts/install_train_deps.sh             # installs all 3 envs
# Or per-env:
bash scripts/install_train_deps.sh --policy    # train-policy: lerobot[smolvla]
bash scripts/install_train_deps.sh --dreamer   # train-dreamer: sheeprl from git (--ignore-requires-python)
bash scripts/install_train_deps.sh --lewm      # train-lewm: lerobot
```

You can also invoke it via `pixi run install-train-deps`.

Override the lerobot extras with `LEROBOT_EXTRAS`:
```bash
LEROBOT_EXTRAS=all bash scripts/install_train_deps.sh --policy
```

The script is idempotent — re-runs skip already-installed envs.

### Why `--ignore-requires-python` for sheeprl?

sheeprl's pyproject.toml pins `requires-python = ">=3.8,<3.12"`. The
`dreamer_v3` algorithm itself works fine on 3.12 in practice, but `pip`
refuses to install based on the metadata pin alone. The helper script
passes `--ignore-requires-python` on Python ≥ 3.12 to bypass the check.
Drop this flag once sheeprl publishes a 3.12-compatible release.

### Known training-backend gap: LeWorldModel

`lerobot 0.5.x` does NOT ship `lerobot.scripts.train_world_model`. The
`le_world_model` adapter target therefore cannot dispatch a real training
run today — `--dry_run` works but actual training will exit with a
`ModuleNotFoundError`. Tracked as a system-improvement gap. The
DreamerV3 path (`--target_arch dreamerv3`) is unaffected.

---

## Step 5: Opt-in recorder dev clone

`robot-data-recorder` is intentionally NOT a workspace dep. It's a standalone hardware
package. To pull a local clone for development:

```bash
pixi run sync-recorder      # clones into src/robot-data-recorder (idempotent)
# Manually install into whichever env you're using (no auto-add to pyproject):
pixi run -e default pip install -e src/robot-data-recorder
```

---

## Step 6: Install standalone (post-spinout, no monorepo)

If you're on a machine that doesn't have the monorepo cloned, use `scripts/install.sh`:

```bash
# Assumes the bare repos already exist at ~/workspaces/spinouts/
bash scripts/install.sh

# Recorder is standalone — install separately if you need the recorder CLI:
pip install git+file:///home/koen/workspaces/spinouts/robot_data_recorder@main
```

Or directly, without the helper script:
```bash
# Meta itself currently lives in the monorepo (no bare repo). When meta is also
# spun out, use this form:
pip install "git+file:///home/koen/workspaces/spinouts/lerobot-isaac-meta@main[post-spinout]"
```

---

## Environment-variable overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPINOUTS_DIR` | `$HOME/workspaces/spinouts` | Used by `scripts/install.sh` to locate bare repos |
| `LEROBOT_SPINOUTS_BASE` | `file:///home/koen/workspaces/spinouts` | Used by `pixi run sync*` tasks |
| `LEROBOT_ISAAC_WORKSPACE` | auto-detected from `__file__` | Used by `lerobot_isaac_meta.workspace_paths` |

For CI / container use, point `SPINOUTS_DIR` / `LEROBOT_SPINOUTS_BASE` at a cloned mirror.

---

## Troubleshooting

### `pixi install -e editable` fails with "No such file or directory" on `src/<pkg>/`
You skipped Step 3a. Run `pixi run sync` first to clone the spinouts into `src/`.

### `pixi install` fails with "conflicting URLs for package `lerobot-isaac-configs`"
A sibling package appears in both `feature.git-siblings` and `feature.editable-siblings`
inside the same environment. The two features must be mutually exclusive — re-check
the `[environments]` table in `pixi.toml`.

### `pixi install` says "expected given path but none found"
Symptom of stale lockfile referring to a pre-archive path. Fix:
```bash
rm pixi.lock
rm -rf .pixi/envs/default
pixi install -e default
```

### `pip show <pkg>` reports an `Editable project location` pointing at the deleted `packages/<pkg>/`
You have a stale user-level editable install from before the spinout. Clean with:
```bash
pip uninstall -y lerobot-isaac-configs lerobot-isaac-dashboard \
                 lerobot-isaac-autoresearch lerobot-isaac-env \
                 lerobot-isaac-adapters lerobot-isaac-synthetic \
                 robot-data-recorder
```
Then re-run `pixi install -e default` to get the canonical git-based installs.

### `git push ~/workspaces/spinouts/<name>` fails with "remote rejected"
First push to a bare repo of branch `spinout/X` as `main` requires the bare repo to be empty.
If you already have content there, either delete and re-init, or push to a different ref name
and rename inside the bare repo. Do NOT `--force` without explicit user OK.

---

## Related runbooks

- `01-bootstrap.md` — full first-time setup (pixi, Isaac Lab, USD assets)
- `09-publish-to-github.md` — future: swap `file://` URLs for `https://github.com/kvgork/<name>.git`
