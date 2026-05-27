# Runbook 00: Install (Thin-Meta-Repo, GitHub Siblings)

**Audience:** Anyone setting up the workspace from scratch.
**Prerequisites:** git, pixi
**Expected outcome:** `src/` populated with the 6 sibling repos cloned from GitHub;
`pixi install` from the monorepo resolves all 7 packages (meta + 6 siblings) via
editable path deps.

---

## Architecture Recap

The workspace uses a **thin-meta-repo** layout:

- **Live workspace member (1):** `packages/lerobot-isaac-meta/` — the umbrella CLI + workspace paths
- **Sibling packages (6) — public GitHub repos:**
  - `https://github.com/kvgork/lerobot-isaac-configs`
  - `https://github.com/kvgork/lerobot-isaac-dashboard`
  - `https://github.com/kvgork/lerobot-isaac-autoresearch`
  - `https://github.com/kvgork/lerobot-isaac-env`
  - `https://github.com/kvgork/lerobot-isaac-adapters`
  - `https://github.com/kvgork/lerobot-isaac-synthetic`
- **Standalone hardware package (NOT a meta dep):**
  - `https://github.com/kvgork/robot-data-recorder` — install separately via `pixi run sync-recorder`

For local development, `bash scripts/setup.sh` clones all 6 siblings into `src/<name>/`
and then runs `pixi install` (which resolves them as editable path deps from `src/`).
For reproducible/CI installs without local clones, use `pixi install -e frozen` which
pulls siblings directly from the GitHub URLs.

---

## Step 1: Clone and run setup (standard first-time install)

```bash
git clone https://github.com/kvgork/lerobot-isaac-training.git
cd lerobot-isaac-training

# Install pixi if absent
curl -fsSL https://pixi.sh/install.sh | bash && source ~/.bashrc

# Clone the 6 siblings from GitHub into src/ + install the default pixi env
bash scripts/setup.sh
```

`scripts/setup.sh` calls `scripts/sync/sync_siblings.sh`, which:
- Tries an optional local mirror first (reads `LEROBOT_LOCAL_MIRROR` env var if set).
- Falls back to `https://github.com/kvgork/<name>.git` for each sibling.
- Creates `src/<name>/` as an independent git checkout.
- Skips any clone that already exists (idempotent).
- Runs `pixi install` at the end.

Verify:
```bash
ls src/                   # expect 6 directories
pixi run test             # expect ~659 passing, ~14 skipped
```

---

## Step 2: Editable dev mode (already active after setup.sh)

The `default` pixi env installs all 6 siblings as editable path deps from `src/`.
No extra steps needed after `bash scripts/setup.sh`.

Verify any sibling resolves from `src/`:
```bash
pixi run python -c "import lerobot_isaac_configs; print(lerobot_isaac_configs.__file__)"
# expected: .../src/lerobot-isaac-configs/src/lerobot_isaac_configs/__init__.py
```

### Updating sibling clones (pull from GitHub)

```bash
pixi run sync-update    # git fetch && git pull --ff-only on each src/lerobot-isaac-*
```

Diverged clones are reported but not rewritten — resolve manually inside `src/<pkg>/`.

### Switching to frozen mode (CI / repro — no `src/` clones needed)

```bash
pixi install -e frozen      # resolves 6 siblings directly from git+https://github.com/kvgork/...
pixi run -e frozen test
```

The two envs live side-by-side in `.pixi/envs/` — no need to uninstall one before using
the other.

---

## Step 3: Install heavyweight training deps (lerobot + sheeprl)

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
run today — `--dry_run` prints the resolved command and exits 0, but a real
run fails with `ModuleNotFoundError: No module named 'lerobot.scripts.train_world_model'`.
Tracked as a system-improvement gap. The DreamerV3 path (`--target_arch dreamerv3`)
is unaffected.

---

## Step 4: Opt-in recorder dev clone

`robot-data-recorder` is intentionally NOT a workspace dep. It's a standalone hardware
package. To pull a local clone for development:

```bash
pixi run sync-recorder      # clones https://github.com/kvgork/robot-data-recorder into src/robot-data-recorder
# Manually install into whichever env you're using (no auto-add to pyproject):
pixi run -e default pip install -e src/robot-data-recorder
```

---

## Step 5: Install standalone (post-spinout, no monorepo)

If you only need the meta package (e.g. on a deployment machine):

```bash
pip install "packages/lerobot-isaac-meta[post-spinout]"
# This pulls the 6 siblings from git+https://github.com/kvgork/<name>.git@main
```

Or install the recorder separately:
```bash
pip install git+https://github.com/kvgork/robot-data-recorder.git@main
```

---

## Environment-variable overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `LEROBOT_LOCAL_MIRROR` | (unset) | Optional local mirror path; `sync_siblings.sh` tries this before falling back to GitHub |
| `LEROBOT_ISAAC_WORKSPACE` | auto-detected from `__file__` | Used by `lerobot_isaac_meta.workspace_paths` |

For CI / container use where GitHub access is available, no overrides are needed —
`bash scripts/setup.sh` (or `pixi install -e frozen`) works out of the box.

---

## Troubleshooting

### `pixi install` fails with "No such file or directory" on `src/<pkg>/`
You ran `pixi install` before `bash scripts/setup.sh`. Run setup.sh first to clone
the siblings into `src/`.

### `pixi install` fails with "conflicting URLs for package `lerobot-isaac-configs`"
A sibling package appears in both `feature.git-siblings` and `feature.editable-siblings`
inside the same environment. The two features must be mutually exclusive — re-check
the `[environments]` table in `pixi.toml`.

### `pixi install` says "expected given path but none found"
Symptom of stale lockfile. Fix:
```bash
rm pixi.lock
rm -rf .pixi/envs/default
bash scripts/setup.sh
```

### `pip show <pkg>` reports an `Editable project location` pointing at a missing path
You have a stale user-level editable install. Clean with:
```bash
pip uninstall -y lerobot-isaac-configs lerobot-isaac-dashboard \
                 lerobot-isaac-autoresearch lerobot-isaac-env \
                 lerobot-isaac-adapters lerobot-isaac-synthetic \
                 robot-data-recorder
```
Then re-run `bash scripts/setup.sh` to get the canonical installs.

---

## Related runbooks

- `01-bootstrap.md` — full first-time setup (pixi, Isaac Lab, USD assets)
- `09-publish-to-github.md` — how the siblings were published to GitHub; mirror sync guide
