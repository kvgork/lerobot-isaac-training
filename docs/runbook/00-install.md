# Runbook 00: Install (Thin-Meta-Repo, Local Bare Repo Phase)

**Audience:** Anyone setting up the workspace from scratch after the 2026-05-13 spinout.
**Prerequisites:** git, pixi
**Expected outcome:** `~/workspaces/spinouts/` exists with 7 bare repos; `pixi install` from the monorepo resolves all 8 packages.

> **TODO (future session):** swap the `file://` URLs in this runbook + `pixi.toml`
> + `packages/lerobot-isaac-meta/pyproject.toml` for `https://github.com/kvgork/<name>.git`
> once the repos are published. See `09-publish-to-github.md`.

---

## Architecture Recap

After Phase B (2026-05-13) the workspace uses a **thin-meta-repo** layout:

- **Live workspace member (1):** `packages/lerobot-isaac-meta/` — the umbrella CLI + workspace paths
- **Spun-out packages (7), bare repos at `~/workspaces/spinouts/`:**
  - `lerobot-isaac-configs.git`
  - `lerobot-isaac-dashboard.git`
  - `lerobot-isaac-autoresearch.git`
  - `lerobot-isaac-env.git`
  - `lerobot-isaac-adapters.git`
  - `lerobot-isaac-synthetic.git`
  - `robot-data-recorder.git`   (standalone — NOT a meta dep)
- **History-only mirrors:** `archive/packages/<name>/` (kept for git blame / archaeology; do NOT edit)

`pixi.toml` installs meta as an editable workspace member and the 7 siblings
from `git+file://` URLs pointing at the bare repos.

---

## Step 1: Create the bare repos (one-time)

If `~/workspaces/spinouts/` does not exist (or you are on a fresh machine), recreate the bare repos from the monorepo's `spinout/*` branches.

```bash
mkdir -p ~/workspaces/spinouts

# The 8 spinout/* branches exist in the monorepo and are the canonical source.
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  git init --bare ~/workspaces/spinouts/$name.git
  git push ~/workspaces/spinouts/$name.git spinout/$name:main
done

# Recorder uses its renamed branch
git init --bare ~/workspaces/spinouts/robot-data-recorder.git
git push ~/workspaces/spinouts/robot-data-recorder.git spinout/lerobot-isaac-recorder:main
```

Verify:
```bash
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic \
            robot-data-recorder; do
  echo "=== $name ==="
  git --git-dir=$HOME/workspaces/spinouts/$name.git log --oneline main | head -3
done
```

You should see 3 recent commits per repo.

---

## Step 2: Install from the monorepo (development mode)

```bash
cd ~/workspaces/lerobot-isaac-training
pixi install                    # default env: meta editable + 7 git+file:// installs
pixi run -e default pytest packages/lerobot-isaac-meta/tests/
```

All 63 meta tests should pass.

To verify the 7 siblings actually came from git+file:// URLs:
```bash
cat .pixi/envs/default/lib/python3.12/site-packages/lerobot_isaac_configs-0.1.0.dist-info/direct_url.json
```
Output should reference `file:///home/koen/workspaces/spinouts/lerobot-isaac-configs.git`.

---

## Step 3: Install standalone (post-spinout, no monorepo)

If you're on a machine that doesn't have the monorepo cloned, use `scripts/install.sh`:

```bash
# Assumes the bare repos already exist at ~/workspaces/spinouts/
bash scripts/install.sh

# Recorder is standalone — install separately if you need the recorder CLI:
pip install git+file:///home/koen/workspaces/spinouts/robot-data-recorder.git@main
```

Or directly, without the helper script:
```bash
# Currently meta itself lives in the monorepo (no bare repo), so use --editable
# against the local monorepo checkout OR a future bare repo when meta is also spun out.
pip install "git+file:///home/koen/workspaces/spinouts/lerobot-isaac-meta.git@main[post-spinout]"
```

---

## Environment-variable overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPINOUTS_DIR` | `$HOME/workspaces/spinouts` | Used by `scripts/install.sh` to locate bare repos |
| `LEROBOT_ISAAC_WORKSPACE` | auto-detected from `__file__` | Used by `lerobot_isaac_meta.workspace_paths` |

For CI / container use, set `SPINOUTS_DIR` to point at a cloned mirror.

---

## Troubleshooting

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

### `git push ~/workspaces/spinouts/<name>.git` fails with "remote rejected"
First push to a bare repo of branch `spinout/X` as `main` requires the bare repo to be empty.
If you already have content there, either delete and re-init, or push to a different ref name
and rename inside the bare repo. Do NOT `--force` without explicit user OK.

---

## Related runbooks

- `01-bootstrap.md` — full first-time setup (pixi, Isaac Lab, USD assets)
- `09-publish-to-github.md` — future: swap `file://` URLs for `https://github.com/kvgork/<name>.git`
