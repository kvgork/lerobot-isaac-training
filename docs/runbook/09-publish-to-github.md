# Runbook 09: GitHub Publishing — Status and Mirror Sync

**Status as of 2026-05-27:** COMPLETE. All 7 sibling packages are published as public
GitHub repositories at `https://github.com/kvgork/<name>`. The `pixi.toml` and all
install scripts have been updated to use `https://github.com/kvgork/...` URLs.

**Prerequisites (for re-sync / mirror operations):** GitHub account, `gh` CLI
authenticated, write access to the `kvgork` org.

---

## Published repos

| Package | GitHub URL | Meta dep? |
|---|---|---|
| `lerobot-isaac-configs` | https://github.com/kvgork/lerobot-isaac-configs | yes |
| `lerobot-isaac-dashboard` | https://github.com/kvgork/lerobot-isaac-dashboard | yes |
| `lerobot-isaac-autoresearch` | https://github.com/kvgork/lerobot-isaac-autoresearch | yes |
| `lerobot-isaac-env` | https://github.com/kvgork/lerobot-isaac-env | yes |
| `lerobot-isaac-adapters` | https://github.com/kvgork/lerobot-isaac-adapters | yes |
| `lerobot-isaac-synthetic` | https://github.com/kvgork/lerobot-isaac-synthetic | yes |
| `lerobot-isaac-deploy` | https://github.com/kvgork/lerobot-isaac-deploy | no (standalone) |
| `robot-data-recorder` | https://github.com/kvgork/robot-data-recorder | no (standalone) |

---

## How the migration was done (historical record)

The 6 meta deps were originally local bare repos (pre-publication phase)
and installed via local file URLs. Migration to GitHub proceeded as follows:

1. Created GitHub repos via `gh repo create kvgork/<name> --public`.
2. Pushed each bare repo's `main` branch to GitHub.
3. Updated `pixi.toml` `[pypi-dependencies]` git URLs to
   `git+https://github.com/kvgork/<name>.git@main`.
4. Updated `packages/lerobot-isaac-meta/pyproject.toml` `[post-spinout]` optional
   deps similarly.
5. Updated `scripts/sync/sync_siblings.sh` to clone from GitHub (with optional local
   mirror override via `LEROBOT_LOCAL_MIRROR`).
6. Removed all local-file URL and bare-repo path references from docs.
7. Re-tested: `rm pixi.lock && pixi install && pixi run test` — all tests passing.

---

## Keeping siblings in sync

### Pushing changes from a local `src/<name>/` clone to GitHub

```bash
cd src/lerobot-isaac-adapters
# make your edits, commit them, then:
git push origin main
```

Each `src/<name>/` directory is an independent git checkout of the sibling's GitHub
repo. The workspace `.gitignore` ignores `src/<pkg>/` entirely.

### Pulling updates from GitHub into all local clones

```bash
pixi run sync-update    # git fetch && git pull --ff-only on each src/lerobot-isaac-*
```

### Optional: keeping a local mirror for offline use

If you want a local mirror (e.g. for offline development), set `LEROBOT_LOCAL_MIRROR`
to a directory containing bare clones before running `setup.sh`:

```bash
export LEROBOT_LOCAL_MIRROR=$HOME/mirrors
mkdir -p $LEROBOT_LOCAL_MIRROR

for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  git clone --mirror https://github.com/kvgork/$name.git $LEROBOT_LOCAL_MIRROR/$name
done

# Re-sync a mirror from GitHub
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  git --git-dir=$LEROBOT_LOCAL_MIRROR/$name fetch --all
done
```

`sync_siblings.sh` checks `$LEROBOT_LOCAL_MIRROR/<name>` first; if present it clones
from the local mirror, then adds the GitHub remote as `origin` for future pushes.

---

## Tagging releases

```bash
# In each sibling repo (or via GitHub releases UI):
cd src/lerobot-isaac-adapters
git tag v0.2.0
git push origin v0.2.0
```

Then update the pinned version in `pixi.toml` / `pyproject.toml` if you want a
version-locked install:
```toml
lerobot-isaac-adapters = {git = "https://github.com/kvgork/lerobot-isaac-adapters.git", tag = "v0.2.0"}
```

---

## Related

- `00-install.md` — current install path (GitHub-based setup.sh flow)
- `01-bootstrap.md` — full first-time setup
