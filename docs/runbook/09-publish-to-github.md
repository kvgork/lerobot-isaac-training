# Runbook 09: Publish to GitHub (Future Step)

**Audience:** Owner planning to swap local bare repos for GitHub repos.
**Status as of 2026-05-13:** NOT YET DONE. The 7 spun-out packages live as bare
repos at `~/workspaces/spinouts/<name>.git`. This runbook is the plan for when
they migrate to `https://github.com/kvgork/<name>.git`.

**Prerequisites:** GitHub account, `gh` CLI authenticated, write access to the
target org (`kvgork`).
**Expected outcome:** 7 public GitHub repos (or however many you elect to make
public), and all `git+file://` URLs in this workspace swapped for
`https://github.com/kvgork/<name>.git`.

---

## Repos to publish

| Bare repo (local) | GitHub repo (target) | Meta dep? |
|---|---|---|
| `~/workspaces/spinouts/lerobot-isaac-configs.git` | `github.com/kvgork/lerobot-isaac-configs` | yes |
| `~/workspaces/spinouts/lerobot-isaac-dashboard.git` | `github.com/kvgork/lerobot-isaac-dashboard` | yes |
| `~/workspaces/spinouts/lerobot-isaac-autoresearch.git` | `github.com/kvgork/lerobot-isaac-autoresearch` | yes |
| `~/workspaces/spinouts/lerobot-isaac-env.git` | `github.com/kvgork/lerobot-isaac-env` | yes |
| `~/workspaces/spinouts/lerobot-isaac-adapters.git` | `github.com/kvgork/lerobot-isaac-adapters` | yes |
| `~/workspaces/spinouts/lerobot-isaac-synthetic.git` | `github.com/kvgork/lerobot-isaac-synthetic` | yes |
| `~/workspaces/spinouts/robot-data-recorder.git` | `github.com/kvgork/robot-data-recorder` | no (standalone) |

---

## Step 1: Create the GitHub repos

```bash
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic \
            robot-data-recorder; do
  gh repo create kvgork/$name --public --description "Spun-out from lerobot-isaac-training monorepo."
done
```

Choose `--private` instead of `--public` per repo as appropriate.

---

## Step 2: Push bare repos to GitHub

```bash
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic \
            robot-data-recorder; do
  git --git-dir=$HOME/workspaces/spinouts/$name.git \
      push https://github.com/kvgork/$name.git main:main
done
```

Verify:
```bash
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic \
            robot-data-recorder; do
  gh repo view kvgork/$name --json defaultBranchRef --jq '.defaultBranchRef.name'
done
```

Each should print `main`.

---

## Step 3: Swap `file://` URLs for `https://` URLs

Files to edit:

1. `pixi.toml` (root) — `[pypi-dependencies]` git URLs
2. `packages/lerobot-isaac-meta/pyproject.toml` — `[project.optional-dependencies]
   post-spinout` git URLs
3. `scripts/install.sh` — comment / fallback URLs
4. `docs/runbook/00-install.md` — example URLs
5. `CLAUDE.md` — package map URLs
6. `README.md` — install snippet URLs

Quick sed pass (preview first; do NOT pipe directly to `-i`):

```bash
git grep -l "file:///home/koen/workspaces/spinouts" -- '*.toml' '*.sh' '*.md'

# Then preview the replacement per file:
sed 's|git+file:///home/koen/workspaces/spinouts/\([^.]\+\)\.git|git+https://github.com/kvgork/\1.git|g' pixi.toml | diff pixi.toml -

# Apply when satisfied:
git grep -l "file:///home/koen/workspaces/spinouts" -- '*.toml' '*.sh' '*.md' \
  | xargs sed -i 's|git+file:///home/koen/workspaces/spinouts/\([^.]\+\)\.git|git+https://github.com/kvgork/\1.git|g'
```

Make sure the result handles the recorder rename correctly — `robot-data-recorder.git`
maps to `github.com/kvgork/robot-data-recorder` (no transformation needed).

---

## Step 4: Re-test

```bash
rm pixi.lock
rm -rf .pixi/envs/default
pixi install -e default
pixi run -e default pytest packages/lerobot-isaac-meta/tests/
```

All 63 meta tests should pass; the resolved `direct_url.json` for each sibling
should now point at `https://github.com/kvgork/...`.

---

## Step 5: Remove TODO markers

After successful swap:
```bash
git grep -n "TODO.*file://" -- '*.toml' '*.sh' '*.md'
```

Remove each marker and commit:
```
chore: switch from local bare repos to github.com/kvgork/<name>.git
```

---

## Step 6: Decide what to do with the local bare repos

Options:
1. **Keep them as offline-mirrors** — useful when GitHub is unreachable. Re-sync
   periodically with `git --git-dir=$HOME/workspaces/spinouts/<name>.git pull origin main`.
2. **Delete them** — `rm -rf ~/workspaces/spinouts/` once GitHub is the canonical source.

Recommended: keep them for at least one development cycle so a network outage
doesn't break `pixi install`.

---

## Step 7: Tag the migration

```bash
git tag github-published-YYYY-MM-DD
```

---

## Related

- `00-install.md` — current install path (local bare repos)
- `01-bootstrap.md` — full first-time setup
