# Contributing to lerobot-isaac-training

## Setup

### 1. Clone and enter the workspace

```bash
git clone <repo-url> lerobot-isaac-training
cd lerobot-isaac-training
```

### 2. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
```

### 3. Install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

After this, every `git commit` will run:
- Trailing whitespace / EOF / YAML / TOML / large-file checks
- `ruff check --fix` + `ruff format`

Every `git push` will additionally run:
- `pytest` on any test files touched since the last commit

---

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check packages/

# Auto-fix
ruff check --fix packages/

# Format
ruff format packages/
```

Configuration lives in the root `pyproject.toml` under `[tool.ruff]`.

Key rules enforced:
- `E` / `W` — pycodestyle errors and warnings
- `F` — pyflakes
- `I` — isort-compatible import ordering
- `UP` — pyupgrade (target: Python 3.10)

Docstrings follow Google style. Type annotations are required on all public functions.

---

## Soft-Import Discipline

Heavy optional dependencies (`isaaclab`, `lerobot`, `dreamerv3`, `leworldmodel`) **must not**
be imported at module top-level. Use lazy imports inside functions:

```python
def get_env():
    try:
        import isaaclab  # noqa: F401
    except ImportError as exc:
        raise ImportError("Isaac Lab not installed. Run: pixi run install-isaac-lab") from exc
    ...
```

Tests that require a heavy dep must be marked:

```python
@pytest.mark.requires_isaaclab
def test_env_step(): ...
```

CI skips these markers. See `docs/concepts/soft-import-discipline.md` for full rationale.

---

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`

Scope is the package short name (e.g., `configs`, `adapters`, `env`) or `workspace`.

Examples:
```
feat(adapters): add DreamerV3 metric extractor
fix(configs): correct SO-101 joint limits in reach_target.yaml
docs(adr): add ADR-0005 modular target arch
ci: pin setup-pixi to v0.8.1
```

---

## Pull Request Flow

1. Branch off `main`:
   ```bash
   git checkout -b feat/<short-name>
   ```

2. Keep PRs focused — one logical change per PR.

3. Ensure all checks pass locally before pushing:
   ```bash
   pixi run test   # unit tests, no heavy deps
   pixi run lint   # ruff check
   pixi run fmt    # ruff format check
   ```

4. Open the PR against `main`. The CI matrix will run per-package jobs for Python 3.10
   and 3.11 plus the workspace-level integration job.

5. All required status checks must pass before merge.

6. Squash-merge is preferred for feature branches; merge commits for release branches.

---

## Running Tests

```bash
# All packages, no heavy deps
pixi run test

# Single package
cd packages/lerobot-isaac-configs
python -m pytest tests/ -q

# With Isaac Lab (requires GPU + install-isaac-lab)
pixi run -e sim python -m pytest packages/lerobot-isaac-env/tests/ -m requires_isaaclab
```

See `USAGE.md` for the full runbook.
