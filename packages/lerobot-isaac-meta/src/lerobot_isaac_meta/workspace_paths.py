"""Canonical path resolver for the lerobot-isaac workspace.

This module supports both monorepo and standalone (post-spinout) layouts:

Resolution order for ``WORKSPACE_ROOT``:
    1. ``LEROBOT_ISAAC_WORKSPACE_ROOT`` env var (preferred name).
    2. ``LEROBOT_ISAAC_WORKSPACE`` env var (legacy name, still honored).
    3. Walk upward from CWD looking for a directory that declares a
       workspace marker (``pixi.toml`` with ``[workspace]`` or
       ``[tool.pixi.workspace]``, or ``pyproject.toml`` with
       ``[tool.uv.workspace]`` / ``[tool.pixi.workspace]``).
    4. Walk upward from this file's location for the same marker.
    5. ``None`` — package was installed standalone outside any workspace.
       Path constants then become ``None`` and ``ensure_dirs()`` becomes a
       no-op. Callers that hard-require a workspace should check for ``None``
       explicitly or call :func:`require_workspace_root`.

Usage:
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT, DATASETS_DIR

    if DATASETS_DIR is None:
        ...  # standalone install, no shared datasets/ dir
    else:
        dataset_path = DATASETS_DIR / "so101_pick_real_v1"

Design rationale:
    The previous implementation used ``Path(__file__).parents[4]`` which
    silently produced wrong paths in any non-monorepo layout. By using
    a marker-driven walk-up we tolerate site-packages installs, virtualenv
    layouts, and standalone-repo layouts without raising at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR_PRIMARY = "LEROBOT_ISAAC_WORKSPACE_ROOT"
_ENV_VAR_LEGACY = "LEROBOT_ISAAC_WORKSPACE"

# Filenames + substring markers we treat as "this is a workspace root".
_PIXI_MARKERS = ("[workspace]", "[tool.pixi.workspace]")
_PYPROJECT_MARKERS = ("[tool.pixi.workspace]", "[tool.uv.workspace]")


def _has_workspace_marker(root: Path) -> bool:
    """Return True iff `root` declares a uv/pixi workspace table."""
    pixi = root / "pixi.toml"
    if pixi.is_file():
        try:
            text = pixi.read_text()
        except OSError:
            text = ""
        if any(m in text for m in _PIXI_MARKERS):
            return True
    py = root / "pyproject.toml"
    if py.is_file():
        try:
            text = py.read_text()
        except OSError:
            text = ""
        if any(m in text for m in _PYPROJECT_MARKERS):
            return True
    return False


def _walk_up_for_marker(start: Path) -> Path | None:
    """Walk upward from `start` returning the first dir with a workspace marker."""
    try:
        start = start.resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if _has_workspace_marker(candidate):
            return candidate
    return None


def _resolve_workspace_root() -> Path | None:
    """Resolve the workspace root directory or ``None`` if standalone.

    Honors env-var overrides first (preferred name then legacy name).
    Falls back to walking up from CWD, then from this file's location.
    """
    for var in (_ENV_VAR_PRIMARY, _ENV_VAR_LEGACY):
        env_root = os.environ.get(var)
        if env_root:
            root = Path(env_root).expanduser().resolve()
            if root.is_dir():
                return root
            raise FileNotFoundError(
                f"{var}={env_root!r} does not exist or is not a directory"
            )

    # Walk up from CWD first — this is what an interactive user expects.
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    if cwd is not None:
        found = _walk_up_for_marker(cwd)
        if found is not None:
            return found

    # Fall back to __file__-relative walk-up. Handles editable installs.
    found = _walk_up_for_marker(Path(__file__).resolve().parent)
    if found is not None:
        return found

    return None


WORKSPACE_ROOT: Path | None = _resolve_workspace_root()

# Workspace-level directories — ``None`` when standalone.
DATASETS_DIR: Path | None = WORKSPACE_ROOT / "datasets" if WORKSPACE_ROOT else None
OUTPUTS_DIR: Path | None = WORKSPACE_ROOT / "outputs" if WORKSPACE_ROOT else None
AGENT_STATE_DIR: Path | None = (
    WORKSPACE_ROOT / ".agent-state" if WORKSPACE_ROOT else None
)
CONFIGS_DIR: Path | None = (
    WORKSPACE_ROOT / "packages" / "lerobot-isaac-configs" / "configs"
    if WORKSPACE_ROOT
    else None
)


def require_workspace_root() -> Path:
    """Return ``WORKSPACE_ROOT`` or raise ``RuntimeError`` with a helpful message.

    Use this in code paths that genuinely cannot proceed without a workspace
    (e.g. dataset placement, run orchestration). Library-level code should
    prefer to handle ``None`` gracefully.
    """
    if WORKSPACE_ROOT is None:
        raise RuntimeError(
            "lerobot_isaac_meta could not locate a workspace root. "
            f"Set ${_ENV_VAR_PRIMARY} to an existing directory containing a "
            "workspace marker (pixi.toml with [workspace] or pyproject.toml "
            "with [tool.pixi.workspace] / [tool.uv.workspace]), or run from "
            "inside a monorepo checkout."
        )
    return WORKSPACE_ROOT


def ensure_dirs() -> None:
    """Create gitignored workspace directories if they don't exist.

    No-op when ``WORKSPACE_ROOT is None`` (standalone install).
    """
    if WORKSPACE_ROOT is None:
        return
    for d in (DATASETS_DIR, OUTPUTS_DIR, AGENT_STATE_DIR):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "WORKSPACE_ROOT",
    "DATASETS_DIR",
    "OUTPUTS_DIR",
    "CONFIGS_DIR",
    "AGENT_STATE_DIR",
    "ensure_dirs",
    "require_workspace_root",
]
