"""Canonical path resolver for the lerobot-isaac workspace.

Resolution order:
    1. LEROBOT_ISAAC_WORKSPACE env var (if set, used as workspace root)
    2. __file__-based: walks up from this file to the workspace root
       (src/lerobot_isaac_meta/ -> src/ -> lerobot-isaac-meta/ -> packages/ -> workspace root)

Usage:
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT, DATASETS_DIR

    dataset_path = DATASETS_DIR / "so101_pick_real_v1"
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_workspace_root() -> Path:
    """Resolve the workspace root directory.

    Prefers the LEROBOT_ISAAC_WORKSPACE environment variable.
    Falls back to __file__-relative resolution.
    """
    env_root = os.environ.get("LEROBOT_ISAAC_WORKSPACE")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.is_dir():
            return root
        raise FileNotFoundError(
            f"LEROBOT_ISAAC_WORKSPACE={env_root!r} does not exist or is not a directory"
        )

    # Walk up: this file is at packages/lerobot-isaac-meta/src/lerobot_isaac_meta/workspace_paths.py
    # workspace root is 4 levels up
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[4]
    if not workspace_root.is_dir():
        raise FileNotFoundError(
            f"Could not resolve workspace root from __file__={this_file}. "
            "Set LEROBOT_ISAAC_WORKSPACE env var explicitly."
        )
    return workspace_root


WORKSPACE_ROOT: Path = _resolve_workspace_root()

# Workspace-level directories
DATASETS_DIR: Path = WORKSPACE_ROOT / "datasets"
OUTPUTS_DIR: Path = WORKSPACE_ROOT / "outputs"
AGENT_STATE_DIR: Path = WORKSPACE_ROOT / ".agent-state"

# Configs live inside the configs package — resolve via the package tree
CONFIGS_DIR: Path = WORKSPACE_ROOT / "packages" / "lerobot-isaac-configs" / "configs"


def ensure_dirs() -> None:
    """Create gitignored workspace directories if they don't exist.

    Call this at the start of any training or data-collection run.
    Does NOT create package source directories — those are managed by the repo.
    """
    for d in (DATASETS_DIR, OUTPUTS_DIR, AGENT_STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "WORKSPACE_ROOT",
    "DATASETS_DIR",
    "OUTPUTS_DIR",
    "CONFIGS_DIR",
    "AGENT_STATE_DIR",
    "ensure_dirs",
]
