"""Config loader — resolves and loads YAML configs by name.

The configs/ directory lives inside the installed package tree at
``lerobot_isaac_configs/configs/`` and is accessed via
``importlib.resources`` so resolution works correctly under wheel install.

Resolution order:
    1. LEROBOT_ISAAC_CONFIGS_DIR env var (if set)
    2. importlib.resources: lerobot_isaac_configs/configs/
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path
from typing import Any

import yaml


def get_configs_dir() -> Path:
    """Return the path to the configs/ directory.

    Deferred to first call — not run at import time.

    Resolution order:
        1. ``LEROBOT_ISAAC_CONFIGS_DIR`` env var (if set and valid).
        2. ``importlib.resources`` lookup inside the installed package
           (works for both editable installs and wheel installs).

    Raises
    ------
    FileNotFoundError
        If ``LEROBOT_ISAAC_CONFIGS_DIR`` is set but the path does not exist,
        or if the package-internal configs/ directory cannot be found.
    """
    env_dir = os.environ.get("LEROBOT_ISAAC_CONFIGS_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"LEROBOT_ISAAC_CONFIGS_DIR={env_dir!r} does not exist or is not a directory"
        )

    # importlib.resources works for both editable and wheel installs.
    pkg_files = importlib.resources.files("lerobot_isaac_configs")
    configs_traversable = pkg_files / "configs"

    # Materialise to a real filesystem path.
    # For non-zipped packages this is a direct Path; for zip/wheel we use
    # as_file() context manager — but since we only need the directory path
    # for glob operations we resolve it directly here (works for sdist/editable).
    configs_dir = Path(str(configs_traversable))
    if not configs_dir.is_dir():
        raise FileNotFoundError(
            f"configs/ directory not found inside lerobot_isaac_configs package "
            f"(resolved to {configs_dir}). "
            "Set LEROBOT_ISAAC_CONFIGS_DIR env var explicitly, or ensure "
            "lerobot_isaac_configs/configs/ exists in the package tree."
        )
    return configs_dir


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by name.

    Args:
        name: Config name without .yaml extension (e.g. "wm_dreamerv3").

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If configs/{name}.yaml does not exist.
        ValueError: If the YAML content is not a mapping (dict) at the top level.
    """
    configs_dir = get_configs_dir()
    config_path = configs_dir / f"{name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in configs_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"Config {name!r} not found at {config_path}. "
            f"Available configs: {available}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config {name!r} must be a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )

    return data


def list_configs() -> list[str]:
    """Return the names of all available configs (without .yaml extension)."""
    configs_dir = get_configs_dir()
    return sorted(p.stem for p in configs_dir.glob("*.yaml"))


__all__ = ["get_configs_dir", "load_config", "list_configs"]
