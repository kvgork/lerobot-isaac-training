"""Tests for lerobot_isaac_meta.workspace_paths.

These tests verify path resolution works correctly from both __file__-relative
and environment-variable-based resolution.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_workspace_root_is_directory():
    """WORKSPACE_ROOT must resolve to an existing directory."""
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT

    assert WORKSPACE_ROOT.is_dir(), (
        f"WORKSPACE_ROOT={WORKSPACE_ROOT} is not an existing directory. "
        "Check that the package is installed from the correct workspace location."
    )


def test_workspace_root_contains_pyproject():
    """WORKSPACE_ROOT should contain pyproject.toml (workspace marker)."""
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT

    assert (WORKSPACE_ROOT / "pyproject.toml").exists(), (
        f"pyproject.toml not found at {WORKSPACE_ROOT / 'pyproject.toml'}. "
        "WORKSPACE_ROOT may be resolving to the wrong directory."
    )


def test_path_constants_are_absolute():
    """All path constants must be absolute paths."""
    from lerobot_isaac_meta.workspace_paths import (
        AGENT_STATE_DIR,
        CONFIGS_DIR,
        DATASETS_DIR,
        OUTPUTS_DIR,
        WORKSPACE_ROOT,
    )

    for name, path in [
        ("WORKSPACE_ROOT", WORKSPACE_ROOT),
        ("DATASETS_DIR", DATASETS_DIR),
        ("OUTPUTS_DIR", OUTPUTS_DIR),
        ("CONFIGS_DIR", CONFIGS_DIR),
        ("AGENT_STATE_DIR", AGENT_STATE_DIR),
    ]:
        assert path.is_absolute(), f"{name}={path} is not absolute"


def test_path_constants_are_under_workspace_root():
    """DATASETS_DIR, OUTPUTS_DIR, etc. must be children of WORKSPACE_ROOT."""
    from lerobot_isaac_meta.workspace_paths import (
        AGENT_STATE_DIR,
        DATASETS_DIR,
        OUTPUTS_DIR,
        WORKSPACE_ROOT,
    )

    for name, path in [
        ("DATASETS_DIR", DATASETS_DIR),
        ("OUTPUTS_DIR", OUTPUTS_DIR),
        ("AGENT_STATE_DIR", AGENT_STATE_DIR),
    ]:
        assert str(path).startswith(str(WORKSPACE_ROOT)), (
            f"{name}={path} is not under WORKSPACE_ROOT={WORKSPACE_ROOT}"
        )


def test_env_var_override(tmp_path: Path):
    """LEROBOT_ISAAC_WORKSPACE env var overrides __file__-based resolution."""
    # Create a fake pyproject.toml so the override dir looks like a workspace
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    original = os.environ.get("LEROBOT_ISAAC_WORKSPACE")
    try:
        os.environ["LEROBOT_ISAAC_WORKSPACE"] = str(tmp_path)

        # Re-import to pick up the env var
        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        importlib.reload(wp)

        assert tmp_path.resolve() == wp.WORKSPACE_ROOT
    finally:
        if original is None:
            os.environ.pop("LEROBOT_ISAAC_WORKSPACE", None)
        else:
            os.environ["LEROBOT_ISAAC_WORKSPACE"] = original

        # Reload to restore original resolution
        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        importlib.reload(wp)


def test_env_var_nonexistent_raises():
    """LEROBOT_ISAAC_WORKSPACE pointing to nonexistent dir raises FileNotFoundError."""
    original = os.environ.get("LEROBOT_ISAAC_WORKSPACE")
    try:
        os.environ["LEROBOT_ISAAC_WORKSPACE"] = "/nonexistent/path/that/does/not/exist"

        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        with pytest.raises(FileNotFoundError):
            importlib.reload(wp)
    finally:
        if original is None:
            os.environ.pop("LEROBOT_ISAAC_WORKSPACE", None)
        else:
            os.environ["LEROBOT_ISAAC_WORKSPACE"] = original

        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        importlib.reload(wp)


def test_ensure_dirs_creates_directories(tmp_path: Path):
    """ensure_dirs() creates DATASETS_DIR, OUTPUTS_DIR, AGENT_STATE_DIR."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    original = os.environ.get("LEROBOT_ISAAC_WORKSPACE")
    try:
        os.environ["LEROBOT_ISAAC_WORKSPACE"] = str(tmp_path)

        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        importlib.reload(wp)
        wp.ensure_dirs()

        assert (tmp_path / "datasets").is_dir()
        assert (tmp_path / "outputs").is_dir()
        assert (tmp_path / ".agent-state").is_dir()
    finally:
        if original is None:
            os.environ.pop("LEROBOT_ISAAC_WORKSPACE", None)
        else:
            os.environ["LEROBOT_ISAAC_WORKSPACE"] = original

        import importlib
        import lerobot_isaac_meta.workspace_paths as wp

        importlib.reload(wp)
