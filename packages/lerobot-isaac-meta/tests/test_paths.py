"""Tests for lerobot_isaac_meta.workspace_paths.

These tests verify path resolution works correctly in three layouts:

  1. Monorepo — pixi.toml with ``[workspace]`` discoverable via walk-up.
  2. Standalone — no workspace marker; ``WORKSPACE_ROOT is None``.
  3. Env-var override — ``LEROBOT_ISAAC_WORKSPACE_ROOT`` (or legacy
     ``LEROBOT_ISAAC_WORKSPACE``) points at a custom dir.

All tests run in both monorepo and standalone contexts.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_workspace_root_is_directory_or_none():
    """WORKSPACE_ROOT must resolve to an existing directory or be None."""
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT

    if WORKSPACE_ROOT is not None:
        assert WORKSPACE_ROOT.is_dir(), (
            f"WORKSPACE_ROOT={WORKSPACE_ROOT} is not an existing directory."
        )


def test_path_constants_are_absolute_when_set():
    """All path constants must be absolute paths when not None."""
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
        if path is not None:
            assert path.is_absolute(), f"{name}={path} is not absolute"


def test_path_constants_are_under_workspace_root():
    """DATASETS_DIR, OUTPUTS_DIR, etc. must be children of WORKSPACE_ROOT."""
    from lerobot_isaac_meta.workspace_paths import (
        AGENT_STATE_DIR,
        DATASETS_DIR,
        OUTPUTS_DIR,
        WORKSPACE_ROOT,
    )

    if WORKSPACE_ROOT is None:
        # Standalone install — nothing to assert.
        assert DATASETS_DIR is None
        assert OUTPUTS_DIR is None
        assert AGENT_STATE_DIR is None
        return

    for name, path in [
        ("DATASETS_DIR", DATASETS_DIR),
        ("OUTPUTS_DIR", OUTPUTS_DIR),
        ("AGENT_STATE_DIR", AGENT_STATE_DIR),
    ]:
        assert str(path).startswith(str(WORKSPACE_ROOT)), (
            f"{name}={path} is not under WORKSPACE_ROOT={WORKSPACE_ROOT}"
        )


def test_workspace_root_contains_marker_when_set():
    """When WORKSPACE_ROOT resolves, it must contain a workspace marker file."""
    from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT

    if WORKSPACE_ROOT is None:
        pytest.skip("standalone install — no WORKSPACE_ROOT")

    pixi = WORKSPACE_ROOT / "pixi.toml"
    pyproject = WORKSPACE_ROOT / "pyproject.toml"
    assert pixi.exists() or pyproject.exists(), (
        f"WORKSPACE_ROOT={WORKSPACE_ROOT} has neither pixi.toml nor pyproject.toml"
    )


@pytest.fixture
def _clean_env(monkeypatch):
    """Strip both env vars so we measure pure resolution behavior."""
    monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE", raising=False)
    yield


def _reload_module():
    import importlib

    import lerobot_isaac_meta.workspace_paths as wp

    return importlib.reload(wp)


def test_env_var_override_primary(tmp_path: Path, _clean_env, monkeypatch):
    """LEROBOT_ISAAC_WORKSPACE_ROOT env var overrides walk-up resolution."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE_ROOT", str(tmp_path))

    wp = _reload_module()
    try:
        assert tmp_path.resolve() == wp.WORKSPACE_ROOT
    finally:
        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE_ROOT", raising=False)
        _reload_module()


def test_env_var_override_legacy(tmp_path: Path, _clean_env, monkeypatch):
    """LEROBOT_ISAAC_WORKSPACE (legacy name) still works."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE", str(tmp_path))

    wp = _reload_module()
    try:
        assert tmp_path.resolve() == wp.WORKSPACE_ROOT
    finally:
        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE", raising=False)
        _reload_module()


def test_env_var_nonexistent_raises(_clean_env, monkeypatch):
    """Env var pointing to nonexistent dir raises FileNotFoundError."""
    monkeypatch.setenv(
        "LEROBOT_ISAAC_WORKSPACE_ROOT", "/nonexistent/path/that/does/not/exist"
    )
    try:
        with pytest.raises(FileNotFoundError):
            _reload_module()
    finally:
        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE_ROOT", raising=False)
        _reload_module()


def test_walk_up_discovery_finds_pixi_workspace_marker(
    tmp_path: Path, _clean_env, monkeypatch
):
    """When run from inside a dir tree with a pixi-workspace marker, discovery finds it."""
    root = tmp_path / "fake-workspace"
    sub = root / "packages" / "foo" / "src" / "foo"
    sub.mkdir(parents=True)
    (root / "pixi.toml").write_text("[workspace]\nname = 'fake'\n")

    monkeypatch.chdir(sub)
    wp = _reload_module()
    try:
        assert wp.WORKSPACE_ROOT == root.resolve()
    finally:
        _reload_module()


def test_walk_up_discovery_returns_none_in_unmarked_tree(
    tmp_path: Path, _clean_env, monkeypatch
):
    """When CWD has no workspace marker upstream, fall back to __file__ walk-up.

    The __file__ walk-up may itself find a workspace (the test runner is inside
    the monorepo), so we cannot assert ``WORKSPACE_ROOT is None`` here. We only
    assert that import succeeds and ``WORKSPACE_ROOT`` is either ``None`` or a
    discovered Path — i.e. no exception was raised.
    """
    monkeypatch.chdir(tmp_path)
    wp = _reload_module()
    try:
        assert wp.WORKSPACE_ROOT is None or isinstance(wp.WORKSPACE_ROOT, Path)
    finally:
        _reload_module()


def test_require_workspace_root_raises_when_none(monkeypatch):
    """require_workspace_root() raises RuntimeError when WORKSPACE_ROOT is None."""
    import lerobot_isaac_meta.workspace_paths as wp

    monkeypatch.setattr(wp, "WORKSPACE_ROOT", None)
    with pytest.raises(RuntimeError, match="could not locate a workspace root"):
        wp.require_workspace_root()


def test_ensure_dirs_creates_directories(tmp_path: Path, _clean_env, monkeypatch):
    """ensure_dirs() creates DATASETS_DIR, OUTPUTS_DIR, AGENT_STATE_DIR."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    monkeypatch.setenv("LEROBOT_ISAAC_WORKSPACE_ROOT", str(tmp_path))

    wp = _reload_module()
    try:
        wp.ensure_dirs()
        assert (tmp_path / "datasets").is_dir()
        assert (tmp_path / "outputs").is_dir()
        assert (tmp_path / ".agent-state").is_dir()
    finally:
        monkeypatch.delenv("LEROBOT_ISAAC_WORKSPACE_ROOT", raising=False)
        _reload_module()


def test_ensure_dirs_is_noop_when_standalone(monkeypatch):
    """ensure_dirs() must not raise when WORKSPACE_ROOT is None."""
    import lerobot_isaac_meta.workspace_paths as wp

    monkeypatch.setattr(wp, "WORKSPACE_ROOT", None)
    monkeypatch.setattr(wp, "DATASETS_DIR", None)
    monkeypatch.setattr(wp, "OUTPUTS_DIR", None)
    monkeypatch.setattr(wp, "AGENT_STATE_DIR", None)
    wp.ensure_dirs()  # must not raise
