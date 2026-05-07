"""Tests for lerobot_isaac_configs.loader."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


def test_list_configs_returns_list():
    """list_configs() returns a list (possibly empty in Phase 0)."""
    from lerobot_isaac_configs.loader import list_configs

    result = list_configs()
    assert isinstance(result, list)


def test_load_config_missing_raises(tmp_path: Path, monkeypatch):
    """load_config() raises FileNotFoundError for nonexistent config."""
    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path))

    from lerobot_isaac_configs.loader import load_config

    with pytest.raises(FileNotFoundError, match="nonexistent_config"):
        load_config("nonexistent_config")


def test_load_config_valid_yaml(tmp_path: Path, monkeypatch):
    """load_config() correctly parses a valid YAML config.

    Note: YAML 1.1 (used by PyYAML) does not auto-parse '3e-4' as a float.
    Use explicit decimal notation (0.0003) or quoted strings for LR values.
    Real configs should use 0.0003 notation.
    """
    (tmp_path / "test_cfg.yaml").write_text(
        "image_size: 64\nbatch_size: 16\nlr: 0.0003\n"
    )

    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path))

    from lerobot_isaac_configs.loader import load_config

    cfg = load_config("test_cfg")
    assert cfg["image_size"] == 64
    assert cfg["batch_size"] == 16
    assert cfg["lr"] == pytest.approx(3e-4)


def test_load_config_non_mapping_raises(tmp_path: Path, monkeypatch):
    """load_config() raises ValueError if YAML top-level is not a mapping."""
    (tmp_path / "bad_cfg.yaml").write_text("- item1\n- item2\n")

    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path))

    from lerobot_isaac_configs.loader import load_config

    with pytest.raises(ValueError, match="mapping"):
        load_config("bad_cfg")


def test_list_configs_with_yaml_files(tmp_path: Path, monkeypatch):
    """list_configs() returns sorted names of .yaml files in configs dir."""
    for name in ["policy_act", "wm_dreamerv3", "policy_smolvla"]:
        (tmp_path / f"{name}.yaml").write_text(f"name: {name}\n")

    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path))

    from lerobot_isaac_configs.loader import list_configs

    result = list_configs()
    assert result == sorted(["policy_act", "wm_dreamerv3", "policy_smolvla"])


def test_get_configs_dir_env_var(tmp_path: Path, monkeypatch):
    """get_configs_dir() returns the env-var path when set."""
    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path))

    from lerobot_isaac_configs.loader import get_configs_dir

    assert get_configs_dir() == tmp_path


def test_get_configs_dir_bad_env_var_raises(tmp_path: Path, monkeypatch):
    """get_configs_dir() raises FileNotFoundError for nonexistent env-var path."""
    monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(tmp_path / "nonexistent"))

    from lerobot_isaac_configs.loader import get_configs_dir

    with pytest.raises(FileNotFoundError, match="LEROBOT_ISAAC_CONFIGS_DIR"):
        get_configs_dir()


def test_package_importable_with_isaaclab_stub(tmp_path, monkeypatch):
    """lerobot_isaac_configs imports cleanly even when an isaaclab stub is on sys.path.

    This regression test guards against import-time side-effects that would
    fail if isaaclab is present but not fully initialised.
    """
    # Create a minimal isaaclab stub on a temp path
    stub_dir = tmp_path / "isaaclab_stub"
    stub_dir.mkdir()
    (stub_dir / "isaaclab").mkdir()
    (stub_dir / "isaaclab" / "__init__.py").write_text("# stub\n")

    monkeypatch.syspath_prepend(str(stub_dir))

    # Force reimport with stub on path
    import importlib
    import lerobot_isaac_configs
    import lerobot_isaac_configs.loader as loader_mod

    importlib.reload(lerobot_isaac_configs)
    importlib.reload(loader_mod)

    assert hasattr(loader_mod, "load_config")
    assert hasattr(loader_mod, "list_configs")
    assert hasattr(loader_mod, "get_configs_dir")
