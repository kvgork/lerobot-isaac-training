"""Unit tests for _sim_eval helpers — CPU-only, no Isaac Lab required.

The module-level imports in scripts/_sim_eval.py are light (argparse, json,
pathlib, uuid), so it imports cleanly without GPU or Isaac Lab.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure scripts/ is on the path so we can import _sim_eval directly.
_SCRIPTS_DIR = Path(__file__).parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _sim_eval import _read_success_term  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build fake env objects
# ---------------------------------------------------------------------------


def _make_env_with_get_term(success_value: bool):
    """Fake env whose termination_manager.get_term('success') returns [success_value]."""
    tm = MagicMock()
    tm.get_term.return_value = [success_value]
    # Ensure hasattr(tm, 'get_term') returns True (MagicMock does by default).
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=tm))
    return env


def _make_env_with_term_dones(success_value: bool):
    """Fake env using _term_dones dict (no get_term attribute)."""

    class FakeTM:
        _term_dones = {"success": [success_value]}

    # Remove get_term so hasattr returns False.
    tm = FakeTM()
    assert not hasattr(tm, "get_term")
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=tm))
    return env


def _make_env_no_termination_manager():
    """Fake env with no termination_manager attribute at all."""
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    return env


def _make_env_get_term_raises():
    """Fake env where get_term raises an exception."""
    tm = MagicMock()
    tm.get_term.side_effect = KeyError("success")
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=tm))
    return env


# ---------------------------------------------------------------------------
# Tests: get_term path
# ---------------------------------------------------------------------------


def test_read_success_term_true_via_get_term():
    env = _make_env_with_get_term(True)
    result = _read_success_term(env)
    assert result is True


def test_read_success_term_false_via_get_term():
    env = _make_env_with_get_term(False)
    result = _read_success_term(env)
    assert result is False


# ---------------------------------------------------------------------------
# Tests: _term_dones fallback path
# ---------------------------------------------------------------------------


def test_read_success_term_true_via_term_dones():
    env = _make_env_with_term_dones(True)
    result = _read_success_term(env)
    assert result is True


def test_read_success_term_false_via_term_dones():
    env = _make_env_with_term_dones(False)
    result = _read_success_term(env)
    assert result is False


# ---------------------------------------------------------------------------
# Tests: None / error paths
# ---------------------------------------------------------------------------


def test_read_success_term_none_when_no_termination_manager():
    env = _make_env_no_termination_manager()
    result = _read_success_term(env)
    assert result is None


def test_read_success_term_none_when_get_term_raises():
    env = _make_env_get_term_raises()
    result = _read_success_term(env)
    assert result is None


def test_read_success_term_none_when_unwrapped_missing():
    """Completely broken env structure returns None, never raises."""
    result = _read_success_term(object())
    assert result is None


def test_read_success_term_none_when_success_key_missing():
    """_term_dones missing 'success' key returns None."""

    class FakeTM:
        _term_dones: dict = {}

    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=FakeTM()))
    result = _read_success_term(env)
    assert result is None


# ---------------------------------------------------------------------------
# Confirm module imports cleanly on CPU (no Isaac Lab)
# ---------------------------------------------------------------------------


def test_module_imports_without_isaac():
    """_sim_eval must be importable without GPU/Isaac — only light stdlib at top level."""
    import importlib

    mod = importlib.import_module("_sim_eval")
    assert callable(mod._read_success_term)
    assert callable(mod.main)
