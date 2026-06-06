"""
test_cli_env_smoke.py
=====================
Tests for `lerobot-isaac env smoke` CLI subcommand (Bundle C.1).

Dry-run tests do NOT require Isaac Lab / Isaac Sim.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest


def test_env_smoke_dry_run_no_cameras(monkeypatch):
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["env", "smoke", "--dry-run"])

    assert rc == 0
    out = buf.getvalue()
    assert "dry-run" in out
    assert "task              : so101_pickplace" in out
    assert "cameras           : (none)" in out
    assert "enable_cameras    : False" in out


def test_env_smoke_dry_run_with_cameras(monkeypatch):
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "env",
                "smoke",
                "--cameras=wrist,overhead",
                "--camera-resolution=256x192",
                "--steps=50",
                "--dry-run",
            ]
        )

    assert rc == 0
    out = buf.getvalue()
    assert "wrist" in out and "overhead" in out
    assert "256x192" in out
    assert "enable_cameras    : True" in out
    assert "30 frames" in out  # warm-up note


def test_env_smoke_dry_run_custom_task(monkeypatch):
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["env", "smoke", "--task=insertion", "--dry-run"])

    assert rc == 0
    assert "insertion" in buf.getvalue()


def test_env_smoke_real_run_isaaclab_missing_returns_2(monkeypatch):
    """If Isaac Lab is unavailable AND user did NOT pass --dry-run,
    the command returns exit code 2 with a clear message."""
    # Force Isaac Lab unavailability
    monkeypatch.setitem(sys.modules, "isaaclab.app", None)
    monkeypatch.setitem(sys.modules, "omni.isaac.lab.app", None)

    from lerobot_isaac_meta.cli import main

    err_buf = io.StringIO()
    # Capture stderr for the error message
    monkeypatch.setattr(sys, "stderr", err_buf)
    rc = main(["env", "smoke"])

    assert rc == 2
    assert "Isaac Lab not installed" in err_buf.getvalue()
