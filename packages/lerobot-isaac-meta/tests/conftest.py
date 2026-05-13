"""
conftest.py — pytest configuration for lerobot-isaac-meta tests.

Two responsibilities:

1. In **monorepo mode**, prepend sibling-package ``src/`` directories to
   ``sys.path`` so tests can ``import lerobot_isaac_adapters`` (etc.) without
   requiring ``pixi install`` to have run.

2. Provide an opt-in ``requires_workspace_root`` marker for tests whose
   semantics are genuinely tied to the monorepo workspace layout (e.g. tests
   that assert specific sibling-coupling behavior). Such tests auto-skip
   when running from a standalone spun-out tree.

   Most tests in this package no longer need this marker. They use
   ``importlib.util.find_spec`` to gate themselves on the actual installable
   packages instead — which is portable across layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Workspace root candidate is 3 levels up from this file when in monorepo.
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
_PACKAGES = _WORKSPACE_ROOT / "packages"

# Sibling packages whose presence indicates monorepo layout.
_REQUIRED_SIBLINGS = (
    "lerobot-isaac-adapters",
    "lerobot-isaac-configs",
    "lerobot-isaac-env",
)


def _has_workspace_marker(root: Path) -> bool:
    """True iff `root` declares a uv-workspace or pixi-workspace table."""
    py = root / "pyproject.toml"
    if py.is_file():
        try:
            if (
                "[tool.uv.workspace]" in py.read_text()
                or "[tool.pixi.workspace]" in py.read_text()
            ):
                return True
        except OSError:
            pass
    pixi = root / "pixi.toml"
    if pixi.is_file():
        try:
            text = pixi.read_text()
            if "[workspace]" in text or "[tool.pixi.workspace]" in text:
                return True
        except OSError:
            pass
    return False


def _in_monorepo() -> bool:
    """Return True iff this tree looks like the monorepo workspace layout."""
    if not _PACKAGES.is_dir():
        return False
    if not any((_PACKAGES / s).is_dir() for s in _REQUIRED_SIBLINGS):
        return False
    return _has_workspace_marker(_WORKSPACE_ROOT)


_MONOREPO = _in_monorepo()


# Add sibling src directories to path (editable-install simulation) — monorepo only.
if _MONOREPO:
    for pkg_src in [
        _PACKAGES / "lerobot-isaac-adapters" / "src",
        _PACKAGES / "lerobot-isaac-configs" / "src",
        _PACKAGES / "lerobot-isaac-synthetic" / "src",
        _PACKAGES / "lerobot-isaac-env" / "src",
        _PACKAGES / "lerobot-isaac-autoresearch" / "src",
        _PACKAGES / "lerobot-isaac-recorder" / "src",
    ]:
        pkg_src_str = str(pkg_src)
        if pkg_src_str not in sys.path and pkg_src.is_dir():
            sys.path.insert(0, pkg_src_str)


@pytest.fixture(autouse=True)
def _skip_requires_workspace_root_when_standalone(request):
    """Auto-skip tests marked `requires_workspace_root` when not in monorepo.

    In monorepo mode this fixture is a no-op — the marked tests still run.
    Reserved for tests that genuinely exercise monorepo-specific behavior
    (cross-package coupling, sibling-source-tree probing, etc.). Tests that
    only need a sibling package importable should use ``importlib.util.find_spec``
    skip-conditions instead, since those are portable to standalone trees
    where the sibling is pip-installed.
    """
    if request.node.get_closest_marker("requires_workspace_root") and not _MONOREPO:
        pytest.skip(
            "requires monorepo workspace layout (sibling packages); "
            "skipped in standalone/spun-out tree"
        )
