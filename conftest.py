"""
conftest.py — Workspace-root pytest configuration.

Registers custom markers so that ``pytest --strict-markers`` does not
warn about unknown markers in any package under packages/*/tests/.

Also adds all sibling package src directories to sys.path so that
tests can be run from the workspace root without pixi install.

Plan reference: §13.1 Bundle A, deliverable A3
Last-updated: 2026-05-07
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).parent
_PACKAGES = _WORKSPACE_ROOT / "packages"


def pytest_configure(config: object) -> None:
    """Register workspace-level pytest markers."""
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "requires_isaaclab: requires Isaac Lab installed",
    )
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "requires_lerobot: requires lerobot installed",
    )
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "requires_dreamerv3: requires sheeprl or dreamer-v3-pytorch installed",
    )
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "requires_pymupdf4llm: requires pymupdf4llm installed",
    )
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "integration: integration test (slow; requires multiple deps)",
    )
    config.addinivalue_line(  # type: ignore[union-attr]
        "markers",
        "smoke: smoke test",
    )

    # Add all sibling package src directories to sys.path so that
    # packages are importable without being pip-installed.
    # This allows running pytest from the workspace root.
    for pkg_dir in sorted(_PACKAGES.iterdir()) if _PACKAGES.is_dir() else []:
        src = pkg_dir / "src"
        src_str = str(src)
        if src.is_dir() and src_str not in sys.path:
            sys.path.insert(0, src_str)
