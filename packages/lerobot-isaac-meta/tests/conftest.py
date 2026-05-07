"""
conftest.py — pytest configuration for lerobot-isaac-meta tests.

Adds sibling package src directories to sys.path so tests can import
lerobot_isaac_adapters and other siblings without requiring pixi install.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Workspace root is 3 levels up from this file
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
_PACKAGES = _WORKSPACE_ROOT / "packages"

# Add sibling src directories to path (editable install simulation)
for pkg_src in [
    _PACKAGES / "lerobot-isaac-adapters" / "src",
    _PACKAGES / "lerobot-isaac-configs" / "src",
    _PACKAGES / "lerobot-isaac-synthetic" / "src",
    _PACKAGES / "lerobot-isaac-env" / "src",
    _PACKAGES / "lerobot-isaac-autoresearch" / "src",
]:
    pkg_src_str = str(pkg_src)
    if pkg_src_str not in sys.path and pkg_src.is_dir():
        sys.path.insert(0, pkg_src_str)
