"""lerobot-isaac-configs — shared YAML configuration package.

This is a leaf package with no internal cross-package dependencies.

Public API:
    load_config(name): load a YAML config by name from the configs/ directory
"""

from lerobot_isaac_configs.loader import load_config

__version__ = "0.1.0"

__all__ = ["load_config", "__version__"]
