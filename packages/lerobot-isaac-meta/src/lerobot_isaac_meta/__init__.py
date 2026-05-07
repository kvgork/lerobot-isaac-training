"""lerobot-isaac-meta — umbrella package for the SO-101 training workspace.

Public API:
    cli: lerobot-isaac CLI entrypoint (argparse subcommand registry)
    workspace_paths: canonical path resolver for workspace directories
"""

__version__ = "0.1.0"

from lerobot_isaac_meta import cli, workspace_paths

__all__ = ["cli", "workspace_paths", "__version__"]
