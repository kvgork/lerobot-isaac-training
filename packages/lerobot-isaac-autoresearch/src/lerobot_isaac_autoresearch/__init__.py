"""
lerobot_isaac_autoresearch

Autoresearch ML loop wiring for LeRobot policies and world models.

Provides:
  - programs/         — program.md configs consumed by autoresearch-loop-orchestrator
  - train_wrapper.py  — shim that forwards args to lerobot_isaac_adapters.train
                        so autoresearch-ml-executor-worker has one stable entrypoint
"""

__version__ = "0.1.0"
