"""AppLauncher-first sheeprl entry for the Isaac Lab WM training.

Isaac Lab requires SimulationApp to boot BEFORE any module-load that
pulls in `omni.kit.*` or shared libs that link against libgobject.
`python -m sheeprl` imports a chain of hydra + lightning + protobuf
packages that claim libgobject FIRST → Isaac Sim's `libgpu.foundation.plugin.so`
later fails to load with `undefined symbol: g_string_copy`.

This entry script:
  1. Boots SimulationApp via AppLauncher (libgobject is now bound to
     Isaac Sim's expected version).
  2. Forwards argv to sheeprl's hydra-decorated `run()`.

Use:
    python scripts/_wm_isaac_entry.py --config-dir <plugin_configs> \
        exp=dreamer_v3 env=isaac_so101 ...

Same args as `python -m sheeprl`, just with AppLauncher pre-booted.
"""
from __future__ import annotations

import sys


def main() -> None:
    # 1. Boot SimulationApp FIRST — claims libgobject + omni.kit.app.
    from isaaclab.app import AppLauncher

    headless = "--no-headless" not in sys.argv
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    # Two update ticks to let extensions settle before sheeprl imports.
    for _ in range(2):
        launcher.app.update()

    # 2. Strip our own flags from sys.argv before handing off to hydra.
    for tok in ("--no-headless",):
        while tok in sys.argv:
            sys.argv.remove(tok)
    # Hydra resolves config_path relative to this file's dir by default.
    # The sheeprl CLI uses `--config-dir <path>` to override → pass through.

    # 3. Call sheeprl's hydra-decorated `run()` with the remaining argv.
    #    sys.argv[0] is expected to be the program name; rewrite to mimic
    #    `python -m sheeprl`.
    sys.argv[0] = "sheeprl"
    from sheeprl.cli import run

    run()


if __name__ == "__main__":
    main()
