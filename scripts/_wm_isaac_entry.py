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


def _patch_gym_transform_observation() -> None:
    """Compat shim: sheeprl 0.5.8 calls ``TransformObservation(env, func)``
    (2-arg), but gymnasium 1.2.1 (hard-pinned by Isaac Lab) made
    ``observation_space`` a required positional with no default →
    ``TypeError: ... missing 1 required positional argument: 'observation_space'``.

    Default it to ``None``. Old gymnasium inferred the env's own (unchanged)
    observation_space when omitted, and ``observation_space=None`` in gymnasium
    1.2.1 yields the same unchanged-space behaviour — faithful, not a semantics
    change. Lets the WM-Isaac DreamerV3 sweep run without downgrading gymnasium
    (which would break Isaac Lab's ``gymnasium==1.2.1`` pin).
    """
    import inspect

    import gymnasium.wrappers as gw

    cls = gw.TransformObservation
    osp = inspect.signature(cls.__init__).parameters.get("observation_space")
    if osp is not None and osp.default is inspect.Parameter.empty:
        _orig_init = cls.__init__

        def _init(self, env, func, observation_space=None, *args, **kwargs):
            return _orig_init(self, env, func, observation_space, *args, **kwargs)

        cls.__init__ = _init


def _patch_gym_vector_final_info() -> None:
    """Compat shim: sheeprl 0.5.8 reads per-episode stats from
    ``infos["final_info"]`` (a per-env list — gymnasium <1.0 vector API):

        for i, ep in enumerate(infos["final_info"]):
            if ep is not None:
                aggregator.update("Rewards/rew_avg", ep["episode"]["r"])

    gymnasium 1.2.1 (hard-pinned by Isaac Lab) **removed** ``final_info``. Its
    vector ``RecordEpisodeStatistics`` now reports finished-episode stats as
    ``infos["episode"] = {"r": array, "l": array, ...}`` plus an
    ``infos["_episode"]`` boolean mask. So ``"final_info" in infos`` is always
    False and ``Rewards/rew_avg`` / ``Game/ep_len_avg`` NEVER log — the WM-Isaac
    sweep then ratchets every trial at the ``-9999`` sentinel even though
    training is healthy (losses, which don't read final_info, log fine).

    Rebuild the ``final_info`` list from the new keys on every vector step, so
    sheeprl's existing reader works unchanged. Same incompatibility class as
    ``_patch_gym_transform_observation`` — apply BEFORE importing sheeprl.
    """
    import numpy as np

    try:
        import gymnasium.vector as gv
    except Exception:  # noqa: BLE001
        return

    def _augment(infos: object, n: int) -> None:
        if not (isinstance(infos, dict) and "final_info" not in infos):
            return
        ep = infos.get("episode")
        mask = infos.get("_episode")
        if not (isinstance(ep, dict) and ep.get("r") is not None and mask is not None):
            return
        r = ep["r"]
        length = ep.get("l")
        final_info = [None] * n
        for i in range(n):
            if bool(mask[i]):
                final_info[i] = {
                    "episode": {
                        "r": np.array([float(r[i])]),
                        "l": np.array([float(length[i])]) if length is not None else np.array([0.0]),
                    }
                }
        infos["final_info"] = final_info

    for cls_name in ("SyncVectorEnv", "AsyncVectorEnv"):
        cls = getattr(gv, cls_name, None)
        if cls is None or getattr(cls, "_lerobot_final_info_patched", False):
            continue
        _orig_step = cls.step

        def _step(self, actions, _orig=_orig_step):  # noqa: ANN001
            obs, rew, term, trunc, infos = _orig(self, actions)
            _augment(infos, getattr(self, "num_envs", 1))
            return obs, rew, term, trunc, infos

        cls.step = _step
        cls._lerobot_final_info_patched = True


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

    # 2b. Patch gymnasium for sheeprl 0.5.8 compat (BEFORE importing sheeprl,
    #     which builds the env + vector wrappers at run() time).
    _patch_gym_transform_observation()
    _patch_gym_vector_final_info()

    # 3. Call sheeprl's hydra-decorated `run()` with the remaining argv.
    #    sys.argv[0] is expected to be the program name; rewrite to mimic
    #    `python -m sheeprl`.
    sys.argv[0] = "sheeprl"
    from sheeprl.cli import run

    # Run sheeprl, then FORCE-exit with os._exit to bypass Isaac Sim's atexit
    # SimulationApp.close(), which hangs forever in render() on shutdown. That
    # hang is the WM-Isaac "stall": a finished OR crashed trial keeps the
    # process alive holding GPU/VRAM, never emits a final metric, and looks
    # like training froze. os._exit skips all atexit handlers, so a trial dies
    # the instant run() returns or raises — freeing the GPU for the next trial.
    import os

    exit_code = 0
    try:
        run()
    except SystemExit as exc:  # hydra may sys.exit on a job error
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:  # noqa: BLE001 — log then hard-exit, never hang
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
