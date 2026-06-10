"""num_envs=2 diagnostic: is env_1's physics state initialised?

Fix 2 smoke gave reward_env_1=-6.6e14 → env_1 likely has NaN/huge positions.
Boot make_env(num_envs=2) directly (no training), reset, step, and print per-env
robot root, object root, and env_origins to see whether env_1 is (a) properly
offset+initialised, (b) at origin (replication failed), or (c) NaN/huge.
"""
from __future__ import annotations
import sys
import numpy as np


def _boot(headless=True):
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=False).app


def main() -> int:
    app = _boot(True)
    import torch
    from lerobot_isaac_env import make_env
    env = make_env(task="pick_and_place", num_envs=2, headless=True, enable_cameras=False)
    robot = env.scene["robot"]
    obj = env.scene["source_object"]

    def show(tag):
        eo = env.scene.env_origins.detach().cpu().numpy()
        rp = robot.data.root_pos_w.detach().cpu().numpy()
        op = obj.data.root_pos_w.detach().cpu().numpy()
        print(f"[vecdiag] {tag}", flush=True)
        print(f"  env_origins:\n{np.array2string(eo, precision=3)}", flush=True)
        print(f"  robot_root_w:\n{np.array2string(rp, precision=3)}", flush=True)
        print(f"  object_root_w:\n{np.array2string(op, precision=3)}", flush=True)
        print(f"  finite robot={np.isfinite(rp).all()} obj={np.isfinite(op).all()} "
              f"|robot|max={np.abs(rp).max():.3e} |obj|max={np.abs(op).max():.3e}", flush=True)

    env.reset()
    show("after reset")
    act = torch.zeros((2, env.action_space.shape[-1]), device=env.device)
    for _ in range(20):
        env.step(act)
    show("after 20 steps")
    try:
        env.close()
    except Exception:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
