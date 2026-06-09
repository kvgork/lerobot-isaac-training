"""Grip-physics diagnostic: can a CLOSED SO-101 gripper hold + lift the cube?

Both run #3b and the place-chase plateau at reward ≈ −12.4 even with the closure
reward + place_success bonus — strong evidence the blocker is PHYSICAL: closing
the jaw doesn't actually grasp the cube, so it can't be lifted/carried. This
probe tests that directly, with NO learning involved:

  1. Move the object between the jaws (teleport to the gripper_link pose).
  2. Command the gripper CLOSED (action toward the closed limit) for N steps.
  3. Command the arm UP (shoulder/elbow lift) while holding the jaw closed.
  4. Measure whether the object's z RISES with the end-effector (gripped) or
     stays on the table (slipped). delta_z_obj vs delta_z_ee.

If the object does NOT rise (slips), the fix ladder is: jaw friction ↑,
gripper effort_limit ↑, cube mass/size ↓, then contact-based grasp. Cheapest
first.

Usage:
  .pixi/envs/sim/bin/python scripts/_grip_physics_probe.py --out outputs/grip-physics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _boot_app(headless: bool):
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=False).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--close_steps", type=int, default=40)
    ap.add_argument("--lift_steps", type=int, default=60)
    ap.add_argument("--out", default="outputs/grip-physics.json")
    args = ap.parse_args()

    app = _boot_app(True)

    import torch
    from lerobot_isaac_env import make_env

    env = make_env(task=args.task, num_envs=1, headless=True, enable_cameras=False)
    device = env.device if hasattr(env, "device") else "cuda"
    robot = env.scene["robot"]
    obj = env.scene["source_object"]
    sim_dt = float(getattr(env, "physics_dt", 1.0 / 120.0))

    def bidx(name):
        try:
            ids, _ = robot.find_bodies(name)
            return int(ids[0]) if len(ids) else 0
        except Exception:
            return 0

    ee_i = bidx("gripper_link")
    action_dim = env.action_space.shape[-1] if env.action_space.shape else 6

    env.reset()

    def step_hold(gripper_val, arm_lift=0.0, n=1):
        """Hold gripper at gripper_val; optionally bias shoulder/elbow up."""
        a = torch.zeros((1, action_dim), device=device)
        a[0, -1] = gripper_val  # gripper (last dim)
        if arm_lift != 0.0:
            # shoulder_lift (idx 1) + elbow_flex (idx 2) bias to raise the EE
            a[0, 1] = arm_lift
            a[0, 2] = -arm_lift
        for _ in range(n):
            env.step(a)

    def obj_xyz():
        p = obj.data.root_pos_w[0].detach().cpu().numpy()
        return [float(p[0]), float(p[1]), float(p[2])]

    def ee_xyz():
        p = robot.data.body_pos_w[0, ee_i, :].detach().cpu().numpy()
        return [float(p[0]), float(p[1]), float(p[2])]

    def put_obj_at_ee():
        ee = robot.data.body_pos_w[0, ee_i, :].clone()
        pose = obj.data.root_pose_w.clone()
        pose[0, :3] = ee
        obj.write_root_pose_to_sim(pose)
        obj.write_data_to_sim()
        env.sim.step(render=False)
        obj.update(sim_dt)

    # Closed direction: SO-101 gripper closes toward the UPPER limit → action +1.
    CLOSE = 1.0

    # 1. open, move object between jaws
    step_hold(-1.0, n=10)
    put_obj_at_ee()
    obj_start = obj_xyz()
    ee_start = ee_xyz()

    # 2. close jaw on the object
    step_hold(CLOSE, n=args.close_steps)
    obj_after_close = obj_xyz()
    ee_after_close = ee_xyz()

    # 3. lift the arm while holding closed
    step_hold(CLOSE, arm_lift=1.0, n=args.lift_steps)
    obj_after_lift = obj_xyz()
    ee_after_lift = ee_xyz()

    d_ee_z = ee_after_lift[2] - ee_after_close[2]
    d_obj_z = obj_after_lift[2] - obj_after_close[2]
    # gripped if the object rose with the EE (track ratio); slipped if it stayed.
    gripped = d_obj_z > 0.02 and (d_ee_z <= 1e-3 or d_obj_z / max(d_ee_z, 1e-6) > 0.5)

    result = {
        "obj_start": obj_start,
        "obj_after_close": obj_after_close,
        "obj_after_lift": obj_after_lift,
        "ee_after_close": ee_after_close,
        "ee_after_lift": ee_after_lift,
        "delta_ee_z_during_lift": d_ee_z,
        "delta_obj_z_during_lift": d_obj_z,
        "VERDICT": "GRIPPED+LIFTED" if gripped else "SLIPPED (closing does not hold the cube)",
    }
    print(json.dumps(result, indent=2), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[grip] wrote {out}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
