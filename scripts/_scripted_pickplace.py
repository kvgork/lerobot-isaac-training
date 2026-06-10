"""Scripted IK pick-and-place controller for SO-101 in sim — demo generator.

The plateau-break (top pick, plans/2026-06-10-data-collection-and-plateau-break-plan.md):
hand-shaped RL plateaus at ~-10.6 (grips + lifts partially, no carry). A SCRIPTED
controller that completes the full pick->place gives demos to warm-start / seed the
policy past the exploration plateau — no hardware, no sim2real gap (sim demos for sim).

Uses Isaac Lab DifferentialIKController to drive gripper_link through Cartesian
waypoints; gripper opened/closed via the last action dim. Action mapping: the env's
JointPositionAction has scale=0.5 + use_default_offset=True, so
    action = (q_desired - q_default) / 0.5
for the arm joints (q_default = rest pose). Gripper action: -1 open, +1 close
(SO-101 closes toward the upper limit).

Phase 1 goal (this script): verify the controller PICKS + PLACES (cube ends within
success radius of the target bin). Then it becomes the demo generator.

Usage:
  LEROBOT_ISAAC_STAGED_REWARD=1 .pixi/envs/sim/bin/python scripts/_scripted_pickplace.py \
      --out outputs/scripted-pickplace.json
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
    ap.add_argument("--out", default="outputs/scripted-pickplace.json")
    ap.add_argument("--obj_x", type=float, default=0.22)
    ap.add_argument("--obj_y", type=float, default=0.05)
    ap.add_argument("--obj_z", type=float, default=0.05)
    ap.add_argument("--tgt_x", type=float, default=0.22)
    ap.add_argument("--tgt_y", type=float, default=-0.13)
    args = ap.parse_args()

    app = _boot_app(True)

    import torch
    from lerobot_isaac_env import make_env
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms

    env = make_env(task=args.task, num_envs=1, headless=True, enable_cameras=False)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["source_object"]

    # EE body + arm joints (exclude the gripper joint from IK).
    ee_name = "gripper_link"
    ee_ids, _ = robot.find_bodies(ee_name)
    ee_idx = int(ee_ids[0])
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    arm_ids, _ = robot.find_joints(arm_joint_names)
    arm_ids = list(arm_ids)
    grip_ids, _ = robot.find_joints("gripper")
    grip_idx = int(grip_ids[0])
    # jacobian row index: for a fixed base, body jacobian index = body_idx - 1
    ee_jacobi_idx = ee_idx - 1

    q_default = robot.data.default_joint_pos.clone()  # (1, n)

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    action_dim = env.action_space.shape[-1]
    obs, _ = env.reset()

    def ee_pos_b():
        """EE position in the robot base frame."""
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        ee_w = robot.data.body_pos_w[:, ee_idx, :]
        ee_quat_w = robot.data.body_quat_w[:, ee_idx, :]
        pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, ee_w, ee_quat_w)
        return pos_b, quat_b

    def step_to(target_b, grip, n=40):
        """Drive EE toward target_b (base frame) for n steps; hold gripper at `grip`."""
        ik.reset()
        cmd = torch.tensor([target_b], device=device, dtype=torch.float32)
        for _ in range(n):
            pos_b, quat_b = ee_pos_b()
            ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :3, arm_ids]
            q_arm = robot.data.joint_pos[:, arm_ids]
            q_des_arm = ik.compute(pos_b, quat_b, jac, q_arm)  # (1, n_arm) absolute joint targets
            # build full action (normalized): arm = (q_des - q_default)/0.5; gripper = grip
            action = torch.zeros((1, action_dim), device=device)
            for k, jid in enumerate(arm_ids):
                action[0, jid] = (q_des_arm[0, k] - q_default[0, jid]) / 0.5
            action[0, grip_idx] = grip
            env.step(action)

    # target z above table for approach/lift
    z_high = args.obj_z + 0.12
    obj_b_target = [args.obj_x, args.obj_y, args.obj_z + 0.02]
    above_obj = [args.obj_x, args.obj_y, z_high]
    above_tgt = [args.tgt_x, args.tgt_y, z_high]
    at_tgt = [args.tgt_x, args.tgt_y, args.obj_z + 0.04]

    trace = {}
    def log(tag):
        op = obj.data.root_pos_w[0].detach().cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
        trace[tag] = {"obj": [float(op[0]), float(op[1]), float(op[2])],
                      "ee": [float(ee[0]), float(ee[1]), float(ee[2])]}
        print(f"[scripted] {tag}: obj={trace[tag]['obj']} ee={trace[tag]['ee']}", flush=True)

    log("start")
    step_to(above_obj, grip=-1.0, n=50)   # 1. move above object, open
    log("above_obj")
    step_to(obj_b_target, grip=-1.0, n=50)  # 2. descend onto object
    log("at_obj")
    step_to(obj_b_target, grip=1.0, n=30)   # 3. close gripper
    log("closed")
    step_to(above_obj, grip=1.0, n=50)      # 4. lift
    log("lifted")
    step_to(above_tgt, grip=1.0, n=60)      # 5. carry above target
    log("above_tgt")
    step_to(at_tgt, grip=1.0, n=40)         # 6. lower
    step_to(at_tgt, grip=-1.0, n=20)        # 7. release
    log("released")

    # success: object xy within 6 cm of target
    op = obj.data.root_pos_w[0].detach().cpu().numpy()
    xy_err = ((op[0] - args.tgt_x) ** 2 + (op[1] - args.tgt_y) ** 2) ** 0.5
    success = bool(xy_err < 0.06)
    result = {"trace": trace, "final_obj": [float(op[0]), float(op[1]), float(op[2])],
              "xy_err_to_target": float(xy_err), "SUCCESS": success}
    print(json.dumps(result, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[scripted] wrote {args.out} SUCCESS={success}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
