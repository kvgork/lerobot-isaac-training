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
    # Grasp tuning (eyeball loop): gripper_link bottoms ~0.10 = fingertips at floor,
    # so grasp_z ~0.105-0.115 brackets the die without ramming the ground.
    ap.add_argument("--grasp_z", type=float, default=0.108,
                    help="gripper_link target z at the grasp (NOT fingertip; ~0.10 offset).")
    ap.add_argument("--bias_x", type=float, default=0.0,
                    help="x offset added to the die xy for descend/close (cancel push/undershoot).")
    ap.add_argument("--bias_y", type=float, default=0.0, help="y offset added to the die xy.")
    ap.add_argument("--dwell", type=int, default=30,
                    help="steps to hold the open gripper at grasp depth before closing (let it settle).")
    ap.add_argument("--close_steps", type=int, default=80,
                    help="steps to RAMP the gripper open->closed (slow close pinches; fast close bats the die out).")
    ap.add_argument("--hold", type=int, default=0,
                    help="after closing on the die, hold the closed grasp in place for N steps "
                         "(keeps the GUI live at the grip moment for collider inspection). Skips lift/carry.")
    ap.add_argument("--interactive", action="store_true",
                    help="keep Isaac booted; after each grasp, read commands from stdin to "
                         "reset+replay with new grasp_z/bias_x/bias_y/roll (no reboot). Type 'q' to quit.")
    ap.add_argument("--gui", action="store_true",
                    help="Open the Isaac viewport (headless=False) so the rollout can be watched. "
                         "Default off for batch demo-gen.")
    args = ap.parse_args()

    app = _boot_app(not args.gui)

    import torch
    from lerobot_isaac_env import make_env
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms

    env = make_env(task=args.task, num_envs=1, headless=not args.gui, enable_cameras=False)
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
    # Jacobian indexing depends on base type:
    #   fixed-base  (is_fixed_base=True):  shape (N, num_bodies-1, 6, num_dof),
    #       EE row = ee_idx-1, joint cols = arm_ids (no offset).
    #   floating-base: shape (N, num_bodies, 6, num_dof+6),
    #       EE row = ee_idx, joint cols = arm_ids + 6 (root DOFs precede joints).
    _fixed = bool(getattr(robot, "is_fixed_base", True))
    ee_jacobi_idx = (ee_idx - 1) if _fixed else ee_idx
    _JAC_DOF_OFFSET = 0 if _fixed else 6

    q_default = robot.data.default_joint_pos.clone()  # (1, n)

    # Pose IK (position + orientation) so the gripper points DOWN to grasp the
    # table die. The 5-DOF arm can't hit an arbitrary 6-DOF pose → DLS approximates.
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)
    # Downward-grasp orientation (gripper_link quat, base≈world) from
    # _approach_axis_probe.py: gripper_link local -z is the finger/approach axis;
    # Straight-DOWN grasp: gripper_link local -z is the finger/approach axis
    # (from _approach_axis_probe.py), so world identity quat points the fingers at
    # world -Z. base≈world for the fixed base at origin. (w,x,y,z). Reach is NOT a
    # limiter (owner-confirmed) — the prior side/forward grab was wrong orientation.
    GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
    # Gripper action sign (eyeball-verified rollout #3): +1 drives the gripper joint
    # toward the upper limit = OPEN; -1 toward the lower limit = CLOSE. (The old
    # docstring claimed the reverse; it descended closed and opened to carry.)
    GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0

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

    def step_to(target_b, grip, n=40, quat=None, grip_end=None):
        """Drive EE toward target_b (base-frame pos) + orientation `quat` (default
        GRASP_QUAT, pointing down) for n steps. Gripper held at `grip`, OR ramped
        linearly from `grip` to `grip_end` across the n steps when grip_end is given
        (a slow close pinches the die instead of batting it through the jaw)."""
        ik.reset()
        q_des = quat if quat is not None else GRASP_QUAT
        cmd = torch.tensor([list(target_b) + list(q_des)], device=device, dtype=torch.float32)  # (1,7)
        for _step in range(n):
            g = grip if grip_end is None else grip + (grip_end - grip) * (_step / max(1, n - 1))
            pos_b, quat_b = ee_pos_b()
            ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
            jac_cols = [_JAC_DOF_OFFSET + j for j in arm_ids]
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :6, jac_cols]  # pose: 6 rows
            q_arm = robot.data.joint_pos[:, arm_ids]
            q_des_arm = ik.compute(pos_b, quat_b, jac, q_arm)  # (1, n_arm) absolute joint targets
            # build full action (normalized): arm = (q_des - q_default)/0.5; gripper = grip
            action = torch.zeros((1, action_dim), device=device)
            for k, jid in enumerate(arm_ids):
                action[0, jid] = (q_des_arm[0, k] - q_default[0, jid]) / 0.5
            action[0, grip_idx] = g
            env.step(action)

    import math
    import select

    def roll_quat(deg):
        """Straight-down (local -z -> world -Z) with `deg` roll about the vertical."""
        p = math.radians(deg) / 2.0
        return [math.cos(p), 0.0, 0.0, math.sin(p)]

    z_high = args.obj_z + 0.12
    trace = {}
    def log(tag):
        op = obj.data.root_pos_w[0].detach().cpu().numpy()
        ee = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
        trace[tag] = {"obj": [float(op[0]), float(op[1]), float(op[2])],
                      "ee": [float(ee[0]), float(ee[1]), float(ee[2])]}
        print(f"[scripted] {tag}: obj={trace[tag]['obj']} ee={trace[tag]['ee']}", flush=True)

    def do_grasp(grasp_z, bias_x, bias_y, roll_deg):
        """Reset, SETTLE the die, then approach + descend + slow close. Leaves the arm
        at the closed grasp pose. Returns (obj_b_target, gx, gy, quat)."""
        env.reset()
        s = torch.zeros((1, action_dim), device=device)
        s[0, grip_idx] = GRIP_OPEN
        for _ in range(30):           # let die settle (spawn z~0.048 -> rest ~0.008)
            env.step(s)
        op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
        gx, gy = float(op0[0]) + bias_x, float(op0[1]) + bias_y
        q = roll_quat(roll_deg)
        obj_b_target = [gx, gy, grasp_z]
        above_obj = [gx, gy, z_high]
        log("start")
        step_to(above_obj, grip=GRIP_OPEN, n=50, quat=q)      # 1. above, open
        log("above_obj")
        step_to(obj_b_target, grip=GRIP_OPEN, n=90, quat=q)   # 2. descend, open
        log("at_obj")
        step_to(obj_b_target, grip=GRIP_OPEN, n=args.dwell, quat=q)        # 2b. settle
        step_to(obj_b_target, grip=GRIP_OPEN, n=args.close_steps,
                quat=q, grip_end=GRIP_CLOSE)                  # 3. SLOW ramped close
        step_to(obj_b_target, grip=GRIP_CLOSE, n=25, quat=q)  # 3b. seat
        log("closed")
        return obj_b_target, gx, gy, q

    def do_place(gx, gy, q):
        """Lift -> carry -> release at the target bin. Returns SUCCESS bool."""
        step_to([gx, gy, z_high], grip=GRIP_CLOSE, n=60, quat=q)
        log("lifted")
        step_to([args.tgt_x, args.tgt_y, z_high], grip=GRIP_CLOSE, n=60, quat=q)
        log("above_tgt")
        step_to([args.tgt_x, args.tgt_y, 0.06], grip=GRIP_CLOSE, n=40, quat=q)
        step_to([args.tgt_x, args.tgt_y, 0.06], grip=GRIP_OPEN, n=20, quat=q)
        log("released")
        op = obj.data.root_pos_w[0].detach().cpu().numpy()
        return bool(((op[0] - args.tgt_x) ** 2 + (op[1] - args.tgt_y) ** 2) ** 0.5 < 0.06)

    # ---- Interactive mode: reset+replay on keyboard commands, no reboot ----
    if args.interactive:
        gz, bx, by, roll = args.grasp_z, args.bias_x, args.bias_y, 0.0
        menu = ("[scripted] INTERACTIVE. Type a command + Enter (sim stays live):\n"
                "  r          replay grasp with current params\n"
                "  z <val>    set grasp_z (gripper_link height)   e.g. z 0.10\n"
                "  x <val>    set bias_x                           e.g. x -0.02\n"
                "  y <val>    set bias_y                           e.g. y 0.01\n"
                "  o <deg>    set roll about vertical              e.g. o 90\n"
                "  f          run full lift->carry->place once\n"
                "  p          print current params\n"
                "  q          quit\n")
        tgt, gx, gy, q = do_grasp(gz, bx, by, roll)
        print(menu, flush=True)
        print(f"[scripted] params: grasp_z={gz} bias_x={bx} bias_y={by} roll={roll}", flush=True)
        while True:
            step_to(tgt, grip=GRIP_CLOSE, n=1, quat=q)   # idle-hold keeps GUI live
            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            line = sys.stdin.readline().strip()
            if not line:
                continue
            c, _, rest = line.partition(" ")
            c = c.lower()
            rest = rest.strip()
            try:
                if c == "q":
                    break
                elif c == "r":
                    pass
                elif c == "z":
                    gz = float(rest)
                elif c == "x":
                    bx = float(rest)
                elif c == "y":
                    by = float(rest)
                elif c == "o":
                    roll = float(rest)
                elif c == "p":
                    print(f"[scripted] params: grasp_z={gz} bias_x={bx} bias_y={by} roll={roll}", flush=True)
                    continue
                elif c == "f":
                    ok = do_place(gx, gy, q)
                    print(f"[scripted] full place SUCCESS={ok}", flush=True)
                    tgt, gx, gy, q = do_grasp(gz, bx, by, roll)  # re-grasp, back to hold
                    continue
                else:
                    print(menu, flush=True)
                    continue
            except ValueError:
                print(f"[scripted] bad value: {line!r}", flush=True)
                continue
            print(f"[scripted] replay: grasp_z={gz} bias_x={bx} bias_y={by} roll={roll}", flush=True)
            tgt, gx, gy, q = do_grasp(gz, bx, by, roll)
        try:
            env.close()
        except Exception:
            pass
        app.close()
        return 0

    # ---- One-shot mode ----
    tgt, gx, gy, q = do_grasp(args.grasp_z, args.bias_x, args.bias_y, 0.0)
    if args.hold > 0:
        print(f"[scripted] HOLD: holding closed grasp {args.hold} steps (inspect now).", flush=True)
        step_to(tgt, grip=GRIP_CLOSE, n=args.hold, quat=q)
        log("held")
        print(json.dumps({"trace": trace, "mode": "hold"}, indent=2), flush=True)
        try:
            env.close()
        except Exception:
            pass
        app.close()
        return 0
    success = do_place(gx, gy, q)
    op = obj.data.root_pos_w[0].detach().cpu().numpy()
    result = {"trace": trace, "final_obj": [float(op[0]), float(op[1]), float(op[2])],
              "xy_err_to_target": float(((op[0] - args.tgt_x) ** 2 + (op[1] - args.tgt_y) ** 2) ** 0.5),
              "SUCCESS": success}
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
