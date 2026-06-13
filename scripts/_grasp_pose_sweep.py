"""Sweep grasp poses (orientation roll + xy bias) and score each by whether the die
actually LIFTS with the gripper. Headless + automated (no eyeball): the physics is the
judge. Reuses the controller's pose-IK. Reports combos sorted by lifted die-z.

Why: 5-DOF arm can't hit true-vertical (~28deg Y-tilt), single moving jaw + gap-center
offset => die squirts out. Instead of hand-tuning one rollout at a time, brute-force a
grid of (roll, bias_x, bias_y) and keep the pose that grips. Fast (os._exit).

  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.18 \
    .pixi/envs/sim/bin/python scripts/_grasp_pose_sweep.py
"""
import os, sys, math, numpy as np
from itertools import product
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
device = env.device
robot = env.scene["robot"]
obj = env.scene["source_object"]
ee_idx = int(robot.find_bodies("gripper_link")[0][0])
arm_ids = list(robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])[0])
grip_idx = int(robot.find_joints("gripper")[0][0])
_fixed = bool(getattr(robot, "is_fixed_base", True))
ee_jacobi_idx = (ee_idx - 1) if _fixed else ee_idx
_OFF = 0 if _fixed else 6
q_default = robot.data.default_joint_pos.clone()
action_dim = env.action_space.shape[-1]
ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
ik = DifferentialIKController(ik_cfg, num_envs=1, device=device)
GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
GRASP_Z = 0.106
FRIC = float(os.environ.get("LEROBOT_ISAAC_OBJECT_FRICTION", "2.0"))

# Bump die contact friction at runtime (default ~0.5 too slippery -> closing jaw
# shoves the die out instead of gripping). PhysX material view; no cfg dependency.
try:
    mat = obj.root_physx_view.get_material_properties().clone()
    mat[..., 0] = FRIC  # static friction
    mat[..., 1] = FRIC  # dynamic friction
    idx = torch.arange(mat.shape[0])
    obj.root_physx_view.set_material_properties(mat, idx)
    print(f"[sweep] die friction set -> static=dynamic={FRIC} (shape {tuple(mat.shape)})", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[sweep] WARN friction set failed: {type(e).__name__}: {e}", flush=True)


def ee_pose_b():
    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
    ew = robot.data.body_pos_w[:, ee_idx, :]
    eq = robot.data.body_quat_w[:, ee_idx, :]
    return subtract_frame_transforms(rp, rq, ew, eq)


def step_to(target_b, quat, grip, n, grip_end=None):
    ik.reset()
    cmd = torch.tensor([list(target_b) + list(quat)], device=device, dtype=torch.float32)
    for s in range(n):
        g = grip if grip_end is None else grip + (grip_end - grip) * (s / max(1, n - 1))
        pos_b, quat_b = ee_pose_b()
        ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :6, [_OFF + j for j in arm_ids]]
        q_des_arm = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids])
        action = torch.zeros((1, action_dim), device=device)
        for k, jid in enumerate(arm_ids):
            action[0, jid] = (q_des_arm[0, k] - q_default[0, jid]) / 0.5
        action[0, grip_idx] = g
        env.step(action)


def roll_quat(deg):
    """Point-down (local -z -> world -z) with roll `deg` about the vertical (world z)."""
    p = math.radians(deg) / 2.0
    return [math.cos(p), 0.0, 0.0, math.sin(p)]


def trial(roll_deg, bx, by):
    env.reset()
    # Let the die SETTLE before reading its height (it spawns at z~0.048 and falls to
    # ~0.008). Reading pre-settle aimed the grasp ~4cm too high -> jaws closed in air.
    settle = torch.zeros((1, action_dim), device=device)
    settle[0, grip_idx] = GRIP_OPEN
    for _ in range(30):
        env.step(settle)
    op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
    q = roll_quat(roll_deg)
    gx, gy = float(op0[0]) + bx, float(op0[1]) + by
    GRASP_Z = float(op0[2]) + 0.10  # gripper_link ~0.10 above die center (fingertips at die/floor)
    z_high = 0.17
    step_to([gx, gy, z_high], q, GRIP_OPEN, 25)
    step_to([gx, gy, GRASP_Z], q, GRIP_OPEN, 45)      # descend
    step_to([gx, gy, GRASP_Z], q, GRIP_OPEN, 12)      # dwell
    step_to([gx, gy, GRASP_Z], q, GRIP_OPEN, 45, grip_end=GRIP_CLOSE)  # slow close
    step_to([gx, gy, GRASP_Z], q, GRIP_CLOSE, 15)     # seat
    step_to([gx, gy, z_high], q, GRIP_CLOSE, 45)      # LIFT
    op = obj.data.root_pos_w[0].detach().cpu().numpy()
    lift_z = float(op[2])
    push = float(((op[0] - op0[0]) ** 2 + (op[1] - op0[1]) ** 2) ** 0.5)
    return lift_z, push


results = []
ROLLS = [0, 45, 90, 135]
BX = [-0.04, -0.02, 0.0, 0.02]
BY = [-0.02, 0.0, 0.02]
print(f"[sweep] grid: {len(ROLLS)}x{len(BX)}x{len(BY)} = {len(ROLLS)*len(BX)*len(BY)} trials. "
      f"die at OBJECT_X env, grasp_z={GRASP_Z}", flush=True)
for roll, bx, by in product(ROLLS, BX, BY):
    lz, push = trial(roll, bx, by)
    gripped = lz > 0.04
    results.append((lz, push, roll, bx, by))
    tag = "GRIP+LIFT" if gripped else "no"
    print(f"[sweep] roll={roll:3d} bx={bx:+.2f} by={by:+.2f} -> lift_z={lz:.3f} push={push*100:4.1f}cm  {tag}", flush=True)

results.sort(reverse=True)
print("[sweep] === TOP 5 by lift_z ===", flush=True)
for lz, push, roll, bx, by in results[:5]:
    print(f"[sweep]   lift_z={lz:.3f} push={push*100:.1f}cm  roll={roll} bias_x={bx} bias_y={by}", flush=True)
best = results[0]
print(f"[sweep] BEST: lift_z={best[0]:.3f} roll={best[2]} bias_x={best[3]} bias_y={best[4]} "
      f"({'GRIPS' if best[0] > 0.04 else 'NO POSE GRIPS - need different approach'})", flush=True)
sys.stdout.flush()
os._exit(0)
