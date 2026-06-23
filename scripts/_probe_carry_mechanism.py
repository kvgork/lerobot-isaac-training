"""MECHANISM + CARRY diagnostic for the scripted grasp (follow-up to _probe_lift_stats.py).

_probe_lift_stats.py showed the die is held >0.07 for ~56 consecutive steps (lift_termination
would fire) — BUT die-z plateaus dead-flat at ~0.072 while the EE rises to 0.17, so the die is
NOT carried up with the gripper. This probe disambiguates a ROBUST HOLD from a CONTINUOUS SLIP /
caught-on-geometry, and tests whether the grasp survives a real horizontal CARRY to the bin.

Per step (lift + carry + place-descend) logs: phase, die_z, ee_z, ee<->die 3D distance, gripper
JOINT. Interpretation:
  - ROBUST HOLD: ee<->die distance stays ~constant through lift+carry; die_z follows ee_z; die
    reaches the bin XY still elevated -> real pick-and-place.
  - SLIP / not-carried: ee<->die distance GROWS during lift (die left behind); die_z flat; die
    does NOT track to the bin -> degenerate lift_termination hit, NOT a usable carry-place grasp.

Run (same friction-fixed SOLVED config as _probe_lift_stats.py):
  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_FRICTION=3.0 \
  LEROBOT_ISAAC_OBJECT_X=0.18 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
  LEROBOT_ISAAC_GRASP_STAGE=1 LEROBOT_ISAAC_LIFT_HOLD_STEPS=999 \
  pixi run -e sim python scripts/_probe_carry_mechanism.py
"""
import os
import sys

sys.argv = [sys.argv[0]]
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import random

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from lerobot_isaac_env import make_env

N_ROLLOUTS = int(os.environ.get("PROBE_N", "5"))
JITTER = float(os.environ.get("PROBE_JITTER", "0.02"))
CENTER_X = float(os.environ.get("LEROBOT_ISAAC_OBJECT_X", "0.18"))
CENTER_Y = float(os.environ.get("LEROBOT_ISAAC_OBJECT_Y", "0.05"))
TGT_X = float(os.environ.get("LEROBOT_ISAAC_BIN_X", "0.22"))
TGT_Y = float(os.environ.get("LEROBOT_ISAAC_BIN_Y", "-0.13"))
random.seed(0)

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=True)
try:
    env.cfg.episode_length_s = 1.0e6  # disable time_out (see _probe_lift_stats.py)
except Exception as e:
    print(f"[mech] WARN episode_length_s: {e}", flush=True)
print(f"[mech] max_episode_length -> {getattr(env, 'max_episode_length', None)}", flush=True)

device = env.device
robot = env.scene["robot"]
obj = env.scene["source_object"]
ee_idx = int(robot.find_bodies("gripper_link")[0][0])
arm_ids = list(robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])[0])
grip_idx = int(robot.find_joints("gripper")[0][0])
_fixed = bool(getattr(robot, "is_fixed_base", True))
ee_jac = (ee_idx - 1) if _fixed else ee_idx
_OFF = 0 if _fixed else 6
q_default = robot.data.default_joint_pos.clone()
adim = env.action_space.shape[-1]
ik = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    num_envs=1,
    device=device,
)
GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
z_high, grasp_z = 0.17, 0.106


def step_to(target_b, grip, n, quat, log, phase, grip_end=None, print_every=10):
    ik.reset()
    cmd = torch.tensor([list(target_b) + list(quat)], device=device, dtype=torch.float32)
    for s in range(n):
        g = grip if grip_end is None else grip + (grip_end - grip) * (s / max(1, n - 1))
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        pos_b, quat_b = subtract_frame_transforms(
            rp, rq, robot.data.body_pos_w[:, ee_idx, :], robot.data.body_quat_w[:, ee_idx, :]
        )
        ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
        jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :6, [_OFF + j for j in arm_ids]]
        q_des = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids])
        action = torch.zeros((1, adim), device=device)
        for k, jid in enumerate(arm_ids):
            action[0, jid] = (q_des[0, k] - q_default[0, jid]) / 0.5
        action[0, grip_idx] = g
        env.step(action)
        diez = float(obj.data.root_pos_w[0, 2])
        eez = float(robot.data.body_pos_w[0, ee_idx, 2])
        dist = float(((robot.data.body_pos_w[0, ee_idx, :] - obj.data.root_pos_w[0]) ** 2).sum() ** 0.5)
        gj = float(robot.data.joint_pos[0, grip_idx])
        log.append((phase, diez, eez, dist, gj))
        if s % print_every == 0 or s == n - 1:
            print(f"[mech]   {phase:8s} s={s:3d} die_z={diez:.4f} ee_z={eez:.4f} ee-die={dist:.4f} gripJ={gj:+.3f}", flush=True)


def run_rollout(idx, ox, oy):
    env.reset()
    root = obj.data.root_state_w.clone()
    root[0, 0], root[0, 1] = ox, oy
    obj.write_root_state_to_sim(root)
    s = torch.zeros((1, adim), device=device)
    s[0, grip_idx] = GRIP_OPEN
    for _ in range(30):
        env.step(s)
    op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
    gx, gy = float(op0[0]), float(op0[1])
    q = GRASP_QUAT
    log = []
    print(f"[mech] --- roll {idx} obj=({gx:.3f},{gy:.3f}) bin=({TGT_X},{TGT_Y}) ---", flush=True)
    # grasp (printed sparsely)
    step_to([gx, gy, z_high], GRIP_OPEN, 50, q, log, "approach", print_every=999)
    step_to([gx, gy, grasp_z], GRIP_OPEN, 90, q, log, "descend", print_every=999)
    step_to([gx, gy, grasp_z], GRIP_OPEN, 30, q, log, "stabilize", print_every=999)
    step_to([gx, gy, grasp_z], GRIP_OPEN, 80, q, log, "close", grip_end=GRIP_CLOSE, print_every=999)
    step_to([gx, gy, grasp_z], GRIP_CLOSE, 25, q, log, "seat", print_every=999)
    # lift + carry + place-descend (printed every 10 steps — the mechanism window)
    step_to([gx, gy, z_high], GRIP_CLOSE, 60, q, log, "lift", print_every=10)
    step_to([TGT_X, TGT_Y, z_high], GRIP_CLOSE, 60, q, log, "carry", print_every=10)
    step_to([TGT_X, TGT_Y, 0.06], GRIP_CLOSE, 40, q, log, "descend2", print_every=10)
    # final die pose vs bin
    dp = obj.data.root_pos_w[0].detach().cpu().numpy()
    bin_xy = float(((dp[0] - TGT_X) ** 2 + (dp[1] - TGT_Y) ** 2) ** 0.5)
    # per-phase die_z + ee-die summary
    for ph in ("lift", "carry", "descend2"):
        zs = [d for (p, d, _, _, _) in log if p == ph]
        ds = [dd for (p, _, _, dd, _) in log if p == ph]
        if zs:
            print(
                f"[mech] roll {idx} {ph:8s}: die_z[min={min(zs):.4f} max={max(zs):.4f} end={zs[-1]:.4f}] "
                f"ee-die[start={ds[0]:.4f} end={ds[-1]:.4f} grew={ds[-1]-ds[0]:+.4f}]",
                flush=True,
            )
    print(
        f"[mech] roll {idx} FINAL die=({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f}) bin_xy_dist={bin_xy:.4f} "
        f"{'<<< reached bin (XY<0.06)' if bin_xy < 0.06 else ''}",
        flush=True,
    )
    return bin_xy, float(dp[2])


print(
    f"[mech] N={N_ROLLOUTS} jitter=+/-{JITTER} center=({CENTER_X},{CENTER_Y}) bin=({TGT_X},{TGT_Y}) "
    f"FRICTION={os.environ.get('LEROBOT_ISAAC_OBJECT_FRICTION','?')}",
    flush=True,
)
reached, finalz = [], []
for i in range(N_ROLLOUTS):
    ox = CENTER_X + random.uniform(-JITTER, JITTER)
    oy = CENTER_Y + random.uniform(-JITTER, JITTER)
    bx, fz = run_rollout(i, ox, oy)
    reached.append(bx < 0.06)
    finalz.append(fz)
print("\n[mech] ===== SUMMARY =====", flush=True)
print(
    f"[mech] reached_bin(XY<0.06): {sum(reached)}/{len(reached)}   "
    f"final die_z: {[round(z,3) for z in finalz]}",
    flush=True,
)
print("[mech] DONE", flush=True)
sys.stdout.flush()
os._exit(0)
