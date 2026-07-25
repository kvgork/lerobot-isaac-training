"""STATISTICAL lift-probe — fixes the 3 flaws of scripts/_probe_demo_grasp.py that voided
the "scripted grasp infeasible" conclusion (workflow wbxrp730v synthesis, 2026-06-23):

  FLAW 1 (wrong threshold): the old probe printed "lift thresh 0.09" — a hardcoded literal.
          The REAL lift_termination threshold is rest_height(0.05)+lift_margin(0.02)=0.07.
  FLAW 2 (n=1): the old probe ran ONE rollout — "always slips" was a single sample.
  FLAW 3 (subsampling): the old probe printed die-z every 20 steps, hiding the hold window.

This probe: N jittered rollouts, a 30-step open-grip SETTLE before reading the die pose, the
IDENTICAL 2026-06-13 SOLVED grasp->lift kinematics (settle/approach50/descend90/stabilize30/
close80/seat25/lift60), logs die-z EVERY step, and reports per-rollout max_die_z plus the
max number of CONSECUTIVE steps the die is held above the lift threshold. It then prints the
would-fire RATE at hold_steps in {1,3,10} for BOTH the current threshold (0.07, margin 0.02)
and the proposed band-aid threshold (0.065, margin 0.015).

time_out fires at episode step 300 (episode_length_s=10 * 30 Hz) and ManagerBasedRLEnv
auto-resets on it; the full sequence + settle is 365 steps, so we DISABLE time_out by bumping
env.max_episode_length, and set LEROBOT_ISAAC_LIFT_HOLD_STEPS=999 so lift_termination never
auto-resets mid-lift either. The {1,3,10} success rates are computed OFFLINE from the clean
die-z trace, independent of the env's configured hold_steps.

Run (friction-fixed SOLVED config):
  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_FRICTION=3.0 \
  LEROBOT_ISAAC_OBJECT_X=0.18 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
  LEROBOT_ISAAC_GRASP_STAGE=1 LEROBOT_ISAAC_LIFT_HOLD_STEPS=999 \
  pixi run -e sim python scripts/_probe_lift_stats.py
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

N_ROLLOUTS = int(os.environ.get("PROBE_N", "30"))
JITTER = float(os.environ.get("PROBE_JITTER", "0.02"))
CENTER_X = float(os.environ.get("LEROBOT_ISAAC_OBJECT_X", "0.18"))
CENTER_Y = float(os.environ.get("LEROBOT_ISAAC_OBJECT_Y", "0.05"))
random.seed(0)  # reproducible jitter sequence

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=True)
# Disable time_out auto-reset: the faithful sequence (365 steps) exceeds the 300-step episode
# cap (episode_length_s=10 * 30 Hz). max_episode_length is a read-only property computed from
# cfg.episode_length_s, so bump the cfg field and the property recomputes huge -> no mid-lift reset.
_orig_max = getattr(env, "max_episode_length", None)
for _attr, _val in (("cfg.episode_length_s", 1.0e6), ("max_episode_length_s", 1.0e6)):
    try:
        if "." in _attr:
            setattr(env.cfg, _attr.split(".", 1)[1], _val)
        elif hasattr(env, _attr):
            setattr(env, _attr, _val)
    except Exception as e:  # pragma: no cover
        print(f"[stats] WARN could not set {_attr}: {e}", flush=True)
print(f"[stats] max_episode_length: {_orig_max} -> {getattr(env, 'max_episode_length', None)}", flush=True)

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
REST = 0.05
# {label: lift threshold}. 0.07 = current env (margin 0.02). 0.065 = proposed band-aid (margin 0.015).
THRESHOLDS = {"thr0.070_m0.020": REST + 0.020, "thr0.065_m0.015": REST + 0.015}
HOLD_STEPS = [1, 3, 10]


def _to_bool(x):
    """Extract a scalar bool from a possibly-CUDA tensor / ndarray / scalar (gym done flags)."""
    try:
        if hasattr(x, "item"):
            return bool(x.reshape(-1)[0].item())
        return bool(np.asarray(x).reshape(-1)[0])
    except Exception:
        return bool(x)


def step_to(target_b, grip, n, quat, trace, phase, grip_end=None):
    """Run n DLS-IK steps toward target_b. Append (phase, die_z) per step to `trace`.
    Returns True if the env terminated/truncated (die-z would be post-reset garbage)."""
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
        out = env.step(action)
        term = _to_bool(out[2])
        trunc = _to_bool(out[3]) if len(out) > 3 else False
        if term or trunc:
            print(f"[stats]   !! {phase} step {s}: term={term} trunc={trunc} -> stop rollout (defensive)", flush=True)
            return True
        oz = float(obj.data.root_pos_w[0, 2])
        trace.append((phase, oz))
    return False


def run_rollout(ox, oy):
    env.reset()
    # teleport die to jittered xy (keep spawn z + identity rot), matching _gen_sim_demos.rollout
    root = obj.data.root_state_w.clone()
    root[0, 0], root[0, 1] = ox, oy
    obj.write_root_state_to_sim(root)
    # FIX #1: 30-step open-grip SETTLE before reading the die pose (die spawn z~0.05 -> rest)
    s = torch.zeros((1, adim), device=device)
    s[0, grip_idx] = GRIP_OPEN
    for _ in range(30):
        env.step(s)
    op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
    gx, gy = float(op0[0]), float(op0[1])
    q = GRASP_QUAT
    trace = []
    seq = [
        ([gx, gy, z_high], GRIP_OPEN, 50, None, "approach"),
        ([gx, gy, grasp_z], GRIP_OPEN, 90, None, "descend"),
        ([gx, gy, grasp_z], GRIP_OPEN, 30, None, "stabilize"),
        ([gx, gy, grasp_z], GRIP_OPEN, 80, GRIP_CLOSE, "close"),
        ([gx, gy, grasp_z], GRIP_CLOSE, 25, None, "seat"),
        ([gx, gy, z_high], GRIP_CLOSE, 60, None, "lift"),
    ]
    early = False
    for tgt, grip, n, gend, phase in seq:
        if step_to(tgt, grip, n, q, trace, phase, grip_end=gend):
            early = True
            break
    return trace, early, (gx, gy)


def max_consec_above(zs, thr):
    best = cur = 0
    for z in zs:
        if z > thr:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


print(
    f"[stats] N={N_ROLLOUTS} jitter=+/-{JITTER} center=({CENTER_X:.3f},{CENTER_Y:.3f}) "
    f"thresholds={ {k: round(v,3) for k,v in THRESHOLDS.items()} } hold_steps={HOLD_STEPS}",
    flush=True,
)
print(
    f"[stats] env: OBJECT_SCALE={os.environ.get('LEROBOT_ISAAC_OBJECT_SCALE','?')} "
    f"FRICTION={os.environ.get('LEROBOT_ISAAC_OBJECT_FRICTION','?')} "
    f"GRASP_STAGE={os.environ.get('LEROBOT_ISAAC_GRASP_STAGE','?')} "
    f"LIFT_HOLD_STEPS={os.environ.get('LEROBOT_ISAAC_LIFT_HOLD_STEPS','?')}",
    flush=True,
)

results = []
for i in range(N_ROLLOUTS):
    ox = CENTER_X + random.uniform(-JITTER, JITTER)
    oy = CENTER_Y + random.uniform(-JITTER, JITTER)
    trace, early, (gx, gy) = run_rollout(ox, oy)
    zs = [z for _, z in trace]
    maxz = max(zs) if zs else -9.0
    consec = {name: max_consec_above(zs, thr) for name, thr in THRESHOLDS.items()}
    results.append({"maxz": maxz, "consec": consec, "early": early, "n": len(zs)})
    print(
        f"[stats] roll {i:2d} obj=({ox:.3f},{oy:.3f}) max_die_z={maxz:.4f} "
        f"consec@0.070={consec['thr0.070_m0.020']:3d} consec@0.065={consec['thr0.065_m0.015']:3d} "
        f"early_term={early} steps={len(zs)}",
        flush=True,
    )
    if i == 0:  # FIX #3: dump the every-step lift trace the old probe hid
        lift_zs = [round(z, 4) for ph, z in trace if ph == "lift"]
        print(f"[stats] roll0 LIFT-phase die_z (every step, n={len(lift_zs)}): {lift_zs}", flush=True)

print("\n[stats] ===== SUMMARY =====", flush=True)
mzs = np.array([r["maxz"] for r in results], dtype="float64")
n_early = sum(1 for r in results if r["early"])
print(
    f"[stats] rollouts={len(results)} early_term={n_early}  "
    f"max_die_z: mean={mzs.mean():.4f} min={mzs.min():.4f} max={mzs.max():.4f} "
    f"frac>0.07={float((mzs > 0.07).mean()):.2f} frac>0.09={float((mzs > 0.09).mean()):.2f}",
    flush=True,
)
for name, thr in THRESHOLDS.items():
    line = f"[stats] {name} (>{thr:.3f}): "
    for h in HOLD_STEPS:
        rate = float(np.mean([1.0 if r["consec"][name] >= h else 0.0 for r in results]))
        line += f"hold>={h:2d}: {rate * 100:5.1f}%   "
    print(line, flush=True)
# Decision rule readout
r1 = float(np.mean([1.0 if r["consec"]["thr0.070_m0.020"] >= 1 else 0.0 for r in results]))
print(
    f"[stats] DECISION: hold_steps=1 @0.07 rate = {r1*100:.1f}%  -> "
    + ("RETRACT 'infeasible'; cheap unblock LIFT_HOLD_STEPS=1 (+ LIFT_MARGIN 0.015) is viable"
       if r1 > 0 else "transient lift does NOT reach 0.07 even at hold=1; revisit grip physics"),
    flush=True,
)
print("[stats] DONE", flush=True)
sys.stdout.flush()
os._exit(0)
