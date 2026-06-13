"""At the straight-down grasp pose over the die, print arm joint positions vs their
limits + the achieved down_dot. Decides whether the ~28deg tilt is a hard kinematic
wall (joints maxed) or IK-damping/reachable (joints have room -> fixable). Fast.

  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.18 \
    .pixi/envs/sim/bin/python scripts/_grasp_joint_diag.py
"""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
device = env.device
r = env.scene["robot"]; obj = env.scene["source_object"]
ee_idx = int(r.find_bodies("gripper_link")[0][0])
arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
arm_ids = list(r.find_joints(arm_names)[0])
grip_idx = int(r.find_joints("gripper")[0][0])
_fixed = bool(getattr(r, "is_fixed_base", True))
ee_jac = (ee_idx - 1) if _fixed else ee_idx
_OFF = 0 if _fixed else 6
q_default = r.data.default_joint_pos.clone()
ad = env.action_space.shape[-1]
ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device)
DOWN = [1.0, 0.0, 0.0, 0.0]
env.reset()
op = obj.data.root_pos_w[0].cpu().numpy()
target = [float(op[0]), float(op[1]), float(op[2]) + 0.098]
cmd = torch.tensor([target + DOWN], device=device, dtype=torch.float32)
ik.reset()
for _ in range(120):
    rp, rq = r.data.root_pos_w, r.data.root_quat_w
    pos_b, quat_b = subtract_frame_transforms(rp, rq, r.data.body_pos_w[:, ee_idx, :], r.data.body_quat_w[:, ee_idx, :])
    ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
    jac = r.root_physx_view.get_jacobians()[:, ee_jac, :6, [_OFF + j for j in arm_ids]]
    q_des = ik.compute(pos_b, quat_b, jac, r.data.joint_pos[:, arm_ids])
    a = torch.zeros((1, ad), device=device)
    for k, jid in enumerate(arm_ids):
        a[0, jid] = (q_des[0, k] - q_default[0, jid]) / 0.5
    a[0, grip_idx] = 1.0
    env.step(a)

lim = r.data.joint_pos_limits[0]
jp = r.data.joint_pos[0]
q_ee = r.data.body_quat_w[0, ee_idx, :]
down = quat_apply(q_ee.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=device))[0]
down_dot = float(down[2] / torch.linalg.norm(down))
ee_p = r.data.body_pos_w[0, ee_idx, :].cpu().numpy()
print(f"[diag] target={np.array2string(np.array(target),precision=3)} ee_reached={np.array2string(ee_p,precision=3)}", flush=True)
print(f"[diag] down_dot={down_dot:+.3f} (1=straight down). die={np.array2string(op,precision=3)}", flush=True)
print("[diag] arm joints (pos vs limits, % of range used toward a limit):", flush=True)
for n, jid in zip(arm_names, arm_ids):
    lo, hi = float(lim[jid, 0]), float(lim[jid, 1])
    p = float(jp[jid])
    near = "  <-- AT LIMIT" if (p - lo < 0.05 or hi - p < 0.05) else ""
    print(f"[diag]   {n:13s} pos={p:+.3f}  limits=[{lo:+.3f},{hi:+.3f}]{near}", flush=True)
print("[diag] If NO joint is AT LIMIT but down_dot<<1 -> IK damping/reach, not a hard wall.", flush=True)
sys.stdout.flush()
os._exit(0)
