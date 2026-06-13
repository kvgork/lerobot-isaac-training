"""Diagnose WHY the closed gripper never pinches the 16mm die (48-pose sweep => 0 grips,
many with push=0 = jaws close in air). Reports:
  - gripper joint position LIMITS (lower/upper) + whether CLOSE command hits the stop,
  - moving_jaw link pose travel open->closed (how far the jaw actually swings),
  - approx fingertip gap proxy at OPEN vs CLOSED (moving_jaw vs gripper_frame distance),
  - friction on the relevant physics materials if reachable.
Fast (os._exit).
"""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r = env.scene["robot"]
gid = int(r.find_joints("gripper")[0][0])
mj = int(r.find_bodies("moving_jaw_so101_v1_link")[0][0])
gf = int(r.find_bodies("gripper_frame_link")[0][0])
ad = env.action_space.shape[-1]

# joint limits
lim = None
for attr in ("joint_pos_limits", "joint_limits", "soft_joint_pos_limits"):
    v = getattr(r.data, attr, None)
    if v is not None:
        lim = v
        print(f"[grip] limits from r.data.{attr}: shape={tuple(v.shape)}", flush=True)
        break
if lim is not None:
    lo = float(lim[0, gid, 0]); hi = float(lim[0, gid, 1])
    print(f"[grip] gripper joint limit: lower={lo:+.4f} upper={hi:+.4f} rad "
          f"(travel {hi-lo:.3f} rad = {np.degrees(hi-lo):.1f} deg)", flush=True)


def drive(grip, label, n=60):
    env.reset()
    a = torch.zeros((1, ad), device=env.device)
    a[0, gid] = grip
    for _ in range(n):
        env.step(a)
    jp = float(r.data.joint_pos[0, gid].item())
    gap = float(torch.linalg.norm(r.data.body_pos_w[0, mj, :] - r.data.body_pos_w[0, gf, :]))
    print(f"[grip] {label}: cmd_action={grip:+.1f} -> joint_pos={jp:+.4f} rad | "
          f"|moving_jaw-gripper_frame|={gap*1000:.1f} mm", flush=True)
    return jp


jo = drive(1.0, "OPEN ")
jc = drive(-1.0, "CLOSE")
# Try commanding BEYOND the normalized range to see if the limit is the action map or the joint.
jcx = drive(-3.0, "CLOSE x3 (probe action-map vs joint stop)")
print(f"[grip] CLOSE reached {jc:+.4f}; CLOSE x3 reached {jcx:+.4f} "
      f"({'ACTION-MAP caps it' if abs(jcx-jc)>0.02 else 'JOINT STOP caps it'})", flush=True)
print(f"[grip] die: OBJECT_SCALE env (0.267 => 16mm cube, half-height rest z=0.008)", flush=True)
sys.stdout.flush()
os._exit(0)
