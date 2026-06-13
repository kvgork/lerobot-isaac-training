"""How low can the SO-101 fingertip actually reach at the die xy, and at what tilt?

Resolves the kinematic question behind the scripted-grasp blocker: the 5-DOF arm
CANNOT point straight down and descend vertically at (0.22,0.05) (approach-axis probe:
best down_dot=0.849 @ gripper_link z=0.116). This probe finds, for the die xy, the
joint config that puts the JAW (fingertip proxy) closest to the die in 3D, and reports
that jaw z + the approach tilt (down_dot) + the gripper_link pose. That tells us the
best ACHIEVABLE grasp pose (likely a forward-tilted reach, not a top-down descent).

Fast: resets between configs (valid physics), coarse grid, os._exit to skip Isaac's
hanging app.close() teardown (CLAUDE.md WM-Isaac pitfall).

Run:  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 \
        .pixi/envs/sim/bin/python scripts/_grasp_reach_probe.py
"""
import os, sys, numpy as np
from itertools import product
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
from isaaclab.utils.math import quat_apply

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r = env.scene["robot"]; obj = env.scene["source_object"]
ee = r.find_bodies("gripper_link")[0][0]
jaw = r.find_bodies("moving_jaw_so101_v1_link")[0][0]
nm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
idx = {n: int(r.find_joints(n)[0][0]) for n in nm}
ad = env.action_space.shape[-1]
env.reset()
op = obj.data.root_pos_w[0].cpu().numpy()
print(f"[reach] object rest={np.array2string(op, precision=3)}", flush=True)
approach_local = torch.tensor([[0.0, 0.0, -1.0]], device=env.device)  # gripper_link local -z = fingers
down = torch.tensor([0.0, 0.0, -1.0], device=env.device)

best = None
# pan aims the arm plane at the die; sweep the 3 pitch joints to drop the jaw low.
for sp, sh, el, wf in product([-0.3, 0.0], [-1, -0.4, 0.3, 1],
                              [-1, -0.4, 0.3, 1], [-1, 0, 1]):
    env.reset()
    a = torch.zeros((1, ad), device=env.device)
    a[0, idx["shoulder_pan"]] = sp; a[0, idx["shoulder_lift"]] = sh
    a[0, idx["elbow_flex"]] = el; a[0, idx["wrist_flex"]] = wf
    a[0, idx["gripper"]] = -1.0  # open
    for _ in range(18):
        env.step(a)
    jp = r.data.body_pos_w[0, jaw, :].cpu().numpy()
    eep = r.data.body_pos_w[0, ee, :]
    q = r.data.body_quat_w[0, ee, :]
    aw = quat_apply(q.unsqueeze(0), approach_local)[0]
    aw = aw / torch.linalg.norm(aw)
    down_dot = float(torch.dot(aw, down))
    d3 = float(((jp[0] - op[0]) ** 2 + (jp[1] - op[1]) ** 2 + (jp[2] - op[2]) ** 2) ** 0.5)
    if best is None or d3 < best["d3"]:
        best = {"d3": d3, "a": [sp, sh, el, wf], "jaw": jp.tolist(),
                "ee": eep.cpu().numpy().tolist(), "quat": q.cpu().numpy().tolist(),
                "down_dot": down_dot}

print(f"[reach] BEST jaw-to-die: d3={best['d3']:.3f} a(sp,sh,el,wf)={best['a']}", flush=True)
print(f"[reach]   jaw_pos={np.array2string(np.array(best['jaw']), precision=3)} "
      f"(die={np.array2string(op, precision=3)})", flush=True)
print(f"[reach]   gripper_link z={best['ee'][2]:.3f} approach down_dot={best['down_dot']:+.3f} "
      f"(1=straight down, 0=horizontal)", flush=True)
print(f"[reach]   gripper_link quat(w,x,y,z)={np.array2string(np.array(best['quat']), precision=4)}", flush=True)
verdict = "REACHES die" if best["d3"] < 0.03 else ("CLOSE" if best["d3"] < 0.06 else "CANNOT reach die")
print(f"[reach] VERDICT: {verdict} (jaw within {best['d3']*100:.1f} cm of die center)", flush=True)
sys.stdout.flush()
os._exit(0)  # skip Isaac's hanging app.close() teardown
