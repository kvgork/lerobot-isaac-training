"""Diagnostic: does IsaacSO101Env.compute_scripted_action() drive a grasp+lift in
isolation? Run an episode applying ONLY the scripted action (no policy, no decay).

LEROBOT_ISAAC_RESIDUAL_PROBE_CLAMP=1 → clamp the action to [-1,1] (mimic the residual
patch) to isolate whether the clamp (grill finding C2) breaks the reach/grasp.
"""
import os
import sys

import numpy as np

CLAMP = os.environ.get("LEROBOT_ISAAC_RESIDUAL_PROBE_CLAMP", "0") not in ("0", "", "false")
print(f"[probe] CLAMP={CLAMP}", flush=True)

from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env

env = IsaacSO101Env(
    task="pickplace", num_envs=1, headless=True, enable_cameras=True,
    image_size=64, max_episode_steps=300,
)
obs, info = env.reset()
robot = env._isaac_env.scene["robot"]
obj = env._isaac_env.scene["source_object"]
ee_idx = int(robot.find_bodies("gripper_link")[0][0])

max_obj_z = -9.0
grasped = False
for t in range(300):
    a = env.compute_scripted_action()
    if a is None:
        print(f"[probe] compute_scripted_action -> None at t={t}", flush=True)
        break
    amax = float(np.abs(a).max())
    if CLAMP:
        a = np.clip(a, -1.0, 1.0)
    obs, rew, term, trunc, info = env.step(a)
    oz = float(obj.data.root_pos_w[0, 2])
    max_obj_z = max(max_obj_z, oz)
    if t % 15 == 0 or term:
        op = obj.data.root_pos_w[0].detach().cpu().numpy()
        ep = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
        print(
            f"[probe] t={t:3d} |a|max={amax:.2f} grip={a[5]:+.2f} "
            f"ee=({ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f}) "
            f"obj=({op[0]:.3f},{op[1]:.3f},{op[2]:.3f}) rew={rew:+.2f} term={bool(term)}",
            flush=True,
        )
    if term:
        grasped = True
        print(f"[probe] *** TERMINATED (held-lift) at t={t} ***", flush=True)
        break

print(f"[probe] RESULT grasped={grasped} max_obj_z={max_obj_z:.3f} (rest~0.05)", flush=True)
sys.stdout.flush()
os._exit(0)
