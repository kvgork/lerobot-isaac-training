"""Min-EE-height probe: can the SO-101 gripper reach DOWN to a table-level object?

Hypothesis: the source_object rests at z≈0.0015 (floor) but the gripper's minimum
reachable height is much higher → ungraspable from the table (a vertical geometry
bug, analogous to the earlier horizontal-reach one). Sweeps shoulder_lift +
elbow_flex + wrist_flex toward downward configs (reliable joint control, NO IK)
and reports the minimum gripper_link z achieved + the object's settled z.
"""
from __future__ import annotations
import argparse, json, sys
from itertools import product
from pathlib import Path


def _boot(headless):
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=False).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--out", default="outputs/reach-down.json")
    args = ap.parse_args()
    app = _boot(True)
    import torch
    from lerobot_isaac_env import make_env
    env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["source_object"]
    ee_ids, _ = robot.find_bodies("gripper_link"); ee_idx = int(ee_ids[0])
    # The fingertip (moving jaw) extends BELOW gripper_link (the wrist) — that's
    # what actually grasps. Measure it, not just the wrist.
    try:
        jaw_ids, _ = robot.find_bodies("moving_jaw_so101_v1_link"); jaw_idx = int(jaw_ids[0])
    except Exception:
        jaw_idx = ee_idx
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    idx = {n: int(robot.find_joints(n)[0][0]) for n in names}
    action_dim = env.action_space.shape[-1]
    env.reset()
    # let object settle
    for _ in range(30):
        env.step(torch.zeros((1, action_dim), device=device))
    obj_rest = obj.data.root_pos_w[0].detach().cpu().tolist()

    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    best = None
    samples = []
    for sh, el, wr in product(levels, repeat=3):
        a = torch.zeros((1, action_dim), device=device)
        a[0, idx["shoulder_lift"]] = sh
        a[0, idx["elbow_flex"]] = el
        a[0, idx["wrist_flex"]] = wr
        for _ in range(args.settle):
            env.step(a)
        ee = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().tolist()
        jaw = robot.data.body_pos_w[0, jaw_idx, :].detach().cpu().tolist()
        r = (jaw[0] ** 2 + jaw[1] ** 2) ** 0.5
        rec = {"a": [sh, el, wr], "ee": [round(c, 3) for c in ee],
               "jaw": [round(c, 3) for c in jaw], "r": round(r, 3)}
        samples.append(rec)
        if best is None or jaw[2] < best["jaw"][2]:
            best = rec

    obj_top = obj_rest[2] * 2.0  # rest center ≈ half-edge → top ≈ 2×
    gap = round(best["jaw"][2] - obj_rest[2], 4)
    result = {
        "object_rest_z": round(obj_rest[2], 4),
        "object_est_edge_m": round(obj_top, 4),
        "object_xy": [round(obj_rest[0], 3), round(obj_rest[1], 3)],
        "min_jaw_z_achieved": best["jaw"][2],
        "min_gripper_link_z": round(min(s["ee"][2] for s in samples), 4),
        "min_jaw_z_config": best,
        "vertical_gap_jaw_to_object_center": gap,
        "VERDICT": ("JAW REACHES OBJECT (graspable height)"
                    if gap < 0.03
                    else "jaw min z above object — check object height / sweep"),
    }
    print(json.dumps(result, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**result, "samples": samples}, indent=2))
    print(f"[reachdown] wrote {args.out}", flush=True)
    try: env.close()
    except Exception: pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
