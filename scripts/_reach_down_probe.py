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
        r = (ee[0] ** 2 + ee[1] ** 2) ** 0.5
        rec = {"a": [sh, el, wr], "ee": [round(c, 3) for c in ee], "r": round(r, 3)}
        samples.append(rec)
        if best is None or ee[2] < best["ee"][2]:
            best = rec

    result = {
        "object_rest_z": round(obj_rest[2], 4),
        "object_xy": [round(obj_rest[0], 3), round(obj_rest[1], 3)],
        "min_ee_z_achieved": best["ee"][2],
        "min_ee_z_config": best,
        "vertical_gap_ee_to_object": round(best["ee"][2] - obj_rest[2], 4),
        "VERDICT": ("OBJECT REACHABLE (ee can get near object z)"
                    if (best["ee"][2] - obj_rest[2]) < 0.04
                    else "OBJECT TOO LOW — gripper min z >> object rest z (vertical geometry bug)"),
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
