"""Generate SIM pick-place demos with the working scripted controller -> LeRobotDataset.

Stage 2 of plans/2026-06-11-demo-warmstart-plan.md. Runs the (now working) scripted
straight-down grasp N times with small object-pose jitter, records per-step
(observation.state, observation.images.d435_rgb @ 64x64, action), and writes ONLY the
SUCCESS episodes to a LeRobotDataset for the DreamerV3 warm-start.

Per-episode jitter is done by teleporting the die with write_root_pose_to_sim AFTER
reset (the spawn xy is fixed at env build; OBJECT_X/Y env vars only set the default).
The controller reads the LIVE die pose so it auto-aims at the jittered position.

  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
    .pixi/envs/sim/bin/python scripts/_gen_sim_demos.py \
      --episodes 40 --out datasets/local/so101-sim-pickplace-demos
"""
from __future__ import annotations
import argparse, sys, math, random
from pathlib import Path


def _boot(headless: bool):
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=True).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/local/so101-sim-pickplace-demos")
    ap.add_argument("--episodes", type=int, default=40, help="number of SUCCESSFUL demos to collect")
    ap.add_argument("--max_attempts", type=int, default=80, help="cap on total rollouts (success or not)")
    ap.add_argument("--img", type=int, default=64, help="square d435 frame size")
    ap.add_argument("--grasp_z", type=float, default=0.106)
    ap.add_argument("--obj_x", type=float, default=0.18)
    ap.add_argument("--obj_y", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.03, help="+/- uniform xy jitter (m) around obj_x/y")
    ap.add_argument("--tgt_x", type=float, default=0.22)
    ap.add_argument("--tgt_y", type=float, default=-0.13)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure_scale", action="store_true",
                    help="C1 action-scale probe: run a few full pick->place rollouts with the "
                         "PROVEN scripted controller, accumulate per-arm-joint max|q_des-q_default| "
                         "(rad), write outputs/action_scale.json, and exit. Records NO dataset. "
                         "The measured range sizes _ACTIONS_SCALE_DICT so |action|<=1.")
    args = ap.parse_args()
    random.seed(args.seed)

    app = _boot(not args.gui)
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from lerobot_isaac_env import make_env
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env = make_env(task="pick_and_place", num_envs=1, headless=not args.gui, enable_cameras=True)
    # Disable time_out: the full scripted sequence (~485 steps) exceeds the 300-step episode
    # cap (episode_length_s=10 * 30 Hz), so time_out would TRUNCATE mid-grasp (~step 300, during
    # the seat phase) and auto-reset the die before lift+carry+place ever happen → 0 demos saved.
    # max_episode_length is a read-only property derived from cfg.episode_length_s; bumping the
    # cfg field recomputes it huge. place_termination then fires at carry once the lifted die
    # reaches the bin (verified scripts/_probe_place_term.py, 2026-06-23).
    try:
        env.cfg.episode_length_s = 1.0e6
        print(f"[demos] time_out disabled: max_episode_length -> {getattr(env, 'max_episode_length', None)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[demos] WARN could not disable time_out: {exc}", flush=True)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["source_object"]
    cam = env.scene["d435_camera"]
    ee_idx = int(robot.find_bodies("gripper_link")[0][0])
    arm_ids = list(robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])[0])
    grip_idx = int(robot.find_joints("gripper")[0][0])
    _fixed = bool(getattr(robot, "is_fixed_base", True))
    ee_jac = (ee_idx - 1) if _fixed else ee_idx
    _OFF = 0 if _fixed else 6
    q_default = robot.data.default_joint_pos.clone()
    action_dim = env.action_space.shape[-1]
    # C1 action-scale probe: per-arm-joint running max |q_des - q_default| (rad), across the
    # full scripted pick->place. Sizes _ACTIONS_SCALE_DICT so the demo/scripted/actor actions
    # all fall in [-1,1] (the residual clamp then becomes a no-op — the ee-descent fix).
    scale_max = {int(jid): 0.0 for jid in arm_ids}
    # Per-joint action scale (C1 fix): the recorded `action` label must use the SAME per-joint
    # scale the env applies, so it stays in [-1,1] and the tanh actor / residual clamp can
    # reproduce it. Defaults to 0.5 everywhere when LEROBOT_ISAAC_ACTION_SCALE_JSON is unset
    # (historical behaviour) — so the --measure_scale run itself is unaffected.
    from lerobot_isaac_env.so101_env_cfg import load_action_scale_dict
    _scale_by_name = load_action_scale_dict()
    arm_scales = [float(_scale_by_name.get(robot.data.joint_names[jid], 0.5)) for jid in arm_ids]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device)
    GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
    GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
    # Release-open amount. Full GRIP_OPEN (1.0) spreads the fingers into the cup wall ->
    # servo stalls -> joint_pos never crosses the released threshold (released=False finger-jam,
    # narrow-cup demo-gen 2026-06-24). A PARTIAL open drops the 16mm die without ramming the
    # wall, yet must still land joint_pos > GRIPPER_OPEN_THRESH (default 0.0) to read as released.
    PART_OPEN = float(os.environ.get("LEROBOT_ISAAC_PLACE_PART_OPEN", str(GRIP_OPEN)))
    # Carry height. Raised from 0.17 so the held die (hangs ~0.096 below gripper_link) clears the
    # ~7 cm cup rim during the lateral carry (die_z ≈ z_high - 0.096; 0.22 -> die ~0.10 > 0.07).
    # Capped by SO-101 vertical reach at the cup radius — verified by the demo-gen maxz log.
    z_high = float(os.environ.get("LEROBOT_ISAAC_CARRY_Z", "0.19"))
    # DAgger-style corrective demos: when >0, EXECUTE a noise-perturbed action (drifts the
    # arm off the clean scripted manifold) but RECORD the clean IK expert action as the label.
    # The IK re-aims from the perturbed state each step, so successful rollouts contain
    # recovery trajectories — the corrective data plain BC lacks (its 0/20 closed-loop failure
    # is compounding error / off-manifold drift). Noise is applied only on non-grasp-critical
    # phases (approach/lift/carry/lower), NOT during close/seat/hold/release (would wreck grasp).
    _DAGGER_NOISE = float(os.environ.get("LEROBOT_ISAAC_DEMO_ACTION_NOISE", "0.0"))

    # LeRobotDataset features — match the env d435 obs (3,H,W) + 13-dim state + 6 action.
    # state = joint_pos_rel[6] + object_pose[7] — EXACTLY the vector the sheeprl wrapper
    # packs (isaac_env.py:355-386: joint_pos_rel ++ object_pose) when
    # LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1. The old (12,) = joint_pos++joint_vel did NOT
    # match the run's obs, so DreamerFD seeding silently shape-failed (vet 2026-06-20).
    feats = {
        "observation.state": {"dtype": "float32", "shape": (13,), "names": None},
        "observation.images.d435_rgb": {"dtype": "image", "shape": (3, args.img, args.img),
                                        "names": ["channels", "height", "width"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    out_dir = Path(args.out)
    ds = None
    rew_dir = None
    if not args.measure_scale:
        if out_dir.exists():
            print(f"[demos] ERROR: {out_dir} exists — remove it first (LeRobotDataset.create won't overwrite).", flush=True)
            os._exit(1)
        ds = LeRobotDataset.create(repo_id=f"local/{out_dir.name}", root=str(out_dir), fps=30, features=feats)
        rew_dir = out_dir / "meta" / "demo_rewards"  # per-episode env-reward sidecar (.npy)

    frames: list[dict] = []
    rewards_buf: list[float] = []  # per-step env reward, parallel to frames (sidecar, not a LeRobot feature)
    # place_termination (success_radius, XY-only) fires the instant the die is carried into
    # the bin and Isaac AUTO-RESETS the die that same step — so the old post-hoc
    # `die-to-target < r` check (read AFTER the rollout) saw the reset spawn pose and ALWAYS
    # failed once the env became episodic + place-terminated. The terminated verdict IS the
    # RLVR success signal; capture it at the step and stop (don't record post-reset garbage).
    _acc = {"maxz": -9.0}  # max object-z reached this rollout (post-hoc lift check)

    def grab_frame(action_vec):
        # state = joint_pos_rel[6] ++ object_pose[7], byte-for-byte matching the sheeprl
        # wrapper's state assembly (isaac_env.py:355-386):
        #   joint_pos_rel = joint_pos - default_joint_pos  (= mdp.joint_pos_rel)
        #   object_pose   = cat(root_pos_w, root_quat_w)    (= observations.object_pose, world frame)
        jp_rel = robot.data.joint_pos[0] - q_default[0]                              # (6,)
        obj_pose = torch.cat([obj.data.root_pos_w[0], obj.data.root_quat_w[0]])       # (7,) pos++quat, world
        st = torch.cat([jp_rel, obj_pose]).detach().cpu().numpy().astype("float32")  # (13,)
        rgb = cam.data.output["rgb"][0]                      # (480,640,3) uint8
        rgb = rgb[..., :3].permute(2, 0, 1).float().unsqueeze(0)  # (1,3,480,640)
        rgb = F.interpolate(rgb, size=(args.img, args.img), mode="bilinear", align_corners=False)
        rgb = rgb[0].clamp(0, 255).to(torch.uint8).cpu().numpy()  # (3,img,img)
        frames.append({"observation.state": st,
                       "observation.images.d435_rgb": rgb,
                       "action": np.asarray(action_vec, dtype="float32"),
                       "task": "pick and place the die in the bin"})

    def step_to(target_b, grip, n, quat, grip_end=None, record=True, noise=True):
        ik.reset()
        cmd = torch.tensor([list(target_b) + list(quat)], device=device, dtype=torch.float32)
        for s in range(n):
            g = grip if grip_end is None else grip + (grip_end - grip) * (s / max(1, n - 1))
            rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
            pos_b, quat_b = subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, ee_idx, :], robot.data.body_quat_w[:, ee_idx, :])
            ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
            jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :6, [_OFF + j for j in arm_ids]]
            q_des = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids])
            action = torch.zeros((1, action_dim), device=device)
            for k, jid in enumerate(arm_ids):
                action[0, jid] = (q_des[0, k] - q_default[0, jid]) / arm_scales[k]
            action[0, grip_idx] = g
            if args.measure_scale:
                for k, jid in enumerate(arm_ids):
                    d = abs(float(q_des[0, k] - q_default[0, jid]))
                    if d > scale_max[int(jid)]:
                        scale_max[int(jid)] = d
            if record:
                grab_frame(action[0].detach().cpu().numpy())  # LABEL = clean IK expert action
            # DAgger: execute a perturbed action (drift off-manifold) but keep the clean label.
            action_exec = action
            if _DAGGER_NOISE > 0.0 and noise:
                pert = action.clone()
                for jid in arm_ids:
                    pert[0, jid] = action[0, jid] + torch.randn((), device=device) * _DAGGER_NOISE
                action_exec = pert
            out = env.step(action_exec)
            _term = out[2]
            if (_term[0].item() if torch.is_tensor(_term) else _term):
                _acc["terminated"] = True  # env place_termination fired (gate ON in eval-check mode)
            _acc["maxz"] = max(_acc["maxz"], float(obj.data.root_pos_w[0, 2]))
            if record:
                rew = out[1]  # gym 5-tuple: (obs, reward, terminated, truncated, info)
                try:
                    rew = float(np.asarray(rew).reshape(-1)[0])
                except Exception:
                    rew = float(rew)
                rewards_buf.append(rew)
            # NO early break: place_termination is SUPPRESSED during demo-gen (launch with
            # LEROBOT_ISAAC_PLACE_REST_Z=-1 so the real-place gate never fires -> no mid-sequence
            # auto-reset). The FULL scripted sequence runs incl the descend+RELEASE, and success
            # is judged post-hoc in rollout(). This captures the gripper reopen the recorded human
            # demos have (validation 2026-06-23 flagged the old break-at-success cut the release).

    def rollout(ox, oy):
        """Reset, jitter die to (ox,oy), settle, full pick->place. Returns SUCCESS."""
        frames.clear()
        rewards_buf.clear()
        _acc["maxz"] = -9.0
        _acc["terminated"] = False
        env.reset()
        # teleport die to jittered xy (keep spawn z + identity rot)
        root = obj.data.root_state_w.clone()
        root[0, 0], root[0, 1] = ox, oy
        obj.write_root_state_to_sim(root)
        s = torch.zeros((1, action_dim), device=device); s[0, grip_idx] = GRIP_OPEN
        for _ in range(30):
            env.step(s)                                       # settle (not recorded)
        op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
        gx, gy = float(op0[0]), float(op0[1])
        q = GRASP_QUAT
        step_to([gx, gy, z_high], GRIP_OPEN, 50, q)                       # approach-above: noise OK
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 90, q, noise=False)    # descend to grasp: precise
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 30, q, noise=False)    # settle: precise
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 80, q, grip_end=GRIP_CLOSE, noise=False)  # close
        step_to([gx, gy, args.grasp_z], GRIP_CLOSE, 25, q, noise=False)   # hold grip
        step_to([gx, gy, z_high], GRIP_CLOSE, 60, q)                      # lift: noise OK (recovery)
        step_to([args.tgt_x, args.tgt_y, z_high], GRIP_CLOSE, 60, q)      # carry: noise OK (recovery)
        step_to([args.tgt_x, args.tgt_y, 0.06], GRIP_CLOSE, 40, q)        # lower over cup: noise OK
        # GENTLE release: slow ramp CLOSE->OPEN (50 steps) to avoid the jaw impulsively
        # ejecting the die forward (~5cm) as it opens (validation 2026-06-23 saw die land
        # ~0.052 past bin center with an instant open).
        step_to([args.tgt_x, args.tgt_y, 0.06], GRIP_CLOSE, 50, q, grip_end=PART_OPEN, noise=False)  # release
        # POST-HOC SUCCESS (real place): die ended in the bin XY (success_radius) AND was lifted
        # at some point (max die-z > rest+lift_margin = 0.07) AND is now RESTING low (< 0.04 =
        # lowered/released, not carried aloft). The full sequence already released the gripper.
        dp = obj.data.root_pos_w[0].detach().cpu().numpy()
        xy = float(((dp[0] - args.tgt_x) ** 2 + (dp[1] - args.tgt_y) ** 2) ** 0.5)
        # Mirror the env's is_placed() exactly so demos "succeed" by the SAME rule the env trains on:
        #   lifted (max die-z > rest+margin) AND resting low AND released (gripper open) AND in radius.
        _lift_thr = float(os.environ.get("LEROBOT_ISAAC_OBJECT_Z", "0.05")) + 0.02  # rest_height + lift_margin
        _radius = float(os.environ.get("LEROBOT_ISAAC_PLACE_SUCCESS_RADIUS", "0.05"))  # match cup / place_termination
        was_lifted = _acc["maxz"] > _lift_thr
        # Post-hoc resting threshold is DECOUPLED from LEROBOT_ISAAC_PLACE_REST_Z: the latter is
        # set to -1 at demo-gen to suppress the ENV place_termination (no mid-sequence auto-reset),
        # but the script must still validate the die actually settled low. Reusing PLACE_REST_Z=-1
        # here made `resting` always False -> every demo SKIP even though die_z ~0.01 (artifact found
        # in the PART_OPEN sweep, 2026-06-24). Use a separate knob (default 0.04 = the real bin-rest).
        resting = float(dp[2]) < float(os.environ.get("LEROBOT_ISAAC_DEMO_REST_Z", "0.04"))
        grip_q = float(robot.data.joint_pos[0, grip_idx])
        released = grip_q > float(os.environ.get("LEROBOT_ISAAC_GRIPPER_OPEN_THRESH", "0.0"))
        ok = bool(xy < _radius and was_lifted and resting and released)
        print(f"[demos]   post-hoc: die_final=({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f}) xy_to_bin={xy:.4f} "
              f"maxz={_acc['maxz']:.4f} grip_q={grip_q:.4f} lifted={was_lifted} resting={resting} released={released} "
              f"ENV_TERMINATED={_acc.get('terminated', False)} -> {'OK' if ok else 'SKIP'}", flush=True)
        return ok

    if args.measure_scale:
        # A few clean rollouts to capture the full per-joint range (jitter varies the reach).
        for _ in range(3):
            ox = args.obj_x + random.uniform(-args.jitter, args.jitter)
            oy = args.obj_y + random.uniform(-args.jitter, args.jitter)
            rollout(ox, oy)
        _names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        _id2name = {int(jid): _names[k] for k, jid in enumerate(arm_ids)}
        _SAFETY = 1.15  # scale = max_delta * safety so worst-case |action| ~= 1/safety < 1
        per_joint = {}
        clamp_innocent = True
        for jid, mx in scale_max.items():
            nm = _id2name.get(jid, f"joint_{jid}")
            per_joint[nm] = {"max_abs_delta_rad": round(mx, 4),
                             "recommended_scale": round(max(mx * _SAFETY, 0.1), 4)}
            if mx > 0.5:
                clamp_innocent = False
        summary = {
            "per_joint": per_joint,
            "gripper_scale_unchanged": True,
            "clamp_innocent": clamp_innocent,  # True => all |delta|<=0.5, clamp never bit, Option A unneeded
            "safety": _SAFETY,
            "note": "set _ACTIONS_SCALE_DICT[arm joints] to recommended_scale so |action|<=1; keep gripper",
        }
        import json as _json
        Path("outputs").mkdir(exist_ok=True)
        with open("outputs/action_scale.json", "w") as f:
            _json.dump(summary, f, indent=2)
        print(f"[measure] wrote outputs/action_scale.json: {summary}", flush=True)
        sys.stdout.flush()
        os._exit(0)

    saved, attempts = 0, 0
    while saved < args.episodes and attempts < args.max_attempts:
        attempts += 1
        ox = args.obj_x + random.uniform(-args.jitter, args.jitter)
        oy = args.obj_y + random.uniform(-args.jitter, args.jitter)
        ok = rollout(ox, oy)
        if ok:
            ep_rewards = np.asarray(rewards_buf, dtype="float32").copy()  # before add_frame mutates
            for fr in frames:
                ds.add_frame(fr)
            ds.save_episode()
            # Reward SIDECAR: lerobot 0.5.1 can't store a (1,) reward feature
            # (save crashes), so per-step env reward is written next to the dataset.
            # demo_buffer.load_sim_demos reads it; replaces the reward-0 default that
            # poisoned the reward model in warmstart-v1.
            rew_dir.mkdir(parents=True, exist_ok=True)
            np.save(rew_dir / f"ep_{saved:04d}.npy", ep_rewards)
            saved += 1
            print(f"[demos] SAVED {saved}/{args.episodes} (attempt {attempts}, obj=({ox:.3f},{oy:.3f}), "
                  f"{len(frames)} frames, reward sum={float(ep_rewards.sum()):.2f} last={float(ep_rewards[-1]):.3f})", flush=True)
        else:
            print(f"[demos] skip fail (attempt {attempts}, obj=({ox:.3f},{oy:.3f}))", flush=True)

    if saved > 0:
        ds.finalize()  # flush episode metadata (episodes.parquet) + final stats
        print(f"[demos] finalized dataset metadata", flush=True)
    print(f"[demos] DONE: {saved} demos in {out_dir} ({attempts} attempts)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
