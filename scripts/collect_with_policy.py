#!/usr/bin/env python3
"""Record new SO-101 episodes with a trained policy driving the arm.

This is the collection step of the self-improving loop: round N's checkpoint drives the
arm, the episodes are written as a LeRobotDataset, round N+1 trains on them (successes
only), and its checkpoint drives the next round.

    round 0 policy ──▶ collect ──▶ label ──▶ train ──▶ round 1 policy ──▶ ...

It exists because the two halves already existed separately and nothing joined them:
``robot-data-runner`` runs a policy on the arm but never writes a dataset, and
``robot-data-recorder`` writes datasets but only from a human leader arm. This is the
glue, and it lives in the workspace because it spans two sibling packages.

FRAME COHERENCE
---------------
``RecordingSession.record_episode`` calls ``camera.read_frame()`` and
``teleop.read_state()`` and only then ``teleop.read_action()``. If the action source
re-read the camera it would infer on a DIFFERENT frame than the one recorded, so the
dataset would pair an image with an action computed from a later image. That is the
buffer-action != executed-action failure this project has already hit twice. The camera
is therefore wrapped in a cache: the action source uses the exact frame just recorded.

SAFETY
------
* a connected LEADER always overrides the policy (enforced in SO101Teleop)
* ``--max-relative-target`` rate-limits every write server-side (default 2.0)
* non-finite or incomplete actions are refused before reaching the motors
* refuses to run without a TTY — an operator must be present and able to cut power
* there is NO software e-stop; the only reliable stop is cutting power

FLYWHEEL CAVEAT
---------------
Training round N+1 on everything round N collected reinforces round N's mistakes. Use
``--successes_only`` downstream, which reads ``meta/episode_labels.json``. Episode
success is currently labelled by the operator keypress the recorder already implements;
an automated verifier needs object pose, which this rig does not yet have.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in ("src/robot-data-runner/src", "src/robot-data-recorder/src"):
    sys.path.insert(0, str(REPO / _p))


class _FrameCache:
    """Transparent camera proxy that remembers the frame it just handed out."""

    def __init__(self, camera):
        self._camera = camera
        self.last = None

    def read_frame(self):
        self.last = self._camera.read_frame()
        return self.last

    def __getattr__(self, name):
        return getattr(self._camera, name)


def build_policy_action_source(policy_path, dataset_root, teleop, frames, device=None):
    """Return a zero-arg callable producing one action dict per control step."""
    from robot_data_runner.mappers import action_to_robot_dict, obs_to_policy_input
    from robot_data_runner.policy_loader import load_policy

    loaded = load_policy(policy_path, dataset_root=dataset_root, device=device)
    motor_names = list(teleop._robot.bus.motors)

    def _source() -> dict:
        frame = frames.last
        if frame is None:
            raise RuntimeError(
                "no camera frame cached yet — the action source ran before "
                "camera.read_frame(), so the policy would infer on a stale image"
            )
        state = teleop.read_state()
        obs = {
            "observation.state": state["joint_pos"],
            "observation.images.overhead": frame["rgb"],
        }
        tensor_obs = obs_to_policy_input(obs, loaded, device=loaded.device)
        action = loaded.policy.select_action(tensor_obs)
        return action_to_robot_dict(action, motor_names)

    return _source, loaded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="collect_with_policy.py", description=__doc__.split("\n")[0]
    )
    p.add_argument(
        "--policy-path",
        required=True,
        type=Path,
        help="checkpoint driving this round (round 0: trained on the last dataset)",
    )
    p.add_argument(
        "--repo-id", required=True, help="output dataset id, e.g. local/so101-round1"
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO / "datasets/local/so101-pickplace-new",
        help="dataset the policy was TRAINED on, for norm stats / shapes",
    )
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--arm-port", default="/dev/ttyACM0")
    p.add_argument("--camera-serial", default=None)
    p.add_argument("--max-relative-target", type=float, default=2.0)
    p.add_argument(
        "--max-steps", type=int, default=900, help="per-episode cap (30 s at 30 Hz)"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.policy_path.exists():
        print(
            f"[collect] FATAL: checkpoint not found: {args.policy_path}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("[collect] DRY RUN — resolved plan:")
        for k, v in vars(args).items():
            print(f"  {k:22s} {v}")
        print(
            "\n[collect] would drive the arm with the policy above and write episodes to"
        )
        print(f"  {args.repo_id}")
        return 0

    if not sys.stdin.isatty():
        print(
            "[collect] FATAL: refusing to run without a TTY. This drives a real arm and "
            "there is no software e-stop; an operator must be present.",
            file=sys.stderr,
        )
        return 3

    from robot_data_recorder.config import RecordingConfig
    from robot_data_recorder.d435 import make_d435
    from robot_data_recorder.dual_writer import DualWriter
    from robot_data_recorder.recorder import RecordingSession
    from robot_data_recorder.so101_teleop import SO101Teleop

    cfg = RecordingConfig(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        arm_port=args.arm_port,
        leader_port=None,
        camera_serial=args.camera_serial,
        max_steps=args.max_steps,
    )
    camera = _FrameCache(
        make_d435(
            serial=cfg.camera_serial,
            resolution=cfg.resolution,
            fps=cfg.fps,
            enable_depth=cfg.enable_depth,
        )
    )
    teleop = SO101Teleop(
        arm_port=cfg.arm_port,
        leader_port=None,
        max_relative_target=args.max_relative_target,
    )
    writer = DualWriter(cfg)

    n_recorded = n_success = 0
    with RecordingSession(cfg, camera=camera, teleop=teleop, writer=writer) as session:
        source, loaded = build_policy_action_source(
            args.policy_path, args.dataset_root, teleop, camera, device=args.device
        )
        teleop.set_action_source(source)
        print(f"[collect] policy {args.policy_path} on {loaded.device}")
        print(
            f"[collect] clamp {args.max_relative_target} deg/step — NO software e-stop; "
            f"stand by the power switch"
        )

        for ep in range(cfg.num_episodes):
            print(
                f"\n[collect] episode {ep + 1}/{cfg.num_episodes} — "
                f"place the cup and die in a NEW position, then press SPACE"
            )
            if not session.wait_for_start(ep, cfg.num_episodes):
                print("[collect] aborted by operator.")
                break
            buf = session.record_episode(ep)
            session.save_episode(buf)
            n_recorded += 1
            n_success += int(bool(getattr(buf, "success", False)))
            print(
                f"[collect] episode {ep} saved (success={getattr(buf, 'success', '?')})"
            )

    print(f"\n[collect] {n_recorded} recorded, {n_success} marked success")
    print("[collect] train the next round on successes only:")
    print(f"  lerobot-isaac-train --dataset <root>/{args.repo_id} --successes_only ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
