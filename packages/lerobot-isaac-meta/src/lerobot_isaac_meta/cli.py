"""lerobot-isaac CLI — top-level argparse entrypoint.

Subcommands:
    train            — train a policy or world model (Phase 2)
    record           — record SO-101 teleop data (Phase 2)
    dr-replay        — replay with Isaac Lab domain randomization (Phase 4)
    mimicgen-augment — augment dataset via MimicGen (Phase 4b, deferred)
    quality-filter   — filter low-quality episodes from a LeRobotDataset (Phase A2)

Each subcommand prints a "not yet wired" message until its phase is complete,
except quality-filter which is fully implemented in Bundle A.

Plan reference: §13.1 Bundle A, deliverable A5
Last-updated: 2026-05-07
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Subcommand handlers — stubs until their phases are implemented
# ---------------------------------------------------------------------------


def _cmd_train(argv: list[str]) -> int:
    """Delegate to lerobot_isaac_adapters.train.main with forwarded argv.

    Takes a raw arg list (not an argparse Namespace) — train args are forwarded
    verbatim. ``main()`` intercepts this subcommand before argparse so that
    leading-dash backend flags (``--target_arch``) are not misparsed
    (argparse.REMAINDER mishandles them, bpo-17050).
    """
    try:
        from lerobot_isaac_adapters.train import main as train_main
    except ImportError as exc:
        print(
            f"Error: cannot import lerobot_isaac_adapters.train: {exc}\n"
            "Ensure lerobot-isaac-adapters is installed.",
            file=sys.stderr,
        )
        return 1
    if argv and argv[0] == "--":
        argv = argv[1:]
    # adapters train.main() calls sys.exit(rc); capture the code.
    try:
        train_main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    """Delegate to robot_data_recorder.cli.main with all forwarded args."""
    try:
        from robot_data_recorder.cli import main as recorder_main
    except ImportError as exc:
        print(
            f"Error: cannot import robot_data_recorder.cli: {exc}\n"
            "Ensure robot-data-recorder is installed: "
            "pip install -e packages/lerobot-isaac-recorder",
            file=sys.stderr,
        )
        return 1
    # Forward args.recorder_args (everything after `record`) to the recorder CLI.
    # Strip leading "--" separator if user used it to disambiguate flags.
    forwarded = list(args.recorder_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return recorder_main(forwarded)


def _cmd_dr_replay(argv: list[str]) -> int:
    """Delegate to lerobot_isaac_synthetic.isaac_dr.replay_runner.main.

    Raw-argv passthrough (see _cmd_train for why argparse is bypassed).
    """
    try:
        from lerobot_isaac_synthetic.isaac_dr.replay_runner import main as replay_main
    except ImportError as exc:
        print(
            f"Error: cannot import replay_runner: {exc}\n"
            "Ensure lerobot-isaac-synthetic is installed.",
            file=sys.stderr,
        )
        return 1
    if argv and argv[0] == "--":
        argv = argv[1:]
    try:
        replay_main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _cmd_mimicgen_augment(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac mimicgen-augment: not yet wired — see Phase 4b (deferred).\n"
        "This is the MimicGen bridge path. It requires MuJoCo/robosuite "
        "and the lerobot_mimicgen_bridge skill.\n"
        "Skill location: ${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/"
    )
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    """`lerobot-isaac env smoke` — boot the Isaac Lab SO-101 env and step it.

    Only the ``smoke`` action is supported. With ``--dry-run`` the command
    prints the resolved parameters and exits 0 without importing Isaac Lab
    (so it works on machines without a GPU). Without ``--dry-run`` it boots
    the env; if Isaac Lab is unavailable it returns exit code 2.
    """
    action = getattr(args, "env_action", None)
    if action != "smoke":
        print(
            f"lerobot-isaac env: unknown action {action!r} (only 'smoke' is supported)",
            file=sys.stderr,
        )
        return 2

    cameras = [c.strip() for c in (args.cameras or "").split(",") if c.strip()]
    enable_cameras = bool(cameras)

    if args.dry_run:
        print("lerobot-isaac env smoke — dry-run mode")
        print(f"{'task':<18}: {args.task}")
        print(f"{'cameras':<18}: {', '.join(cameras) if cameras else '(none)'}")
        print(f"{'camera-resolution':<18}: {args.camera_resolution}")
        print(f"{'steps':<18}: {args.steps}")
        print(f"{'enable_cameras':<18}: {enable_cameras}")
        if enable_cameras:
            print(f"{'warm-up':<18}: 30 frames (camera sensors need warm-up)")
        return 0

    # Real run — requires Isaac Lab. Verify availability without importing any
    # lerobot_isaac_env symbol yet (ordering matters — see below).
    try:
        import isaaclab.app  # noqa: F401
    except Exception:
        try:
            import omni.isaac.lab.app  # noqa: F401
        except Exception:
            print(
                "Isaac Lab not installed (neither 'isaaclab' nor "
                "'omni.isaac.lab' importable). Install Isaac Lab or pass "
                "--dry-run. See docs/runbook/01-bootstrap.md.",
                file=sys.stderr,
            )
            return 2

    # CRITICAL ordering: construct the Isaac Sim app BEFORE importing ANY
    # lerobot_isaac_env symbol. `import lerobot_isaac_env[.smoke]` triggers the
    # package's module-level isaaclab/USD imports (so101_env_cfg); if those load
    # before Kit boots, a stale pxr crashes Kit (SIGSEGV, "extension class
    # wrapper ... not created yet"). So AppLauncher is built here, inline, in
    # this heavy-import-free meta module — then the env path is imported.
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    # Kit also inspects sys.argv at construction; the leftover subcommand flags
    # crash it, so strip argv to argv[0] for the launch.
    _saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        simulation_app = AppLauncher(
            headless=True, enable_cameras=enable_cameras
        ).app
    finally:
        sys.argv = _saved_argv

    # Now safe to import the env-construction path — Kit is up.
    from lerobot_isaac_env.smoke import run_env_smoke

    return run_env_smoke(
        task=args.task,
        cameras=cameras,
        camera_resolution=args.camera_resolution,
        steps=args.steps,
        simulation_app=simulation_app,
    )


def _cmd_quality_filter(args: argparse.Namespace) -> int:
    """Run quality filtering via lerobot_isaac_adapters.quality.apply_quality_filter.

    Dispatches to the SAL+TED quality skill bridge.
    Skill path: ${CLAUDE_CODE_ROOT}/skills/lerobot_dataset_quality/
    """
    from pathlib import Path

    dataset = Path(args.dataset)
    output = Path(args.output) if args.output else Path(str(dataset) + "_filtered")

    if args.dry_run:
        print("lerobot-isaac quality-filter — dry-run mode")
        print(f"  dataset          : {dataset}")
        print(f"  output           : {output}")
        print(f"  sal-threshold    : {args.sal_threshold}")
        print(f"  ted-threshold    : {args.ted_threshold}")
        print(f"  min-episode-len  : {args.min_episode_length}")
        print(
            "  skill path       : ${CLAUDE_CODE_ROOT}/skills/lerobot_dataset_quality/"
        )
        print(
            "  would invoke     : lerobot_isaac_adapters.quality.apply_quality_filter("
        )
        print(f"      dataset_path={str(dataset)!r},")
        print(f"      sal_threshold={args.sal_threshold},")
        print(f"      ted_threshold={args.ted_threshold},")
        print(f"      min_episode_length={args.min_episode_length},")
        print(f"      output_path={str(output)!r},")
        print("      dry_run=True,")
        print("  )")
        return 0

    # Lazy import — avoids forcing lerobot_isaac_adapters to be importable at CLI load
    try:
        from lerobot_isaac_adapters.quality import apply_quality_filter
    except ImportError as exc:
        print(
            f"Error: cannot import lerobot_isaac_adapters.quality: {exc}\n"
            "Ensure lerobot-isaac-adapters is installed: pip install -e packages/lerobot-isaac-adapters",
            file=sys.stderr,
        )
        return 1

    result = apply_quality_filter(
        dataset_path=dataset,
        sal_threshold=args.sal_threshold,
        ted_threshold=args.ted_threshold,
        min_episode_length=args.min_episode_length,
        output_path=output,
        dry_run=False,
    )

    if result.success:
        data = result.data or {}
        print(
            f"quality-filter complete: "
            f"kept={data.get('kept', '?')} "
            f"removed={data.get('removed', '?')} "
            f"output={data.get('output_path', str(output))}"
        )
        return 0
    else:
        print(f"quality-filter failed: {result.error}", file=sys.stderr)
        if result.suggestions:
            for s in result.suggestions:
                print(f"  suggestion: {s}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Subcommand registry
# ---------------------------------------------------------------------------

# Passthrough subcommands: argv after the name is forwarded verbatim to a
# sibling entrypoint. Intercepted in main() BEFORE argparse because
# argparse.REMAINDER mishandles leading-dash backend flags (bpo-17050).
_PASSTHROUGH: dict[str, tuple[callable, str]] = {
    "train": (_cmd_train, "train a policy or world model (forwards to adapters.train)"),
    "dr-replay": (
        _cmd_dr_replay,
        "replay with Isaac Lab domain randomization (forwards to synthetic.replay_runner)",
    ),
}

_SUBCOMMANDS: dict[str, tuple[callable, str]] = {
    "record": (
        _cmd_record,
        "record SO-101 teleop data (D435 + dual-write Parquet+HDF5 via robot-data-recorder)",
    ),
    "mimicgen-augment": (
        _cmd_mimicgen_augment,
        "augment dataset via MimicGen (Phase 4b, deferred)",
    ),
    "quality-filter": (
        _cmd_quality_filter,
        "filter low-quality episodes from a LeRobotDataset using SAL+TED metrics",
    ),
    "env": (
        _cmd_env,
        "boot + step the Isaac Lab SO-101 env (smoke test; Bundle C.1)",
    ),
}


def _add_env_args(sub: argparse.ArgumentParser) -> None:
    """Attach `env`-subcommand arguments (currently just the `smoke` action)."""
    sub.add_argument(
        "env_action",
        choices=["smoke"],
        help="env action to run (only 'smoke' is supported).",
    )
    sub.add_argument(
        "--task",
        default="so101_pickplace",
        metavar="NAME",
        help="Task config to boot. Default: %(default)s.",
    )
    sub.add_argument(
        "--cameras",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated camera names to enable (e.g. 'd435' or "
            "'wrist,overhead'). Omit for a no-camera state-only smoke."
        ),
    )
    sub.add_argument(
        "--camera-resolution",
        dest="camera_resolution",
        default="640x480",
        metavar="WxH",
        help="Camera render resolution. Default: %(default)s.",
    )
    sub.add_argument(
        "--steps",
        type=int,
        default=100,
        metavar="N",
        help="Number of env steps to run. Default: %(default)s.",
    )
    sub.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print resolved parameters and exit 0 without booting Isaac Lab.",
    )


def _add_quality_filter_args(sub: argparse.ArgumentParser) -> None:
    """Attach quality-filter-specific arguments to a subparser."""
    sub.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to a LeRobotDataset directory to filter.",
    )
    sub.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=("Output path for the filtered dataset. Defaults to <dataset>_filtered."),
    )
    sub.add_argument(
        "--sal-threshold",
        dest="sal_threshold",
        type=float,
        default=0.2,
        metavar="F",
        help=(
            "Fraction of worst-SAL episodes to remove (0.0–1.0). "
            "0.2 removes the bottom 20%%. Default: %(default)s."
        ),
    )
    sub.add_argument(
        "--ted-threshold",
        dest="ted_threshold",
        type=float,
        default=2.0,
        metavar="F",
        help=(
            "Absolute TED upper bound; episodes above this are additionally removed. "
            "Default: %(default)s."
        ),
    )
    sub.add_argument(
        "--min-episode-length",
        dest="min_episode_length",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Remove episodes shorter than N timesteps unconditionally. "
            "Default: %(default)s."
        ),
    )
    sub.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Print resolved parameters and would-be skill invocation, then exit 0 "
            "without writing any files."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac",
        description=(
            "LeRobot + Isaac Lab training workspace CLI.\n"
            "Workspace: ~/workspaces/lerobot-isaac-training/\n"
            "Status: Bundle A complete. quality-filter subcommand active."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="lerobot-isaac 0.1.0 (Bundle A)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    for name, (handler, help_text) in _SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(func=handler)
        # Attach additional args for quality-filter
        if name == "quality-filter":
            _add_quality_filter_args(sub)
        # env: positional action + smoke flags
        if name == "env":
            _add_env_args(sub)
        # record: pass-through all remaining args to recorder CLI
        if name == "record":
            sub.add_argument(
                "recorder_args",
                nargs=argparse.REMAINDER,
                help="Args forwarded to lerobot-isaac-record (e.g. --repo-id ... --num-episodes N --dry-run)",
            )

    # Register passthrough subcommands for `--help` listing only. Their argv is
    # intercepted in main() before parse_args, so a single REMAINDER positional
    # is enough to document them.
    for name, (_handler, help_text) in _PASSTHROUGH.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("backend_args", nargs=argparse.REMAINDER, help="forwarded verbatim")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Intercept passthrough subcommands before argparse (REMAINDER mishandles
    # leading-dash backend flags). Everything after the name goes to the backend.
    if argv and argv[0] in _PASSTHROUGH:
        handler, _help = _PASSTHROUGH[argv[0]]
        return handler(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
