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


def _cmd_train(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac train: not yet wired — see Phase 2 "
        "(packages/lerobot-isaac-adapters).\n"
        "When implemented: python -m lerobot_isaac_adapters.train --target_arch <arch> ..."
    )
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    """Delegate to lerobot_isaac_recorder.cli.main with all forwarded args."""
    try:
        from lerobot_isaac_recorder.cli import main as recorder_main
    except ImportError as exc:
        print(
            f"Error: cannot import lerobot_isaac_recorder.cli: {exc}\n"
            "Ensure lerobot-isaac-recorder is installed: "
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


def _cmd_dr_replay(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac dr-replay: not yet wired — see Phase 4 "
        "(packages/lerobot-isaac-synthetic).\n"
        "When implemented: python -m lerobot_isaac_synthetic.isaac_dr.replay_runner ..."
    )
    return 0


def _cmd_mimicgen_augment(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac mimicgen-augment: not yet wired — see Phase 4b (deferred).\n"
        "This is the MimicGen bridge path. It requires MuJoCo/robosuite "
        "and the lerobot_mimicgen_bridge skill.\n"
        "Skill location: /home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/"
    )
    return 0


def _cmd_quality_filter(args: argparse.Namespace) -> int:
    """Run quality filtering via lerobot_isaac_adapters.quality.apply_quality_filter.

    Dispatches to the SAL+TED quality skill bridge.
    Skill path: /home/koen/tools/claude_code/skills/lerobot_dataset_quality/
    """
    import os
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
        print(f"  skill path       : /home/koen/tools/claude_code/skills/lerobot_dataset_quality/")
        print(f"  would invoke     : lerobot_isaac_adapters.quality.apply_quality_filter(")
        print(f"      dataset_path={str(dataset)!r},")
        print(f"      sal_threshold={args.sal_threshold},")
        print(f"      ted_threshold={args.ted_threshold},")
        print(f"      min_episode_length={args.min_episode_length},")
        print(f"      output_path={str(output)!r},")
        print(f"      dry_run=True,")
        print(f"  )")
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

_SUBCOMMANDS: dict[str, tuple[callable, str]] = {
    "train": (_cmd_train, "train a policy or world model (Phase 2+)"),
    "record": (
        _cmd_record,
        "record SO-101 teleop data (D435 + dual-write Parquet+HDF5 via lerobot-isaac-recorder)",
    ),
    "dr-replay": (_cmd_dr_replay, "replay with Isaac Lab domain randomization (Phase 4+)"),
    "mimicgen-augment": (
        _cmd_mimicgen_augment,
        "augment dataset via MimicGen (Phase 4b, deferred)",
    ),
    "quality-filter": (
        _cmd_quality_filter,
        "filter low-quality episodes from a LeRobotDataset using SAL+TED metrics",
    ),
}


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
        help=(
            "Output path for the filtered dataset. "
            "Defaults to <dataset>_filtered."
        ),
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
        # record: pass-through all remaining args to recorder CLI
        if name == "record":
            sub.add_argument(
                "recorder_args",
                nargs=argparse.REMAINDER,
                help="Args forwarded to lerobot-isaac-record (e.g. --repo-id ... --num-episodes N --dry-run)",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
