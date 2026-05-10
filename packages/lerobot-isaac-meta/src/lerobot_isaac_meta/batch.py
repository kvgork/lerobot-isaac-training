"""batch.py — Sequential multi-run training + automatic compare.

Reads a YAML batch description (see :mod:`batch_config`), trains each run by
shelling out to ``lerobot-isaac-train``, snapshots the workspace state after
every successful run, and finally renders an N-way (or 2-way) static HTML
compare report via the dashboard package.

CLI
---
::

    lerobot-isaac-batch --config batch.yaml [--workspace .] [--dry_run]

Programmatic use
----------------
::

    from lerobot_isaac_meta.batch import run_batch
    from lerobot_isaac_meta.batch_config import load_batch_config

    cfg = load_batch_config("batch.yaml")
    result = run_batch(cfg, workspace_root=Path("."))
    print(result.compare_report)

Failure handling
----------------
* ``on_failure: continue`` (default) — failed runs are recorded and skipped from
  the compare step; remaining runs proceed.
* ``on_failure: abort`` — first non-zero exit aborts the batch; no compare.

Coupling
--------
* Adapter — invoked via ``subprocess.run([sys.executable, "-m",
  "lerobot_isaac_adapters.train", ...])``.
* Dashboard — soft-imported at compare time.  When the dashboard is missing
  the batch still runs, but the compare step prints a warning and is skipped.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from lerobot_isaac_meta.batch_config import (
    BatchConfig,
    BatchConfigError,
    RunSpec,
    load_batch_config,
)

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of a single run inside a batch."""

    run_id: str
    target_arch: str
    exit_code: int
    snapshot_id: str | None = None
    snapshot_path: Path | None = None
    cmd: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregate outcome of an entire batch."""

    batch_id: str
    runs: list[RunResult]
    compare_report: Path | None = None
    aborted: bool = False

    @property
    def successful_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.exit_code == 0]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _build_train_command(
    run: RunSpec, batch: BatchConfig, *, dry_run: bool
) -> list[str]:
    """Build the ``python -m lerobot_isaac_adapters.train ...`` command for a run."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "lerobot_isaac_adapters.train",
        "--target_arch",
        run.target_arch,
        "--dataset",
        batch.resolved_dataset(run),
    ]
    if run.config:
        cmd += ["--config", run.config]
    if run.steps is not None:
        cmd += ["--steps", str(run.steps)]
    if run.batch_size is not None:
        cmd += ["--batch_size", str(run.batch_size)]
    if run.lr is not None:
        cmd += ["--lr", str(run.lr)]
    if run.seed is not None:
        cmd += ["--seed", str(run.seed)]

    output_dir = run.output_dir or f"{batch.output_root}/{batch.batch_id}/{run.id}"
    cmd += ["--output_dir", output_dir]

    if dry_run:
        cmd.append("--dry_run")

    if run.extra_args:
        cmd.append("--")
        cmd.extend(run.extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Snapshot + compare integration (soft-imported)
# ---------------------------------------------------------------------------


def _snapshot_run(
    workspace_root: Path, batch_id: str, run: RunSpec
) -> tuple[str | None, Path | None]:
    """Snapshot the current workspace state, tagging it with the run.

    Returns ``(snapshot_id, snapshot_path)`` or ``(None, None)`` on failure.
    """
    try:
        from lerobot_isaac_dashboard.snapshots import save_snapshot
    except ImportError as exc:
        logger.warning("Cannot snapshot — lerobot-isaac-dashboard missing: %s", exc)
        return (None, None)

    snapshot_id = f"{batch_id}-{run.id}"
    label = run.label or f"{run.target_arch}/{run.id}"
    try:
        path = save_snapshot(
            workspace_root=workspace_root,
            label=label,
            snapshot_id=snapshot_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("snapshot failed for run=%s: %s", run.id, exc)
        return (None, None)
    return (snapshot_id, path)


def _export_compare(
    workspace_root: Path,
    batch: BatchConfig,
    snapshot_ids: list[str],
) -> Path | None:
    """Render the N-way / 2-way compare HTML report."""
    try:
        from lerobot_isaac_dashboard.compare import export_compare_report
    except ImportError as exc:
        logger.warning(
            "Cannot render compare report — lerobot-isaac-dashboard missing: %s", exc
        )
        return None

    if len(snapshot_ids) < 2:
        logger.info(
            "Skipping compare: need ≥ 2 successful snapshots (have %d)",
            len(snapshot_ids),
        )
        return None

    output_dir = (
        Path(batch.compare.output_dir)
        if batch.compare.output_dir
        else workspace_root / "outputs" / "reports" / f"compare-{batch.batch_id}"
    )

    try:
        return export_compare_report(
            workspace_root=workspace_root,
            snapshot_ids=snapshot_ids,
            mode=batch.compare.mode,  # type: ignore[arg-type]
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compare export failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_batch(
    cfg: BatchConfig,
    *,
    workspace_root: Path,
    dry_run: bool = False,
    train_runner: callable | None = None,
) -> BatchResult:
    """Execute every run in ``cfg`` sequentially, then render compare.

    Parameters
    ----------
    cfg:
        Validated batch configuration.
    workspace_root:
        Workspace root passed to snapshot / compare APIs.
    dry_run:
        Forward ``--dry_run`` to every train invocation; skip snapshot + compare.
    train_runner:
        Callable mirroring :func:`subprocess.run`.  Resolved at call time so
        tests can monkey-patch ``batch.subprocess.run``.  Must accept
        ``(cmd, check)`` kwargs and return an object with ``returncode``.
    """
    if train_runner is None:
        train_runner = subprocess.run
    workspace_root = Path(workspace_root).resolve()
    results: list[RunResult] = []
    successful_snapshots: list[str] = []
    aborted = False

    for run in cfg.runs:
        cmd = _build_train_command(run, cfg, dry_run=dry_run)
        logger.info("[%s] %s", run.id, " ".join(cmd))
        print(f"\n=== batch[{cfg.batch_id}] running {run.id} ({run.target_arch}) ===")
        print("    " + " ".join(cmd))

        try:
            proc = train_runner(cmd, check=False)
            rc = int(proc.returncode)
        except Exception as exc:  # noqa: BLE001
            results.append(
                RunResult(
                    run_id=run.id,
                    target_arch=run.target_arch,
                    exit_code=-1,
                    cmd=cmd,
                    error=str(exc),
                )
            )
            if cfg.on_failure == "abort":
                aborted = True
                break
            continue

        snapshot_id: str | None = None
        snapshot_path: Path | None = None
        if rc == 0 and not dry_run:
            snapshot_id, snapshot_path = _snapshot_run(
                workspace_root, cfg.batch_id, run
            )
            if snapshot_id:
                successful_snapshots.append(snapshot_id)

        results.append(
            RunResult(
                run_id=run.id,
                target_arch=run.target_arch,
                exit_code=rc,
                snapshot_id=snapshot_id,
                snapshot_path=snapshot_path,
                cmd=cmd,
            )
        )

        if rc != 0 and cfg.on_failure == "abort":
            aborted = True
            break

    compare_report: Path | None = None
    if not dry_run and not aborted and cfg.compare.enabled:
        compare_report = _export_compare(workspace_root, cfg, successful_snapshots)

    return BatchResult(
        batch_id=cfg.batch_id,
        runs=results,
        compare_report=compare_report,
        aborted=aborted,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lerobot-isaac-batch",
        description=(
            "Train multiple lerobot-isaac models sequentially on the same dataset, "
            "then render an automatic compare report."
        ),
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a batch YAML file (see batch_config schema).",
    )
    p.add_argument(
        "--workspace",
        default=".",
        metavar="PATH",
        help="Workspace root used for snapshot + compare paths. Default: %(default)s.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Forward --dry_run to every run; skip snapshot + compare.",
    )
    return p


def _print_summary(result: BatchResult) -> None:
    print(f"\n=== batch[{result.batch_id}] summary ===")
    for r in result.runs:
        status = "ok" if r.exit_code == 0 else f"fail ({r.exit_code})"
        snap = f"  snapshot={r.snapshot_id}" if r.snapshot_id else ""
        print(f"  - {r.run_id:<20} arch={r.target_arch:<16} {status}{snap}")
    if result.aborted:
        print("  [aborted on first failure]")
    if result.compare_report:
        print(f"\ncompare report: {result.compare_report}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_batch_config(args.config)
    except (FileNotFoundError, BatchConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workspace_root = Path(args.workspace).resolve()
    result = run_batch(cfg, workspace_root=workspace_root, dry_run=args.dry_run)
    _print_summary(result)

    failed = [r for r in result.runs if r.exit_code != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
