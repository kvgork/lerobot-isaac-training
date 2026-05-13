"""Tests for lerobot_isaac_meta.batch — runner, command construction, CLI."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock


from lerobot_isaac_meta import batch as batch_mod
from lerobot_isaac_meta.batch import _build_train_command, run_batch
from lerobot_isaac_meta.batch_config import (
    BatchConfig,
    CompareSpec,
    RunSpec,
)


# ---------------------------------------------------------------------------
# _build_train_command
# ---------------------------------------------------------------------------


class TestBuildTrainCommand:
    def _cfg(self, **overrides):
        return BatchConfig(
            batch_id="b",
            dataset="data/default",
            runs=[
                RunSpec(id="r1", target_arch="smolvla", **overrides),
                RunSpec(id="r2", target_arch="le_world_model"),
            ],
        )

    def test_minimal_command(self):
        cfg = self._cfg()
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        assert sys.executable in cmd
        assert "-m" in cmd
        assert "lerobot_isaac_adapters.train" in cmd
        assert "--target_arch" in cmd
        assert "smolvla" in cmd
        assert "--dataset" in cmd
        assert "data/default" in cmd

    def test_per_run_dataset_override(self):
        cfg = self._cfg(dataset="data/override")
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        assert "data/override" in cmd
        assert "data/default" not in cmd

    def test_optional_args_emitted_only_when_set(self):
        cfg = self._cfg(steps=42, batch_size=8, lr=1e-3, seed=7, config="cfg.yaml")
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        assert "--steps" in cmd and "42" in cmd
        assert "--batch_size" in cmd and "8" in cmd
        assert "--lr" in cmd and "0.001" in cmd
        assert "--seed" in cmd and "7" in cmd
        assert "--config" in cmd and "cfg.yaml" in cmd

    def test_default_output_dir_uses_batch_and_run_id(self):
        cfg = self._cfg()
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        idx = cmd.index("--output_dir")
        assert cmd[idx + 1] == "outputs/runs/b/r1"

    def test_explicit_output_dir_respected(self):
        cfg = self._cfg(output_dir="custom/dir")
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        idx = cmd.index("--output_dir")
        assert cmd[idx + 1] == "custom/dir"

    def test_dry_run_flag_appended(self):
        cfg = self._cfg()
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=True)
        assert "--dry_run" in cmd

    def test_extra_args_separator(self):
        cfg = self._cfg(extra_args=["--policy.n_action_steps=100"])
        cmd = _build_train_command(cfg.runs[0], cfg, dry_run=False)
        assert cmd[-2] == "--"
        assert cmd[-1] == "--policy.n_action_steps=100"


# ---------------------------------------------------------------------------
# run_batch — happy paths and failure modes
# ---------------------------------------------------------------------------


def _make_cfg(*, on_failure="continue", compare_enabled=True, compare_mode="nway"):
    return BatchConfig(
        batch_id="bx",
        dataset="data",
        runs=[
            RunSpec(id="a", target_arch="smolvla"),
            RunSpec(id="b", target_arch="le_world_model"),
        ],
        on_failure=on_failure,
        compare=CompareSpec(enabled=compare_enabled, mode=compare_mode),
    )


class TestRunBatch:
    def test_dry_run_skips_snapshot_and_compare(self, tmp_path):
        cfg = _make_cfg()
        calls = []

        def fake_runner(cmd, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(batch_mod, "_snapshot_run") as snap,
            mock.patch.object(batch_mod, "_export_compare") as cmp_,
        ):
            result = run_batch(
                cfg,
                workspace_root=tmp_path,
                dry_run=True,
                train_runner=fake_runner,
            )
            snap.assert_not_called()
            cmp_.assert_not_called()

        assert len(calls) == 2
        assert all(r.exit_code == 0 for r in result.runs)
        assert result.compare_report is None

    def test_all_success_triggers_snapshot_and_compare(self, tmp_path):
        cfg = _make_cfg()

        def fake_runner(cmd, check):
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(batch_mod, "_snapshot_run") as snap,
            mock.patch.object(batch_mod, "_export_compare") as cmp_,
        ):
            snap.side_effect = [
                ("bx-a", tmp_path / "snap-a"),
                ("bx-b", tmp_path / "snap-b"),
            ]
            cmp_.return_value = tmp_path / "report.html"

            result = run_batch(cfg, workspace_root=tmp_path, train_runner=fake_runner)

        assert snap.call_count == 2
        assert cmp_.call_count == 1
        assert cmp_.call_args.args == (tmp_path.resolve(), cfg, ["bx-a", "bx-b"])
        assert result.compare_report == tmp_path / "report.html"
        assert len(result.successful_runs) == 2

    def test_continue_on_failure(self, tmp_path):
        cfg = _make_cfg(on_failure="continue")
        codes = iter([1, 0])

        def fake_runner(cmd, check):
            return SimpleNamespace(returncode=next(codes))

        with (
            mock.patch.object(batch_mod, "_snapshot_run") as snap,
            mock.patch.object(batch_mod, "_export_compare") as cmp_,
        ):
            snap.return_value = ("bx-b", tmp_path / "snap-b")
            cmp_.return_value = None

            result = run_batch(cfg, workspace_root=tmp_path, train_runner=fake_runner)

        assert [r.exit_code for r in result.runs] == [1, 0]
        # Only the second (successful) run is snapshotted.
        assert snap.call_count == 1
        # Compare invoked but with a single snapshot — runner short-circuits.
        cmp_.assert_called_once()
        assert not result.aborted

    def test_abort_on_failure_stops_immediately(self, tmp_path):
        cfg = _make_cfg(on_failure="abort")
        codes = iter([2, 0])

        def fake_runner(cmd, check):
            return SimpleNamespace(returncode=next(codes))

        with (
            mock.patch.object(batch_mod, "_snapshot_run") as snap,
            mock.patch.object(batch_mod, "_export_compare") as cmp_,
        ):
            result = run_batch(cfg, workspace_root=tmp_path, train_runner=fake_runner)

        assert result.aborted is True
        assert len(result.runs) == 1
        assert result.runs[0].exit_code == 2
        snap.assert_not_called()
        cmp_.assert_not_called()

    def test_subprocess_exception_recorded(self, tmp_path):
        cfg = _make_cfg()

        def fake_runner(cmd, check):
            if "smolvla" in cmd:
                raise FileNotFoundError("python missing")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(batch_mod, "_snapshot_run") as snap,
            mock.patch.object(batch_mod, "_export_compare") as cmp_,
        ):
            snap.return_value = ("bx-b", tmp_path / "snap-b")
            cmp_.return_value = None

            result = run_batch(cfg, workspace_root=tmp_path, train_runner=fake_runner)

        assert result.runs[0].exit_code == -1
        assert "python missing" in (result.runs[0].error or "")
        assert result.runs[1].exit_code == 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


class TestCliEntry:
    def test_invalid_config_returns_2(self, tmp_path, capsys):
        rc = batch_mod.main(["--config", str(tmp_path / "missing.yaml")])
        assert rc == 2
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_dry_run_through_cli(self, tmp_path, capsys, monkeypatch):
        cfg_path = tmp_path / "batch.yaml"
        cfg_path.write_text(
            "batch_id: c\n"
            "dataset: ds\n"
            "runs:\n"
            "  - id: a\n"
            "    target_arch: smolvla\n"
            "  - id: b\n"
            "    target_arch: act\n",
            encoding="utf-8",
        )

        recorded: list[list[str]] = []

        def fake_runner(cmd, check):
            recorded.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(batch_mod.subprocess, "run", fake_runner)

        rc = batch_mod.main(
            [
                "--config",
                str(cfg_path),
                "--workspace",
                str(tmp_path),
                "--dry_run",
            ]
        )
        assert rc == 0
        assert len(recorded) == 2
        assert "--dry_run" in recorded[0]
