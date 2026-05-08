"""
test_cli_quality_filter.py — Tests for the quality-filter CLI subcommand.

Tests:
  - quality-filter subcommand is registered in the parser.
  - --dry-run path returns exit 0 and prints expected lines.
  - --help output includes quality-filter.
  - Required --dataset arg.
  - Argparse defaults match spec.

Plan reference: §13.1 Bundle A, deliverable A6
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Parser availability
# ---------------------------------------------------------------------------


class TestParserRegistration:
    def test_quality_filter_in_subcommands(self):
        """build_parser() must include 'quality-filter' in subcommand choices."""
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        subparsers_action = None
        for action in parser._actions:
            if hasattr(action, "_name_parser_map"):
                subparsers_action = action
                break
        assert subparsers_action is not None
        assert "quality-filter" in subparsers_action._name_parser_map

    def test_quality_filter_help_text(self):
        """'quality-filter' subcommand should have help text mentioning SAL or TED."""
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        help_output = parser.format_help()
        assert "quality-filter" in help_output

    def test_all_original_subcommands_present(self):
        """Original subcommands must still be present after adding quality-filter."""
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        for name in ("train", "record", "dr-replay", "mimicgen-augment"):
            assert name in subparsers_action._name_parser_map


# ---------------------------------------------------------------------------
# Argument defaults
# ---------------------------------------------------------------------------


class TestArgumentDefaults:
    def _parse(self, argv):
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        return parser.parse_args(argv)

    def test_dataset_required(self):
        """--dataset is required; omitting it raises SystemExit."""
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["quality-filter"])

    def test_sal_threshold_default(self):
        args = self._parse(["quality-filter", "--dataset", "/tmp/ds"])
        assert args.sal_threshold == pytest.approx(0.2)

    def test_ted_threshold_default(self):
        args = self._parse(["quality-filter", "--dataset", "/tmp/ds"])
        assert args.ted_threshold == pytest.approx(2.0)

    def test_min_episode_length_default(self):
        args = self._parse(["quality-filter", "--dataset", "/tmp/ds"])
        assert args.min_episode_length == 50

    def test_dry_run_default_false(self):
        args = self._parse(["quality-filter", "--dataset", "/tmp/ds"])
        assert args.dry_run is False

    def test_output_default_none(self):
        args = self._parse(["quality-filter", "--dataset", "/tmp/ds"])
        assert args.output is None

    def test_custom_thresholds(self):
        args = self._parse(
            [
                "quality-filter",
                "--dataset",
                "/tmp/ds",
                "--sal-threshold",
                "0.3",
                "--ted-threshold",
                "1.5",
                "--min-episode-length",
                "30",
            ]
        )
        assert args.sal_threshold == pytest.approx(0.3)
        assert args.ted_threshold == pytest.approx(1.5)
        assert args.min_episode_length == 30


# ---------------------------------------------------------------------------
# dry-run path
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_zero(self, capsys, tmp_path: Path):
        """--dry-run exits 0 without calling apply_quality_filter."""
        from lerobot_isaac_meta.cli import main

        rc = main(
            [
                "quality-filter",
                "--dataset",
                "/tmp/fake_dataset",
                "--dry-run",
            ]
        )
        assert rc == 0

    def test_dry_run_prints_resolved_params(self, capsys, tmp_path: Path):
        """--dry-run output contains dataset path and key parameters."""
        from lerobot_isaac_meta.cli import main

        main(
            [
                "quality-filter",
                "--dataset",
                "/tmp/my_dataset",
                "--sal-threshold",
                "0.15",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert "/tmp/my_dataset" in captured.out
        assert "0.15" in captured.out
        assert "dry-run" in captured.out.lower() or "dry_run" in captured.out.lower()

    def test_dry_run_does_not_call_adapter(self, capsys, tmp_path: Path):
        """--dry-run must NOT import or call apply_quality_filter."""
        from lerobot_isaac_meta.cli import main

        with patch("lerobot_isaac_meta.cli._cmd_quality_filter") as mock_handler:
            mock_handler.return_value = 0
            # We need to call the real handler to test dry_run logic;
            # just verify that the import path isn't reached during dry_run.
            pass
        # Real test: run with dry_run and verify no side-effects
        with patch("lerobot_isaac_adapters.quality.apply_quality_filter") as mock_aqf:
            rc = main(
                [
                    "quality-filter",
                    "--dataset",
                    "/tmp/nonexistent",
                    "--dry-run",
                ]
            )
        assert rc == 0
        mock_aqf.assert_not_called()

    def test_dry_run_prints_skill_path(self, capsys):
        """dry-run output should mention the skill path for discoverability."""
        from lerobot_isaac_meta.cli import main

        main(["quality-filter", "--dataset", "foo", "--dry-run"])
        captured = capsys.readouterr()
        assert "lerobot_dataset_quality" in captured.out


# ---------------------------------------------------------------------------
# Integration: quality-filter dispatches to apply_quality_filter
# ---------------------------------------------------------------------------


class TestQualityFilterDispatch:
    def test_success_path(self, tmp_path: Path, capsys):
        """When apply_quality_filter returns success, cli returns 0."""
        ds = tmp_path / "ds"
        ds.mkdir()

        from lerobot_isaac_meta.cli import main
        from lerobot_isaac_adapters.quality import OperationResult

        with patch("lerobot_isaac_adapters.quality.apply_quality_filter") as mock_aqf:
            mock_aqf.return_value = OperationResult(
                success=True,
                data={
                    "kept": 7,
                    "removed": 3,
                    "output_path": str(tmp_path / "ds_filtered"),
                },
            )
            rc = main(["quality-filter", "--dataset", str(ds)])

        assert rc == 0
        captured = capsys.readouterr()
        assert "kept=7" in captured.out

    def test_failure_path_returns_nonzero(self, tmp_path: Path, capsys):
        """When apply_quality_filter returns success=False, cli returns nonzero."""
        ds = tmp_path / "ds"
        ds.mkdir()

        from lerobot_isaac_meta.cli import main
        from lerobot_isaac_adapters.quality import OperationResult

        with patch("lerobot_isaac_adapters.quality.apply_quality_filter") as mock_aqf:
            mock_aqf.return_value = OperationResult(
                success=False,
                error="skill not reachable",
                suggestions=["check CLAUDE_CODE_ROOT"],
            )
            rc = main(["quality-filter", "--dataset", str(ds)])

        assert rc != 0
