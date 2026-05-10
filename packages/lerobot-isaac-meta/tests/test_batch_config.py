"""Tests for lerobot_isaac_meta.batch_config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lerobot_isaac_meta.batch_config import (
    BatchConfig,
    BatchConfigError,
    CompareSpec,
    RunSpec,
    load_batch_config,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "batch.yaml"
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------------


class TestRunSpec:
    def test_minimal_valid(self):
        r = RunSpec(id="a", target_arch="smolvla")
        assert r.id == "a"
        assert r.target_arch == "smolvla"
        assert r.config is None
        assert r.extra_args == []

    def test_unknown_arch_rejected(self):
        with pytest.raises(BatchConfigError, match="target_arch"):
            RunSpec(id="a", target_arch="bogus")

    def test_blank_id_rejected(self):
        with pytest.raises(BatchConfigError, match="run.id"):
            RunSpec(id="   ", target_arch="smolvla")


# ---------------------------------------------------------------------------
# CompareSpec
# ---------------------------------------------------------------------------


class TestCompareSpec:
    def test_default(self):
        c = CompareSpec()
        assert c.enabled is True
        assert c.mode == "nway"

    def test_invalid_mode(self):
        with pytest.raises(BatchConfigError, match="compare.mode"):
            CompareSpec(mode="3way")


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------


class TestBatchConfig:
    def test_minimal_valid(self):
        cfg = BatchConfig(
            batch_id="b",
            dataset="data/x",
            runs=[
                RunSpec(id="a", target_arch="smolvla"),
                RunSpec(id="b", target_arch="le_world_model"),
            ],
        )
        assert cfg.batch_id == "b"
        assert cfg.on_failure == "continue"
        assert cfg.compare.mode == "nway"

    def test_empty_runs_rejected(self):
        with pytest.raises(BatchConfigError, match="at least one"):
            BatchConfig(batch_id="b", dataset="d", runs=[])

    def test_duplicate_ids_rejected(self):
        with pytest.raises(BatchConfigError, match="duplicate run ids"):
            BatchConfig(
                batch_id="b",
                dataset="d",
                runs=[
                    RunSpec(id="x", target_arch="smolvla"),
                    RunSpec(id="x", target_arch="act"),
                ],
            )

    def test_invalid_failure_policy(self):
        with pytest.raises(BatchConfigError, match="on_failure"):
            BatchConfig(
                batch_id="b",
                dataset="d",
                on_failure="explode",
                runs=[RunSpec(id="a", target_arch="smolvla")],
            )

    def test_2way_requires_exactly_two_runs(self):
        with pytest.raises(BatchConfigError, match="2way"):
            BatchConfig(
                batch_id="b",
                dataset="d",
                runs=[RunSpec(id="a", target_arch="smolvla")],
                compare=CompareSpec(mode="2way"),
            )

    def test_resolved_dataset_uses_run_override(self):
        cfg = BatchConfig(
            batch_id="b",
            dataset="default/path",
            runs=[
                RunSpec(id="a", target_arch="smolvla", dataset="override/path"),
                RunSpec(id="b", target_arch="act"),
            ],
        )
        assert cfg.resolved_dataset(cfg.runs[0]) == "override/path"
        assert cfg.resolved_dataset(cfg.runs[1]) == "default/path"


# ---------------------------------------------------------------------------
# load_batch_config
# ---------------------------------------------------------------------------


class TestLoadBatchConfig:
    def test_valid_yaml(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            batch_id: smolvla-vs-lewm
            dataset: datasets/so101/pick
            output_root: outputs/runs
            on_failure: continue
            compare:
              enabled: true
              mode: nway
            runs:
              - id: a
                target_arch: smolvla
                config: cfg/a.yaml
                steps: 100
                label: SmolVLA
              - id: b
                target_arch: le_world_model
                config: cfg/b.yaml
                steps: 100
            """,
        )
        cfg = load_batch_config(p)
        assert cfg.batch_id == "smolvla-vs-lewm"
        assert cfg.dataset == "datasets/so101/pick"
        assert len(cfg.runs) == 2
        assert cfg.runs[0].id == "a"
        assert cfg.runs[0].label == "SmolVLA"
        assert cfg.runs[1].target_arch == "le_world_model"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_batch_config(tmp_path / "nope.yaml")

    def test_missing_dataset(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            batch_id: x
            runs:
              - id: a
                target_arch: smolvla
            """,
        )
        with pytest.raises(BatchConfigError, match="dataset"):
            load_batch_config(p)

    def test_runs_not_a_list(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            batch_id: x
            dataset: d
            runs:
              not: a-list
            """,
        )
        with pytest.raises(BatchConfigError, match="must be a list"):
            load_batch_config(p)

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(BatchConfigError):
            load_batch_config(p)
