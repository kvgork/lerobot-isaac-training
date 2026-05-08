"""
test_yaml_configs.py — Tests for the 6 YAML config files in lerobot_isaac_configs.

Verifies:
  - Each file parses without error.
  - Required top-level keys are present with correct types.
  - Numeric values are within reasonable ranges for RTX 3080 10 GB.

Plan reference: §13.1 Bundle A, deliverable A6
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_real_config(name: str) -> dict:
    """Load a config from the package's configs/ directory (bypasses env var)."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "lerobot_isaac_configs" / "configs"
    )
    path = configs_dir / f"{name}.yaml"
    assert path.exists(), f"Config file not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"Top-level must be a mapping, got {type(data)}"
    return data


# ---------------------------------------------------------------------------
# Policy configs
# ---------------------------------------------------------------------------


class TestPolicySmolvla:
    def test_parses(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg is not None

    def test_target_arch(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg["target_arch"] == "smolvla"

    def test_batch_size(self):
        cfg = load_real_config("policy_smolvla")
        assert isinstance(cfg["training"]["batch_size"], int)
        assert 1 <= cfg["training"]["batch_size"] <= 64

    def test_lr_type_and_range(self):
        cfg = load_real_config("policy_smolvla")
        lr = cfg["training"]["lr"]
        assert isinstance(lr, (int, float))
        assert 1e-7 < lr < 1e-1

    def test_steps(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg["training"]["steps"] == 50000

    def test_use_amp(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg["training"]["use_amp"] is True

    def test_num_workers(self):
        cfg = load_real_config("policy_smolvla")
        assert isinstance(cfg["training"]["num_workers"], int)

    def test_eval_interval(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg["training"]["eval_interval"] == 5000

    def test_dataset_num_episodes_target(self):
        cfg = load_real_config("policy_smolvla")
        assert cfg["dataset"]["num_episodes_target"] == 50


class TestPolicyAct:
    def test_parses(self):
        cfg = load_real_config("policy_act")
        assert cfg is not None

    def test_target_arch(self):
        cfg = load_real_config("policy_act")
        assert cfg["target_arch"] == "act"

    def test_batch_size(self):
        cfg = load_real_config("policy_act")
        assert cfg["training"]["batch_size"] == 4

    def test_lr(self):
        cfg = load_real_config("policy_act")
        lr = cfg["training"]["lr"]
        assert isinstance(lr, (int, float))
        assert abs(lr - 1e-5) < 1e-8

    def test_steps(self):
        cfg = load_real_config("policy_act")
        assert cfg["training"]["steps"] == 100000

    def test_use_amp(self):
        cfg = load_real_config("policy_act")
        assert cfg["training"]["use_amp"] is True

    def test_grad_accumulation(self):
        cfg = load_real_config("policy_act")
        assert cfg["training"]["grad_accumulation"] == 2

    def test_chunk_size(self):
        cfg = load_real_config("policy_act")
        assert cfg["model"]["chunk_size"] == 100


class TestPolicyDiffusion:
    def test_parses(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg is not None

    def test_target_arch(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg["target_arch"] == "diffusion"

    def test_batch_size(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg["training"]["batch_size"] == 8

    def test_lr(self):
        cfg = load_real_config("policy_diffusion")
        lr = cfg["training"]["lr"]
        assert isinstance(lr, (int, float))
        assert abs(lr - 1e-4) < 1e-7

    def test_steps(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg["training"]["steps"] == 75000

    def test_use_amp(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg["training"]["use_amp"] is True

    def test_grad_clip_norm(self):
        cfg = load_real_config("policy_diffusion")
        assert cfg["training"]["grad_clip_norm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# World model configs
# ---------------------------------------------------------------------------


class TestWmDreamerv3:
    def test_parses(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg is not None

    def test_target_arch(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["target_arch"] == "dreamerv3"

    def test_batch_size(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["training"]["batch_size"] == 16

    def test_batch_length(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["training"]["batch_length"] == 64

    def test_steps(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["training"]["steps"] == 200000

    def test_world_model_lr(self):
        cfg = load_real_config("wm_dreamerv3")
        wm_lr = cfg["world_model"]["world_model_lr"]
        assert isinstance(wm_lr, (int, float))
        assert abs(wm_lr - 1e-4) < 1e-7

    def test_deter_size(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["world_model"]["deter_size"] == 512

    def test_stoch_size(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["world_model"]["stoch_size"] == 32

    def test_metric_key(self):
        cfg = load_real_config("wm_dreamerv3")
        assert cfg["metric_key"] == "recon_loss"


class TestWmLeworldmodel:
    def test_parses(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg is not None

    def test_target_arch(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg["target_arch"] == "le_world_model"

    def test_batch_size(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg["training"]["batch_size"] == 8

    def test_lr(self):
        cfg = load_real_config("wm_leworldmodel")
        lr = cfg["training"]["lr"]
        assert isinstance(lr, (int, float))
        assert abs(lr - 5e-5) < 1e-8

    def test_steps(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg["training"]["steps"] == 100000

    def test_prediction_horizon(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg["model"]["prediction_horizon"] == 16

    def test_metric_key(self):
        cfg = load_real_config("wm_leworldmodel")
        assert cfg["metric_key"] == "pred_loss"


# ---------------------------------------------------------------------------
# Isaac env config
# ---------------------------------------------------------------------------


class TestIsaacSo101Pickplace:
    def test_parses(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg is not None

    def test_task(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg["task"] == "pick_and_place"

    def test_num_envs(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg["env"]["num_envs"] == 4

    def test_episode_length(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg["env"]["episode_length_s"] == pytest.approx(10.0)

    def test_decimation(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg["env"]["decimation"] == 2

    def test_sim_dt(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert isinstance(cfg["sim"]["dt"], (int, float))
        assert abs(cfg["sim"]["dt"] - 0.00833) < 1e-5

    def test_dr_stage(self):
        cfg = load_real_config("isaac_so101_pickplace")
        assert cfg["dr_stage"] == 1


# ---------------------------------------------------------------------------
# Cross-cutting: load via package API (integration smoke)
# ---------------------------------------------------------------------------


class TestLoadViaApi:
    """load_config() should return the same data as direct YAML parsing."""

    @pytest.mark.parametrize(
        "name",
        [
            "policy_smolvla",
            "policy_act",
            "policy_diffusion",
            "wm_dreamerv3",
            "wm_leworldmodel",
            "isaac_so101_pickplace",
        ],
    )
    def test_load_config_api_matches_direct(
        self, name: str, tmp_path: Path, monkeypatch
    ):
        """load_config() returns same data as direct YAML load."""
        # Use the real configs dir via LEROBOT_ISAAC_CONFIGS_DIR
        configs_dir = (
            Path(__file__).parent.parent / "src" / "lerobot_isaac_configs" / "configs"
        )
        monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(configs_dir))

        from lerobot_isaac_configs.loader import load_config

        cfg = load_config(name)
        expected = load_real_config(name)
        assert cfg == expected

    def test_list_configs_contains_all_six(self, monkeypatch):
        """list_configs() should return exactly 6 entries after Bundle A."""
        configs_dir = (
            Path(__file__).parent.parent / "src" / "lerobot_isaac_configs" / "configs"
        )
        monkeypatch.setenv("LEROBOT_ISAAC_CONFIGS_DIR", str(configs_dir))

        from lerobot_isaac_configs.loader import list_configs

        names = list_configs()
        expected = {
            "policy_smolvla",
            "policy_act",
            "policy_diffusion",
            "wm_dreamerv3",
            "wm_leworldmodel",
            "isaac_so101_pickplace",
        }
        assert set(names) == expected
