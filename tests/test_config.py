"""
Tests for configuration management.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    DatasetConfig,
    LoRAConfig,
    ModelConfig,
    PipelineConfig,
    load_config,
)


class TestConfigLoading:
    def test_default_config(self):
        """Loading a non-existent config should return defaults."""
        config = load_config("/nonexistent/path.yaml")
        assert isinstance(config, PipelineConfig)
        assert config.model.name == "Qwen/Qwen2.5-Coder-1.5B"
        assert config.lora.r == 16

    def test_yaml_override(self, tmp_path):
        """YAML values should override defaults."""
        yaml_content = {
            "model": {"name": "custom/model"},
            "lora": {"r": 32, "alpha": 64},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_path)
        assert config.model.name == "custom/model"
        assert config.lora.r == 32
        assert config.lora.alpha == 64
        # Non-overridden values should keep defaults
        assert config.training.learning_rate == 2e-4

    def test_env_override(self, tmp_path):
        """Environment variables should override YAML values."""
        yaml_content = {"training": {"learning_rate": 1e-4}}
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        os.environ["LFP_TRAINING_LEARNING_RATE"] = "5e-5"
        try:
            config = load_config(config_path)
            assert config.training.learning_rate == 5e-5
        finally:
            del os.environ["LFP_TRAINING_LEARNING_RATE"]

    def test_empty_yaml(self, tmp_path):
        """Empty YAML should return defaults."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(config_path)
        assert isinstance(config, PipelineConfig)


class TestConfigDataclasses:
    def test_model_config_defaults(self):
        mc = ModelConfig()
        assert mc.name == "Qwen/Qwen2.5-Coder-1.5B"
        assert mc.trust_remote_code is True

    def test_lora_config_target_modules(self):
        lc = LoRAConfig()
        assert "q_proj" in lc.target_modules
        assert len(lc.target_modules) == 7


class TestEvalSampleCap:
    def test_defaults_to_no_cap(self):
        assert DatasetConfig().max_eval_samples is None

    def test_reads_from_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("dataset:\n  max_eval_samples: 500\n  max_samples: 1000\n")
        config = load_config(str(path))
        assert config.dataset.max_eval_samples == 500

    def test_is_independent_of_max_samples(self):
        """The point of the knob: shrink eval without shrinking training data."""
        config = DatasetConfig(max_samples=None, max_eval_samples=500)
        assert config.max_samples is None
        assert config.max_eval_samples == 500


class TestCheckpointCadence:
    """
    HF requires save_steps to be a multiple of eval_steps when
    load_best_model_at_end is set, which finetune.py sets unconditionally.
    A config that violates it fails at Trainer construction, ~20 minutes into
    a run after model loading and tokenization.
    """

    def test_shipped_config_satisfies_the_trainer_constraint(self):
        config = load_config("configs/finetune_config.yaml")
        assert config.training.save_steps % config.training.eval_steps == 0

    def test_shipped_config_caps_eval(self):
        config = load_config("configs/finetune_config.yaml")
        assert config.dataset.max_eval_samples is not None, (
            "frequent eval is only affordable with a capped eval split"
        )
