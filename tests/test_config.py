"""
Tests for configuration management.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config, PipelineConfig, ModelConfig, LoRAConfig


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
