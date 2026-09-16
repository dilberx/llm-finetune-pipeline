"""
Configuration management for the LLM fine-tuning pipeline.

Loads YAML configs with validation, environment variable overrides,
and sensible defaults for consumer GPU hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console()


@dataclass
class ModelConfig:
    """Model identification and loading settings."""
    name: str = "Qwen/Qwen2.5-Coder-1.5B"
    revision: str = "main"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization settings for QLoRA."""
    enabled: bool = True
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """LoRA adapter settings."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters and logging settings."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    optim: str = "paged_adamw_8bit"
    report_to: str = "none"
    run_name: str = "qwen2.5-coder-1.5b-code-lora"
    packing: bool = False


@dataclass
class DatasetConfig:
    """Dataset paths and settings."""
    name: str = "curated"
    curated_path: str = "./data/curated/train.jsonl"
    eval_path: str = "./data/curated/eval.jsonl"
    text_field: str = "text"
    max_samples: int | None = None
    # Caps the eval split alone. `max_samples` scales both splits together, so
    # without this the only way to shorten a mid-training eval is to throw away
    # training data with it.
    max_eval_samples: int | None = None


@dataclass
class HardwareConfig:
    """Hardware and reproducibility settings."""
    device_map: str = "auto"
    seed: int = 42


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides.

    Convention: LFP_<SECTION>_<KEY>=value
    Example: LFP_TRAINING_LEARNING_RATE=1e-5
    """
    prefix = "LFP_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section in config and isinstance(config[section], dict):
            # Attempt type coercion based on existing value
            existing = config[section].get(key)
            if existing is not None:
                try:
                    if isinstance(existing, bool):
                        config[section][key] = env_value.lower() in ("true", "1", "yes")
                    elif isinstance(existing, int):
                        config[section][key] = int(env_value)
                    elif isinstance(existing, float):
                        config[section][key] = float(env_value)
                    else:
                        config[section][key] = env_value
                except (ValueError, TypeError):
                    config[section][key] = env_value
            else:
                config[section][key] = env_value
            console.print(f"[dim]env override: {env_key} → {section}.{key} = {config[section][key]}[/dim]")
    return config


def _dict_to_dataclass(section: str, data: dict) -> Any:
    """Convert a dict to the appropriate dataclass based on section name."""
    mapping = {
        "model": ModelConfig,
        "quantization": QuantizationConfig,
        "lora": LoRAConfig,
        "training": TrainingConfig,
        "dataset": DatasetConfig,
        "hardware": HardwareConfig,
    }
    cls = mapping.get(section)
    if cls is None:
        return data
    # Only pass keys that the dataclass accepts
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(config_path: str | Path) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.

    Applies defaults → YAML overrides → environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Fully resolved PipelineConfig instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        console.print(f"[yellow]Config file not found: {config_path}, using defaults[/yellow]")
        return PipelineConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Apply environment overrides
    raw = _apply_env_overrides(raw)

    # Build PipelineConfig from parsed YAML
    return PipelineConfig(
        model=_dict_to_dataclass("model", raw.get("model", {})),
        quantization=_dict_to_dataclass("quantization", raw.get("quantization", {})),
        lora=_dict_to_dataclass("lora", raw.get("lora", {})),
        training=_dict_to_dataclass("training", raw.get("training", {})),
        dataset=_dict_to_dataclass("dataset", raw.get("dataset", {})),
        hardware=_dict_to_dataclass("hardware", raw.get("hardware", {})),
    )


def print_config(config: PipelineConfig) -> None:
    """Pretty-print the pipeline configuration."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(title="Pipeline Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="bold")
    table.add_column("Parameter")
    table.add_column("Value", style="green")

    sections = {
        "Model": vars(config.model),
        "Quantization": vars(config.quantization),
        "LoRA": vars(config.lora),
        "Training": vars(config.training),
        "Dataset": vars(config.dataset),
        "Hardware": vars(config.hardware),
    }

    for section_name, params in sections.items():
        for i, (key, value) in enumerate(params.items()):
            section_display = section_name if i == 0 else ""
            table.add_row(section_display, key, str(value))
        table.add_section()

    console.print(Panel(table, border_style="blue"))
