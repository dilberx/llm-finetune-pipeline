"""
LoRA/QLoRA fine-tuning pipeline for code LLMs.

Supports:
- 4-bit QLoRA with BitsAndBytes for consumer GPUs (10GB VRAM)
- Configurable LoRA rank, target modules, and training hyperparameters
- Multi-GPU training via HuggingFace Accelerate
- Gradient checkpointing for memory efficiency
- W&B / TensorBoard logging
- Automatic checkpoint management with best-model selection
"""

from __future__ import annotations

import argparse
import json
import os
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from rich.console import Console
from rich.panel import Panel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig

from src.utils.config import load_config, print_config, PipelineConfig

console = Console()


# ============================================================================
# Model Loading
# ============================================================================

def _build_bnb_config(config: PipelineConfig) -> BitsAndBytesConfig | None:
    """Build BitsAndBytes quantization config for QLoRA."""
    if not config.quantization.enabled:
        return None

    compute_dtype = getattr(torch, config.quantization.compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=(config.quantization.bits == 4),
        load_in_8bit=(config.quantization.bits == 8),
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora_config(config: PipelineConfig) -> LoraConfig:
    """Build PEFT LoRA configuration."""
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(config: PipelineConfig) -> tuple:
    """
    Load the base model with quantization and apply LoRA adapters.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    console.print(Panel(
        f"[bold]Loading model:[/bold] {config.model.name}\n"
        f"Quantization: {config.quantization.bits}-bit {config.quantization.quant_type}\n"
        f"LoRA rank: {config.lora.r}, alpha: {config.lora.alpha}",
        title="Model Setup",
        border_style="blue",
    ))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build quantization config
    bnb_config = _build_bnb_config(config)

    # Load base model
    torch_dtype = getattr(torch, config.model.torch_dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=bnb_config,
        device_map=config.hardware.device_map,
        torch_dtype=torch_dtype,
    )

    # Prepare for k-bit training (handles gradient computation through quantized layers)
    if config.quantization.enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )

    # Apply LoRA adapters
    lora_config = _build_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameter count
    trainable_params, total_params = model.get_nb_trainable_parameters()
    console.print(
        f"\n[bold green]Trainable parameters:[/bold green] "
        f"{trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


# ============================================================================
# Dataset Loading
# ============================================================================

def load_training_data(config: PipelineConfig, tokenizer) -> tuple:
    """
    Load and prepare training and evaluation datasets.

    Supports:
    - Local JSONL files (from the curation pipeline)
    - HuggingFace dataset names
    """
    tc = config.training

    if config.dataset.name == "curated":
        # Load from local curated JSONL
        train_path = config.dataset.curated_path
        eval_path = config.dataset.eval_path

        console.print(f"\n[bold blue]Loading curated data:[/bold blue]")
        console.print(f"  Train: {train_path}")
        console.print(f"  Eval:  {eval_path}")

        train_dataset = load_dataset("json", data_files=train_path, split="train")
        eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    else:
        # Load from HuggingFace Hub
        console.print(f"\n[bold blue]Loading HF dataset:[/bold blue] {config.dataset.name}")
        dataset = load_dataset(config.dataset.name)
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation") or dataset.get("test")

    # Limit samples if configured
    if config.dataset.max_samples:
        train_dataset = train_dataset.select(
            range(min(config.dataset.max_samples, len(train_dataset)))
        )
        if eval_dataset:
            eval_dataset = eval_dataset.select(
                range(min(config.dataset.max_samples, len(eval_dataset)))
            )

    # Mid-training eval exists to pick a checkpoint, not to benchmark, and
    # eval_loss is a mean over tokens that settles long before the full split
    # is consumed. Capping it decouples how often you can afford to evaluate
    # from how much held-out data was curated.
    if eval_dataset is not None and config.dataset.max_eval_samples:
        eval_dataset = eval_dataset.select(
            range(min(config.dataset.max_eval_samples, len(eval_dataset)))
        )

    console.print(f"  Train samples: {len(train_dataset)}")
    if eval_dataset:
        console.print(f"  Eval samples:  {len(eval_dataset)}")

    return train_dataset, eval_dataset


# ============================================================================
# Training
# ============================================================================

def build_training_args(config: PipelineConfig) -> SFTConfig:
    """Build SFT training arguments from pipeline config."""
    tc = config.training

    args = dict(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_ratio=tc.warmup_ratio,
        weight_decay=tc.weight_decay,
        max_grad_norm=tc.max_grad_norm,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=tc.bf16,
        fp16=tc.fp16,
        gradient_checkpointing=tc.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=tc.dataloader_num_workers,
        optim=tc.optim,
        report_to=tc.report_to if tc.report_to != "none" else [],
        run_name=tc.run_name,
        seed=config.hardware.seed,
        dataset_text_field=config.dataset.text_field,
        packing=tc.packing,
    )
    length_arg = (
        "max_length"
        if "max_length" in signature(SFTConfig.__init__).parameters
        else "max_seq_length"
    )
    args[length_arg] = tc.max_seq_length
    return SFTConfig(**args)


# Checkpoint management is in its own module (no GPU deps → testable in CI)
from src.training.checkpoint import find_latest_checkpoint, validate_checkpoint


def train(config: PipelineConfig, resume_from: str | None = None) -> Path:
    """
    Execute the full fine-tuning pipeline with infrastructure-grade reliability.

    Features:
    - Automatic checkpoint resume on failure (finds latest valid checkpoint)
    - GPU memory monitoring with OOM early warning
    - Structured JSON logging for observability
    - Checkpoint integrity validation before resume
    - Training state metadata persistence

    Args:
        config: Pipeline configuration.
        resume_from: Optional explicit checkpoint path to resume from.

    Returns:
        Path to the saved adapter checkpoint.
    """
    from src.utils.logging import get_logger, set_correlation_id
    from src.utils.gpu import GPUResourceManager

    logger = get_logger("training")
    run_id = set_correlation_id()
    gpu_manager = GPUResourceManager()

    logger.pipeline_stage("training", "started", run_id=run_id, model=config.model.name)
    print_config(config)

    # Set seed for reproducibility
    torch.manual_seed(config.hardware.seed)

    # --- GPU Resource Check ---
    snapshots = gpu_manager.get_all_snapshots()
    if snapshots:
        for snap in snapshots:
            logger.info("GPU detected", **snap.to_dict())
    else:
        logger.warning("No GPU detected — training will be slow or fail")

    # --- Load model and tokenizer ---
    logger.pipeline_stage("model_loading", "started")
    with gpu_manager.track_operation("model_load") as load_metrics:
        model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded", **load_metrics)

    # --- Load datasets ---
    logger.pipeline_stage("data_loading", "started")
    train_dataset, eval_dataset = load_training_data(config, tokenizer)

    # --- Build training arguments ---
    training_args = build_training_args(config)

    # --- Checkpoint Resume Logic ---
    resume_checkpoint = resume_from
    if resume_checkpoint is None:
        # Auto-detect latest valid checkpoint
        latest = find_latest_checkpoint(config.training.output_dir)
        if latest:
            logger.checkpoint_loaded(path=latest, step=0)
            resume_checkpoint = latest
        else:
            logger.info("No previous checkpoint found — starting fresh")

    # --- Initialize trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # --- Train with GPU monitoring ---
    logger.pipeline_stage("training", "running")

    # Record pre-training GPU state
    oom_check = gpu_manager.record_usage()
    if oom_check:
        logger.oom_warning(**{k: v for k, v in oom_check.get("snapshot", {}).items()
                             if k in ("device_id", "used_mb", "total_mb")})

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Log training metrics ---
    metrics = train_result.metrics
    logger.training_complete(metrics=metrics)

    # --- Save the final adapter ---
    output_dir = Path(config.training.output_dir) / "best"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- Save training metadata with infra context ---
    gpu_metrics = gpu_manager.export_metrics()
    metadata = {
        "run_id": run_id,
        "base_model": config.model.name,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "quantization_bits": config.quantization.bits,
        "training_metrics": metrics,
        "trainable_params": model.get_nb_trainable_parameters()[0],
        "total_params": model.get_nb_trainable_parameters()[1],
        "gpu_metrics": gpu_metrics,
        "resumed_from": resume_checkpoint,
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.checkpoint_saved(path=str(output_dir), step=metrics.get("global_step", 0))
    logger.pipeline_stage("training", "completed", output_dir=str(output_dir))

    return output_dir


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune a code LLM with QLoRA")
    parser.add_argument(
        "--config", type=str, default="configs/finetune_config.yaml",
        help="Path to fine-tuning config YAML",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint to resume training from (auto-detects if not specified)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
