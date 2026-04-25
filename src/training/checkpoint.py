"""
Checkpoint management with fault tolerance.

Provides:
- Automatic discovery of the latest valid checkpoint
- Integrity validation (corrupt/incomplete saves)
- Resume-safe checkpoint selection

Designed to be importable without GPU dependencies (no torch/peft)
so checkpoint logic can be tested in CI without GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest valid checkpoint in the output directory.

    Scans for checkpoint-* directories sorted by step number
    (highest first) and returns the first one that passes
    integrity validation.

    Args:
        output_dir: Directory containing checkpoint-N subdirectories.

    Returns:
        Path to the latest valid checkpoint, or None.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )

    for ckpt in checkpoints:
        if validate_checkpoint(ckpt):
            return str(ckpt)

    return None


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate checkpoint integrity before resuming.

    Checks:
    1. trainer_state.json exists, is non-empty, and is valid JSON
    2. trainer_state.json contains a global_step field
    3. At least one model artifact file exists

    This prevents resuming from a corrupted save (e.g., crash
    during write, disk full, or interrupted checkpoint).

    Args:
        checkpoint_path: Path to a checkpoint-N directory.

    Returns:
        True if the checkpoint is safe to resume from.
    """
    # trainer_state.json must exist and be valid JSON
    trainer_state = checkpoint_path / "trainer_state.json"
    if not trainer_state.exists() or trainer_state.stat().st_size == 0:
        return False

    try:
        with open(trainer_state) as f:
            state = json.load(f)
        # Must have a global_step recorded
        if "global_step" not in state:
            return False
    except (json.JSONDecodeError, KeyError):
        return False

    # At least one model file must exist
    model_indicators = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    has_model = any((checkpoint_path / f).exists() for f in model_indicators)
    if not has_model:
        return False

    return True


def get_checkpoint_step(checkpoint_path: str | Path) -> int | None:
    """Extract the step number from a checkpoint directory name or state file."""
    path = Path(checkpoint_path)

    # Try from directory name
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            pass

    # Try from trainer_state.json
    state_file = path / "trainer_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            return state.get("global_step")
        except (json.JSONDecodeError, KeyError):
            pass

    return None
