"""
Structured logging for production ML infrastructure.

Replaces ad-hoc console.print() with structured JSON logs that can
be ingested by monitoring systems (ELK, Datadog, CloudWatch).

Features:
- JSON log format with timestamp, level, component, correlation_id
- Configurable output: console (human-readable) + file (JSON)
- Training-specific event types (epoch, checkpoint, oom_warning)
- Context propagation via correlation IDs across pipeline stages
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Output format:
    {"timestamp": "2026-04-25T14:30:00Z", "level": "INFO", "component": "training",
     "message": "Epoch 1 complete", "correlation_id": "abc-123", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "component": getattr(record, "component", "general"),
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }

        # Add extra structured fields
        extra = getattr(record, "structured_data", None)
        if extra and isinstance(extra, dict):
            log_entry["data"] = extra

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """
    Human-readable console formatter with color support.

    For local development — switches to JSON in production
    via LOG_FORMAT=json environment variable.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        component = getattr(record, "component", "general")
        msg = record.getMessage()

        formatted = f"{color}[{record.levelname:>8}]{self.RESET} [{component}] {msg}"

        extra = getattr(record, "structured_data", None)
        if extra and isinstance(extra, dict):
            formatted += f"  {json.dumps(extra, default=str)}"

        return formatted


# ============================================================================
# Logger Factory
# ============================================================================

# Global correlation ID for the current pipeline run
_correlation_id: str = str(uuid.uuid4())[:8]


def set_correlation_id(cid: str | None = None) -> str:
    """Set or generate a correlation ID for this pipeline run."""
    global _correlation_id
    _correlation_id = cid or str(uuid.uuid4())[:8]
    return _correlation_id


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id


class StructuredLogger:
    """
    Wrapper around Python logging with structured data support.

    Usage:
        logger = get_logger("training")
        logger.info("Epoch complete", epoch=1, loss=0.45, lr=2e-4)
        logger.warning("OOM risk detected", device_id=0, free_mb=512)
    """

    def __init__(self, name: str, component: str):
        self._logger = logging.getLogger(name)
        self._component = component

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log a message with structured data."""
        extra = {
            "component": self._component,
            "correlation_id": _correlation_id,
            "structured_data": kwargs if kwargs else None,
        }
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    # --- Training-Specific Events ---

    def epoch_start(self, epoch: int, total_epochs: int, **kwargs: Any) -> None:
        self._log(logging.INFO, f"Epoch {epoch}/{total_epochs} started", epoch=epoch, total_epochs=total_epochs, event="epoch_start", **kwargs)

    def epoch_end(self, epoch: int, loss: float, **kwargs: Any) -> None:
        self._log(logging.INFO, f"Epoch {epoch} complete — loss={loss:.4f}", epoch=epoch, loss=loss, event="epoch_end", **kwargs)

    def checkpoint_saved(self, path: str, step: int, **kwargs: Any) -> None:
        self._log(logging.INFO, f"Checkpoint saved at step {step}", path=path, step=step, event="checkpoint_saved", **kwargs)

    def checkpoint_loaded(self, path: str, step: int, **kwargs: Any) -> None:
        self._log(logging.INFO, f"Resumed from checkpoint at step {step}", path=path, step=step, event="checkpoint_loaded", **kwargs)

    def oom_warning(self, device_id: int, used_mb: float, total_mb: float, **kwargs: Any) -> None:
        self._log(logging.WARNING, f"OOM risk on GPU {device_id}: {used_mb:.0f}/{total_mb:.0f} MB", device_id=device_id, used_mb=used_mb, total_mb=total_mb, event="oom_warning", **kwargs)

    def training_complete(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        self._log(logging.INFO, "Training complete", metrics=metrics, event="training_complete", **kwargs)

    def pipeline_stage(self, stage: str, status: str, **kwargs: Any) -> None:
        self._log(logging.INFO, f"Pipeline stage: {stage} — {status}", stage=stage, status=status, event="pipeline_stage", **kwargs)


def get_logger(component: str = "general") -> StructuredLogger:
    """
    Get a structured logger for a pipeline component.

    Configures handlers based on environment:
    - LOG_FORMAT=json → JSON output (production)
    - LOG_FORMAT=human (default) → colored console output
    - LOG_FILE=path → also write to file (always JSON)
    - LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (default: INFO)
    """
    logger_name = f"llm_pipeline.{component}"
    logger = logging.getLogger(logger_name)

    # Only configure once
    if logger.handlers:
        return StructuredLogger(logger_name, component)

    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    log_format = os.environ.get("LOG_FORMAT", "human").lower()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(HumanFormatter())
    logger.addHandler(console_handler)

    # File handler (always JSON)
    log_file = os.environ.get("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return StructuredLogger(logger_name, component)
