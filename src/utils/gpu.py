"""
GPU resource manager for monitoring, allocation, and OOM prevention.

Provides:
- Real-time VRAM tracking (used, free, peak, fragmentation estimate)
- OOM early warning based on allocation trajectory
- Multi-GPU device enumeration and memory balancing
- Structured metrics export for monitoring systems

Works with NVIDIA GPUs via pynvml. Gracefully degrades on non-GPU
machines (Apple Silicon, CPU-only containers).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from rich.console import Console

console = Console()

# Optional NVIDIA imports — graceful degradation on non-GPU systems
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except (ImportError, Exception):
    HAS_NVML = False


@dataclass
class GPUSnapshot:
    """Point-in-time snapshot of GPU memory state."""
    device_id: int
    device_name: str
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_pct: float
    temperature_c: int
    power_w: float
    timestamp: float = field(default_factory=time.time)

    @property
    def used_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "total_mb": round(self.total_mb, 1),
            "used_mb": round(self.used_mb, 1),
            "free_mb": round(self.free_mb, 1),
            "used_pct": round(self.used_pct, 1),
            "utilization_pct": round(self.utilization_pct, 1),
            "temperature_c": self.temperature_c,
            "power_w": round(self.power_w, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryTracker:
    """Tracks VRAM usage over time for a single GPU."""
    device_id: int
    snapshots: list[float] = field(default_factory=list)
    peak_mb: float = 0.0
    _oom_threshold_pct: float = 92.0

    def record(self, used_mb: float) -> None:
        """Record a memory usage sample."""
        self.snapshots.append(used_mb)
        self.peak_mb = max(self.peak_mb, used_mb)

    def predict_oom(self, total_mb: float, window: int = 10) -> dict[str, Any]:
        """
        Predict OOM risk based on recent allocation trajectory.

        Fits a linear trend to the last `window` samples and
        estimates time until VRAM exhaustion.
        """
        if len(self.snapshots) < 3:
            return {"risk": "unknown", "reason": "insufficient_samples"}

        recent = self.snapshots[-window:]
        if len(recent) < 2:
            return {"risk": "low", "trend_mb_per_sample": 0}

        # Simple linear regression on recent samples
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0
        current = recent[-1]
        threshold = total_mb * (self._oom_threshold_pct / 100)

        if slope <= 0:
            return {
                "risk": "low",
                "trend_mb_per_sample": round(slope, 2),
                "current_mb": round(current, 1),
                "peak_mb": round(self.peak_mb, 1),
            }

        samples_to_oom = (threshold - current) / slope if slope > 0 else float("inf")

        if samples_to_oom < 5:
            risk = "critical"
        elif samples_to_oom < 20:
            risk = "high"
        elif samples_to_oom < 50:
            risk = "medium"
        else:
            risk = "low"

        return {
            "risk": risk,
            "trend_mb_per_sample": round(slope, 2),
            "samples_to_threshold": round(samples_to_oom, 0),
            "current_mb": round(current, 1),
            "peak_mb": round(self.peak_mb, 1),
            "threshold_mb": round(threshold, 1),
        }


class GPUResourceManager:
    """
    Manages GPU resources for ML training infrastructure.

    Responsibilities:
    - Enumerate available GPUs and their capabilities
    - Track VRAM usage over time with peak detection
    - Predict OOM before it crashes training
    - Export structured metrics for monitoring ingestion
    - Recommend optimal device_map for multi-GPU setups
    """

    def __init__(self, oom_threshold_pct: float = 92.0):
        self.oom_threshold_pct = oom_threshold_pct
        self._trackers: dict[int, MemoryTracker] = {}
        self._available = HAS_NVML or HAS_TORCH

    @property
    def num_gpus(self) -> int:
        """Return number of available GPUs."""
        if HAS_NVML:
            return pynvml.nvmlDeviceGetCount()
        if HAS_TORCH and torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def get_snapshot(self, device_id: int = 0) -> GPUSnapshot | None:
        """Capture a point-in-time GPU memory snapshot."""
        if HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                return GPUSnapshot(
                    device_id=device_id,
                    device_name=name,
                    total_mb=mem_info.total / (1024 ** 2),
                    used_mb=mem_info.used / (1024 ** 2),
                    free_mb=mem_info.free / (1024 ** 2),
                    utilization_pct=util.gpu,
                    temperature_c=temp,
                    power_w=power,
                )
            except Exception:
                pass

        if HAS_TORCH and torch.cuda.is_available() and device_id < torch.cuda.device_count():
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
            used = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            return GPUSnapshot(
                device_id=device_id,
                device_name=torch.cuda.get_device_name(device_id),
                total_mb=total,
                used_mb=used,
                free_mb=total - used,
                utilization_pct=0,  # Not available via torch
                temperature_c=0,
                power_w=0,
            )

        return None

    def get_all_snapshots(self) -> list[GPUSnapshot]:
        """Capture snapshots for all available GPUs."""
        return [
            snap for i in range(self.num_gpus)
            if (snap := self.get_snapshot(i)) is not None
        ]

    def record_usage(self, device_id: int = 0) -> dict[str, Any] | None:
        """
        Record current VRAM usage and check for OOM risk.

        Returns OOM prediction if risk is medium or higher.
        """
        snapshot = self.get_snapshot(device_id)
        if snapshot is None:
            return None

        if device_id not in self._trackers:
            self._trackers[device_id] = MemoryTracker(
                device_id=device_id,
                _oom_threshold_pct=self.oom_threshold_pct,
            )

        tracker = self._trackers[device_id]
        tracker.record(snapshot.used_mb)

        prediction = tracker.predict_oom(snapshot.total_mb)
        if prediction.get("risk") in ("high", "critical"):
            return {
                "warning": "OOM_RISK",
                "device_id": device_id,
                "snapshot": snapshot.to_dict(),
                "prediction": prediction,
            }

        return None

    @contextmanager
    def track_operation(self, operation_name: str, device_id: int = 0) -> Generator[dict, None, None]:
        """
        Context manager to track GPU memory during an operation.

        Usage:
            with gpu_manager.track_operation("forward_pass") as metrics:
                output = model(input_ids)
            print(metrics)  # {'operation': 'forward_pass', 'peak_mb': ..., 'delta_mb': ...}
        """
        metrics: dict[str, Any] = {"operation": operation_name, "device_id": device_id}

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)
            before = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        else:
            before = 0

        try:
            yield metrics
        finally:
            if HAS_TORCH and torch.cuda.is_available():
                after = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                peak = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
                metrics.update({
                    "before_mb": round(before, 1),
                    "after_mb": round(after, 1),
                    "peak_mb": round(peak, 1),
                    "delta_mb": round(after - before, 1),
                })

    def recommend_device_map(self, model_size_mb: float) -> dict[str, Any]:
        """
        Recommend device placement for a model across available GPUs.

        Returns a strategy based on available VRAM vs model size.
        """
        snapshots = self.get_all_snapshots()

        if not snapshots:
            return {"strategy": "cpu", "reason": "no_gpu_available"}

        total_free = sum(s.free_mb for s in snapshots)

        if len(snapshots) == 1:
            if snapshots[0].free_mb >= model_size_mb * 1.2:
                return {"strategy": "single_gpu", "device": 0, "free_mb": round(snapshots[0].free_mb, 1)}
            else:
                return {"strategy": "offload_cpu", "device": 0, "reason": "insufficient_vram",
                        "free_mb": round(snapshots[0].free_mb, 1), "needed_mb": round(model_size_mb, 1)}

        if total_free >= model_size_mb * 1.1:
            devices = sorted(range(len(snapshots)), key=lambda i: snapshots[i].free_mb, reverse=True)
            return {"strategy": "multi_gpu", "devices": devices, "total_free_mb": round(total_free, 1)}

        return {"strategy": "offload_cpu", "reason": "insufficient_total_vram",
                "total_free_mb": round(total_free, 1), "needed_mb": round(model_size_mb, 1)}

    def export_metrics(self) -> dict[str, Any]:
        """Export all tracked metrics as structured JSON for monitoring systems."""
        snapshots = self.get_all_snapshots()
        tracker_metrics = {}

        for device_id, tracker in self._trackers.items():
            snap = next((s for s in snapshots if s.device_id == device_id), None)
            total_mb = snap.total_mb if snap else 0
            tracker_metrics[f"gpu_{device_id}"] = {
                "peak_mb": round(tracker.peak_mb, 1),
                "num_samples": len(tracker.snapshots),
                "oom_prediction": tracker.predict_oom(total_mb),
            }

        return {
            "timestamp": time.time(),
            "num_gpus": self.num_gpus,
            "devices": [s.to_dict() for s in snapshots],
            "trackers": tracker_metrics,
        }
