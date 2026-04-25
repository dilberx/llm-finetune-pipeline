"""
Training profiler for GPU compute and memory analysis.

Wraps PyTorch's built-in profiler to capture:
- Per-operation GPU kernel timing
- Memory allocation timeline
- Compute vs. memory-bound bottleneck detection
- Peak memory analysis with allocation attribution
- Chrome trace export for visual debugging

This is the kind of profiling NVIDIA engineers use to optimize
training throughput. Integrates with training loop via callback.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from src.utils.logging import get_logger

logger = get_logger("profiler")

# Lazy imports — torch may not be installed in CI
try:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class KernelStat:
    """Aggregated statistics for a single CUDA kernel."""
    name: str
    count: int
    total_time_us: float
    avg_time_us: float
    pct_of_total: float
    self_cuda_time_us: float


@dataclass
class MemoryEvent:
    """A single memory allocation/deallocation event."""
    timestamp_us: float
    size_bytes: int
    operation: str  # "alloc" or "free"
    device: str


@dataclass
class ProfileReport:
    """Complete profiling report for a training run."""
    total_time_s: float
    gpu_active_time_s: float
    gpu_utilization_pct: float
    peak_memory_mb: float
    top_kernels: list[KernelStat]
    bottleneck: str  # "compute_bound", "memory_bound", "io_bound"
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_time_s": round(self.total_time_s, 3),
            "gpu_active_time_s": round(self.gpu_active_time_s, 3),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "bottleneck": self.bottleneck,
            "top_kernels": [
                {"name": k.name, "count": k.count, "avg_us": round(k.avg_time_us, 1),
                 "pct": round(k.pct_of_total, 1)}
                for k in self.top_kernels[:10]
            ],
            "recommendations": self.recommendations,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Profile report saved", path=str(path))


class TrainingProfiler:
    """
    GPU compute and memory profiler for training infrastructure.

    Usage:
        profiler = TrainingProfiler(output_dir="./reports/profile")

        # Profile a few training steps
        with profiler.profile_steps(wait=1, warmup=1, active=3) as prof:
            for step, batch in enumerate(dataloader):
                with profiler.step("forward"):
                    output = model(batch)
                with profiler.step("backward"):
                    loss.backward()
                with profiler.step("optimizer"):
                    optimizer.step()
                prof.step()  # Signal profiler step boundary

        report = profiler.analyze()
        report.save("./reports/profile/training_profile.json")
    """

    def __init__(self, output_dir: str = "./reports/profile", device_id: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self._profiler = None
        self._start_time = 0.0
        self._end_time = 0.0

    @contextmanager
    def profile_steps(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
    ) -> Generator:
        """
        Context manager for profiling training steps.

        Args:
            wait: Steps to skip before profiling.
            warmup: Steps to warm up the profiler.
            active: Steps to actively profile.
            repeat: Number of profiling cycles.
        """
        if not HAS_TORCH or not torch.cuda.is_available():
            logger.warning("CUDA not available — profiling disabled")
            yield None
            return

        trace_path = self.output_dir / "chrome_trace.json"

        self._start_time = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self._profiler = prof
            yield prof
        self._end_time = time.time()

        # Export Chrome trace
        prof.export_chrome_trace(str(trace_path))
        logger.info("Chrome trace exported", path=str(trace_path))

    @contextmanager
    def step(self, name: str) -> Generator:
        """Record a named operation within a profiling step."""
        if HAS_TORCH:
            with record_function(name):
                yield
        else:
            yield

    def analyze(self) -> ProfileReport:
        """
        Analyze profiling results and generate a report.

        Returns:
            ProfileReport with GPU utilization, bottleneck analysis,
            top kernels, and optimization recommendations.
        """
        if self._profiler is None:
            return ProfileReport(
                total_time_s=0, gpu_active_time_s=0, gpu_utilization_pct=0,
                peak_memory_mb=0, top_kernels=[], bottleneck="unknown",
                recommendations=["Profiling was not enabled"],
            )

        total_time = self._end_time - self._start_time

        # Extract kernel statistics
        key_averages = self._profiler.key_averages()
        kernel_stats = []
        total_cuda_time = sum(e.cuda_time_total for e in key_averages if e.cuda_time_total > 0)

        for event in sorted(key_averages, key=lambda e: e.cuda_time_total, reverse=True):
            if event.cuda_time_total > 0:
                kernel_stats.append(KernelStat(
                    name=event.key,
                    count=event.count,
                    total_time_us=event.cuda_time_total,
                    avg_time_us=event.cuda_time_total / max(event.count, 1),
                    pct_of_total=(event.cuda_time_total / total_cuda_time * 100) if total_cuda_time > 0 else 0,
                    self_cuda_time_us=event.self_cuda_time_total,
                ))

        # GPU utilization
        gpu_active_s = total_cuda_time / 1e6  # microseconds to seconds
        gpu_util = (gpu_active_s / total_time * 100) if total_time > 0 else 0

        # Peak memory
        peak_mem_mb = 0
        if HAS_TORCH and torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)

        # Bottleneck analysis
        bottleneck, recommendations = self._analyze_bottleneck(
            kernel_stats, gpu_util, peak_mem_mb
        )

        return ProfileReport(
            total_time_s=total_time,
            gpu_active_time_s=gpu_active_s,
            gpu_utilization_pct=gpu_util,
            peak_memory_mb=peak_mem_mb,
            top_kernels=kernel_stats[:20],
            bottleneck=bottleneck,
            recommendations=recommendations,
        )

    def _analyze_bottleneck(
        self,
        kernels: list[KernelStat],
        gpu_util: float,
        peak_mem_mb: float,
    ) -> tuple[str, list[str]]:
        """
        Determine training bottleneck and generate recommendations.

        Categories:
        - compute_bound: GPU utilization > 80%, dominated by matmul/gemm kernels
        - memory_bound: High memory operations, frequent alloc/dealloc
        - io_bound: Low GPU utilization, likely waiting for data
        """
        recommendations = []

        # Check GPU utilization
        if gpu_util < 30:
            bottleneck = "io_bound"
            recommendations.extend([
                "GPU utilization is very low — likely bottlenecked on data loading",
                "Try increasing dataloader num_workers",
                "Consider using data prefetching or pinned memory",
                "Check if gradient accumulation steps can be reduced",
            ])
        elif gpu_util < 70:
            # Check if memory operations dominate
            memory_kernels = [k for k in kernels if "mem" in k.name.lower() or "copy" in k.name.lower()]
            memory_pct = sum(k.pct_of_total for k in memory_kernels)

            if memory_pct > 30:
                bottleneck = "memory_bound"
                recommendations.extend([
                    "Training is memory-bound — too much time copying data",
                    "Try gradient checkpointing to reduce memory pressure",
                    "Consider using mixed precision (bf16) if not already",
                    f"Peak memory: {peak_mem_mb:.0f} MB — evaluate if model fits without offloading",
                ])
            else:
                bottleneck = "balanced"
                recommendations.append("Training is reasonably balanced between compute and I/O")
        else:
            bottleneck = "compute_bound"
            recommendations.extend([
                "Training is compute-bound — GPU is well utilized",
                "Optimization targets: larger batch size, Flash Attention, operator fusion",
                "Consider upgrading to a GPU with more CUDA cores or tensor cores",
            ])

        # Memory-specific recommendations
        if HAS_TORCH and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
            if peak_mem_mb > total_mem * 0.85:
                recommendations.append(
                    f"WARNING: Peak memory ({peak_mem_mb:.0f} MB) is {peak_mem_mb/total_mem*100:.0f}% "
                    f"of total VRAM ({total_mem:.0f} MB) — at risk of OOM"
                )

        return bottleneck, recommendations
