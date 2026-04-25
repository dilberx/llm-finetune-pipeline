"""
Prometheus-compatible metrics exporter for ML infrastructure monitoring.

Exports GPU, training, and serving metrics in Prometheus text exposition
format for scraping by Prometheus/Grafana/Datadog.

Pipeshift-style: the kind of observability layer that GPU cloud
platforms need to monitor multi-tenant training workloads.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("metrics")


@dataclass
class MetricSample:
    """A single metric observation."""
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsRegistry:
    """
    Thread-safe metrics registry with Prometheus text format export.

    Usage:
        registry = MetricsRegistry()

        # Register and update metrics
        registry.gauge("gpu_memory_used_mb", 8500, {"device": "0", "model": "qwen-1.5b"})
        registry.counter("training_steps_total", 1500, {"run_id": "abc"})
        registry.gauge("gpu_utilization_pct", 85.2, {"device": "0"})

        # Export for Prometheus scraping
        print(registry.to_prometheus())

        # Export for JSON monitoring
        print(registry.to_dict())
    """

    def __init__(self):
        self._metrics: dict[str, MetricSample] = {}
        self._descriptions: dict[str, str] = {}
        self._lock = threading.Lock()

        # Register default ML infra metrics
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register standard ML infrastructure metric descriptions."""
        self._descriptions.update({
            "gpu_memory_used_mb": "GPU memory currently in use (MB)",
            "gpu_memory_total_mb": "Total GPU memory available (MB)",
            "gpu_memory_peak_mb": "Peak GPU memory usage during run (MB)",
            "gpu_utilization_pct": "GPU compute utilization percentage",
            "gpu_temperature_celsius": "GPU temperature in Celsius",
            "gpu_power_watts": "GPU power consumption in Watts",
            "training_loss": "Current training loss",
            "training_learning_rate": "Current learning rate",
            "training_steps_total": "Total training steps completed",
            "training_samples_per_second": "Training throughput in samples/sec",
            "training_tokens_per_second": "Training throughput in tokens/sec",
            "training_epoch": "Current training epoch",
            "training_gpu_hours": "Cumulative GPU-hours consumed",
            "training_estimated_cost_usd": "Estimated training cost in USD",
            "serving_requests_total": "Total inference requests served",
            "serving_latency_ms": "Average inference latency in milliseconds",
            "serving_tokens_per_second": "Inference throughput in tokens/sec",
            "serving_queue_depth": "Current inference request queue depth",
            "checkpoint_last_saved_timestamp": "Unix timestamp of last checkpoint save",
            "checkpoint_size_mb": "Size of last checkpoint in MB",
            "data_samples_processed": "Total data samples processed in curation",
            "data_duplicates_removed": "Number of duplicates removed by dedup",
        })

    def _metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate a unique key for a metric + label combination."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric (current value, can go up or down)."""
        labels = labels or {}
        key = self._metric_key(name, labels)
        with self._lock:
            self._metrics[key] = MetricSample(
                name=name, value=value, labels=labels,
                metric_type="gauge",
            )

    def counter(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a counter metric (monotonically increasing)."""
        labels = labels or {}
        key = self._metric_key(name, labels)
        with self._lock:
            self._metrics[key] = MetricSample(
                name=name, value=value, labels=labels,
                metric_type="counter",
            )

    def histogram_observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation for a histogram metric."""
        # Simplified: store as gauge for now
        self.gauge(name, value, labels)

    def to_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text exposition format.

        Output can be served at /metrics endpoint for Prometheus scraping.

        Example output:
            # HELP gpu_memory_used_mb GPU memory currently in use (MB)
            # TYPE gpu_memory_used_mb gauge
            gpu_memory_used_mb{device="0",model="qwen-1.5b"} 8500.0
        """
        lines = []
        seen_names: set[str] = set()

        with self._lock:
            for sample in sorted(self._metrics.values(), key=lambda s: s.name):
                # Add HELP and TYPE lines once per metric name
                if sample.name not in seen_names:
                    seen_names.add(sample.name)
                    desc = self._descriptions.get(sample.name, "")
                    if desc:
                        lines.append(f"# HELP {sample.name} {desc}")
                    lines.append(f"# TYPE {sample.name} {sample.metric_type}")

                # Format labels
                if sample.labels:
                    label_str = ",".join(
                        f'{k}="{v}"' for k, v in sorted(sample.labels.items())
                    )
                    lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
                else:
                    lines.append(f"{sample.name} {sample.value}")

        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as a JSON-serializable dictionary."""
        with self._lock:
            return {
                key: {
                    "name": sample.name,
                    "value": sample.value,
                    "labels": sample.labels,
                    "type": sample.metric_type,
                    "timestamp": sample.timestamp,
                }
                for key, sample in self._metrics.items()
            }

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


class TrainingCostTracker:
    """
    Tracks GPU compute costs during training.

    Estimates cost based on GPU-hours consumed and configurable
    pricing (cloud spot/on-demand rates).

    Useful for:
    - Training budget management
    - Cost-per-token analysis
    - Cloud vs. on-prem cost comparison
    """

    # Default GPU pricing (USD/hour, approximate cloud spot rates)
    DEFAULT_PRICING = {
        "RTX 3080": 0.35,
        "RTX 3090": 0.50,
        "RTX 4090": 0.75,
        "A100-40GB": 1.50,
        "A100-80GB": 2.00,
        "H100": 3.50,
        "L40S": 1.20,
        "T4": 0.25,
        "V100": 0.80,
    }

    def __init__(self, gpu_name: str = "RTX 3080", price_per_hour: float | None = None):
        self.gpu_name = gpu_name
        self.price_per_hour = price_per_hour or self._lookup_price(gpu_name)
        self._start_time: float | None = None
        self._total_seconds: float = 0
        self._total_tokens: int = 0
        self._total_steps: int = 0
        self._num_gpus: int = 1

    def _lookup_price(self, gpu_name: str) -> float:
        """Look up GPU price from defaults."""
        for key, price in self.DEFAULT_PRICING.items():
            if key.lower() in gpu_name.lower():
                return price
        return 0.50  # Default fallback

    def start(self, num_gpus: int = 1) -> None:
        """Start tracking a training run."""
        self._start_time = time.time()
        self._num_gpus = num_gpus
        logger.info("Cost tracking started", gpu=self.gpu_name,
                    price_per_hour=self.price_per_hour, num_gpus=num_gpus)

    def record_step(self, tokens: int = 0) -> None:
        """Record a completed training step."""
        self._total_steps += 1
        self._total_tokens += tokens

    def stop(self) -> None:
        """Stop tracking."""
        if self._start_time:
            self._total_seconds += time.time() - self._start_time
            self._start_time = None

    @property
    def gpu_hours(self) -> float:
        """Total GPU-hours consumed (accounting for multi-GPU)."""
        elapsed = self._total_seconds
        if self._start_time:
            elapsed += time.time() - self._start_time
        return (elapsed / 3600) * self._num_gpus

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD."""
        return self.gpu_hours * self.price_per_hour

    @property
    def cost_per_token(self) -> float:
        """Cost per training token in USD."""
        if self._total_tokens == 0:
            return 0
        return self.estimated_cost_usd / self._total_tokens

    @property
    def cost_per_1k_tokens(self) -> float:
        """Cost per 1K training tokens in USD."""
        return self.cost_per_token * 1000

    def report(self) -> dict[str, Any]:
        """Generate a cost report."""
        return {
            "gpu": self.gpu_name,
            "num_gpus": self._num_gpus,
            "price_per_hour_usd": self.price_per_hour,
            "gpu_hours": round(self.gpu_hours, 4),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "total_tokens": self._total_tokens,
            "total_steps": self._total_steps,
            "cost_per_1k_tokens_usd": round(self.cost_per_1k_tokens, 6),
        }
