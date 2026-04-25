"""
Tests for GPU resource manager, structured logging, metrics, and profiling.
"""

import json
import os
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.gpu import GPUResourceManager, GPUSnapshot, MemoryTracker
from src.utils.logging import get_logger, set_correlation_id, get_correlation_id, JSONFormatter
from src.utils.metrics_export import MetricsRegistry, TrainingCostTracker


# ============================================================================
# GPU Resource Manager Tests
# ============================================================================

class TestMemoryTracker:
    def test_peak_tracking(self):
        tracker = MemoryTracker(device_id=0)
        tracker.record(1000)
        tracker.record(2000)
        tracker.record(1500)
        assert tracker.peak_mb == 2000

    def test_oom_prediction_insufficient_samples(self):
        tracker = MemoryTracker(device_id=0)
        tracker.record(1000)
        result = tracker.predict_oom(total_mb=10000)
        assert result["risk"] == "unknown"

    def test_oom_prediction_stable(self):
        tracker = MemoryTracker(device_id=0)
        for _ in range(10):
            tracker.record(5000)
        result = tracker.predict_oom(total_mb=10000)
        assert result["risk"] == "low"
        assert result["trend_mb_per_sample"] == 0

    def test_oom_prediction_rising(self):
        tracker = MemoryTracker(device_id=0)
        for i in range(20):
            tracker.record(8000 + i * 100)  # rising fast toward 10GB
        result = tracker.predict_oom(total_mb=10000)
        assert result["risk"] in ("high", "critical")
        assert result["trend_mb_per_sample"] > 0

    def test_oom_prediction_decreasing(self):
        tracker = MemoryTracker(device_id=0)
        for i in range(10):
            tracker.record(8000 - i * 100)  # decreasing
        result = tracker.predict_oom(total_mb=10000)
        assert result["risk"] == "low"


class TestGPUSnapshot:
    def test_used_pct(self):
        snap = GPUSnapshot(
            device_id=0, device_name="RTX 3080", total_mb=10240,
            used_mb=5120, free_mb=5120, utilization_pct=50,
            temperature_c=65, power_w=200,
        )
        assert snap.used_pct == 50.0

    def test_to_dict(self):
        snap = GPUSnapshot(
            device_id=0, device_name="RTX 3080", total_mb=10240,
            used_mb=5120, free_mb=5120, utilization_pct=50,
            temperature_c=65, power_w=200,
        )
        d = snap.to_dict()
        assert d["device_name"] == "RTX 3080"
        assert d["used_pct"] == 50.0
        assert "timestamp" in d


class TestGPUResourceManager:
    def test_no_gpu_graceful(self):
        """Should not crash when no GPU is available."""
        manager = GPUResourceManager()
        snapshot = manager.get_snapshot(device_id=99)
        # Returns None when device doesn't exist
        # (may return actual data if test machine has GPUs)

    def test_recommend_device_map_no_gpu(self):
        manager = GPUResourceManager()
        # Patch to simulate no GPUs
        with patch.object(manager, 'get_all_snapshots', return_value=[]):
            result = manager.recommend_device_map(model_size_mb=5000)
            assert result["strategy"] == "cpu"

    def test_recommend_device_map_single_gpu_fits(self):
        manager = GPUResourceManager()
        snap = GPUSnapshot(
            device_id=0, device_name="RTX 3080", total_mb=10240,
            used_mb=1000, free_mb=9240, utilization_pct=10,
            temperature_c=40, power_w=50,
        )
        with patch.object(manager, 'get_all_snapshots', return_value=[snap]):
            result = manager.recommend_device_map(model_size_mb=3000)
            assert result["strategy"] == "single_gpu"

    def test_recommend_device_map_single_gpu_insufficient(self):
        manager = GPUResourceManager()
        snap = GPUSnapshot(
            device_id=0, device_name="RTX 3080", total_mb=10240,
            used_mb=8000, free_mb=2240, utilization_pct=80,
            temperature_c=70, power_w=250,
        )
        with patch.object(manager, 'get_all_snapshots', return_value=[snap]):
            result = manager.recommend_device_map(model_size_mb=5000)
            assert result["strategy"] == "offload_cpu"

    def test_export_metrics(self):
        manager = GPUResourceManager()
        metrics = manager.export_metrics()
        assert "timestamp" in metrics
        assert "num_gpus" in metrics
        assert isinstance(metrics["devices"], list)


# ============================================================================
# Structured Logging Tests
# ============================================================================

class TestStructuredLogging:
    def test_get_logger(self):
        logger = get_logger("test_component")
        assert logger is not None

    def test_correlation_id(self):
        cid = set_correlation_id("test-123")
        assert cid == "test-123"
        assert get_correlation_id() == "test-123"

    def test_auto_correlation_id(self):
        cid = set_correlation_id()
        assert len(cid) == 8  # uuid4[:8]

    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None,
        )
        record.component = "training"
        record.correlation_id = "abc-123"
        record.structured_data = {"epoch": 1, "loss": 0.5}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["component"] == "training"
        assert parsed["message"] == "Test message"
        assert parsed["correlation_id"] == "abc-123"
        assert parsed["data"]["epoch"] == 1

    def test_training_events(self):
        """Verify training-specific log methods don't crash."""
        logger = get_logger("test_training_events")
        logger.epoch_start(epoch=1, total_epochs=3)
        logger.epoch_end(epoch=1, loss=0.45)
        logger.checkpoint_saved(path="/tmp/ckpt", step=100)
        logger.checkpoint_loaded(path="/tmp/ckpt", step=100)
        logger.oom_warning(device_id=0, used_mb=9000, total_mb=10240)
        logger.training_complete(metrics={"loss": 0.3})
        logger.pipeline_stage("training", "completed")

    def test_json_log_to_file(self, tmp_path):
        """Verify JSON logs are written to file correctly."""
        log_file = tmp_path / "test.log"
        os.environ["LOG_FORMAT"] = "json"
        os.environ["LOG_FILE"] = str(log_file)

        # Force new logger (clear handlers)
        logger_name = "llm_pipeline.test_file_output"
        py_logger = logging.getLogger(logger_name)
        py_logger.handlers.clear()

        logger = get_logger("test_file_output")
        logger.info("File test message", key="value")

        # Clean up env
        del os.environ["LOG_FORMAT"]
        del os.environ["LOG_FILE"]

        assert log_file.exists()
        content = log_file.read_text().strip()
        if content:
            parsed = json.loads(content)
            assert parsed["message"] == "File test message"


# ============================================================================
# Checkpoint Fault Tolerance Tests
# ============================================================================

class TestCheckpointFaultTolerance:
    def test_find_latest_checkpoint_empty(self, tmp_path):
        from src.training.checkpoint import find_latest_checkpoint
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_find_latest_checkpoint_valid(self, tmp_path):
        from src.training.checkpoint import find_latest_checkpoint

        # Create valid checkpoint
        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text('{"global_step": 100}')
        (ckpt / "adapter_model.safetensors").write_text("fake")

        result = find_latest_checkpoint(str(tmp_path))
        assert result == str(ckpt)

    def test_find_latest_checkpoint_picks_highest(self, tmp_path):
        from src.training.checkpoint import find_latest_checkpoint

        for step in [50, 100, 200]:
            ckpt = tmp_path / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "trainer_state.json").write_text(f'{{"global_step": {step}}}')
            (ckpt / "adapter_model.safetensors").write_text("fake")

        result = find_latest_checkpoint(str(tmp_path))
        assert "checkpoint-200" in result

    def test_skips_corrupted_checkpoint(self, tmp_path):
        from src.training.checkpoint import find_latest_checkpoint

        # Corrupted checkpoint (empty trainer_state.json)
        bad = tmp_path / "checkpoint-200"
        bad.mkdir()
        (bad / "trainer_state.json").write_text("")

        # Valid older checkpoint
        good = tmp_path / "checkpoint-100"
        good.mkdir()
        (good / "trainer_state.json").write_text('{"global_step": 100}')
        (good / "adapter_model.safetensors").write_text("fake")

        result = find_latest_checkpoint(str(tmp_path))
        assert "checkpoint-100" in result

    def test_validate_checkpoint_missing_model(self, tmp_path):
        from src.training.checkpoint import validate_checkpoint

        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text('{"global_step": 100}')
        # No model file → invalid
        assert validate_checkpoint(ckpt) is False

    def test_validate_checkpoint_invalid_json(self, tmp_path):
        from src.training.checkpoint import validate_checkpoint

        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text("{broken json")
        (ckpt / "adapter_model.safetensors").write_text("fake")
        assert validate_checkpoint(ckpt) is False


# ============================================================================
# Prometheus Metrics Tests
# ============================================================================

class TestMetricsRegistry:
    def test_gauge(self):
        registry = MetricsRegistry()
        registry.gauge("gpu_memory_used_mb", 8500, {"device": "0"})
        d = registry.to_dict()
        key = 'gpu_memory_used_mb{device=0}'
        assert key in d
        assert d[key]["value"] == 8500

    def test_counter(self):
        registry = MetricsRegistry()
        registry.counter("serving_requests_total", 1500)
        d = registry.to_dict()
        assert "serving_requests_total" in d
        assert d["serving_requests_total"]["type"] == "counter"

    def test_prometheus_format(self):
        registry = MetricsRegistry()
        registry.gauge("gpu_utilization_pct", 85.2, {"device": "0"})
        output = registry.to_prometheus()
        assert "# TYPE gpu_utilization_pct gauge" in output
        assert 'gpu_utilization_pct{device="0"} 85.2' in output

    def test_prometheus_help_text(self):
        registry = MetricsRegistry()
        registry.gauge("gpu_memory_used_mb", 5000)
        output = registry.to_prometheus()
        assert "# HELP gpu_memory_used_mb" in output

    def test_multiple_labels(self):
        registry = MetricsRegistry()
        registry.gauge("gpu_memory_used_mb", 5000, {"device": "0", "gpu": "RTX 3080"})
        registry.gauge("gpu_memory_used_mb", 3000, {"device": "1", "gpu": "RTX 3080"})
        output = registry.to_prometheus()
        assert 'device="0"' in output
        assert 'device="1"' in output

    def test_reset(self):
        registry = MetricsRegistry()
        registry.gauge("test_metric", 100)
        registry.reset()
        assert len(registry.to_dict()) == 0


class TestTrainingCostTracker:
    def test_basic_tracking(self):
        tracker = TrainingCostTracker(gpu_name="RTX 3080", price_per_hour=0.35)
        tracker.start()
        time.sleep(0.1)
        tracker.stop()
        assert tracker.gpu_hours > 0
        assert tracker.estimated_cost_usd > 0

    def test_cost_per_token(self):
        tracker = TrainingCostTracker(gpu_name="RTX 3080", price_per_hour=1.00)
        tracker.start()
        tracker.record_step(tokens=1000)
        tracker.record_step(tokens=1000)
        time.sleep(0.1)
        tracker.stop()
        assert tracker.cost_per_token > 0
        assert tracker.cost_per_1k_tokens > 0

    def test_report(self):
        tracker = TrainingCostTracker(gpu_name="A100-80GB")
        tracker.start()
        tracker.record_step(tokens=2048)
        time.sleep(0.05)
        tracker.stop()
        report = tracker.report()
        assert report["gpu"] == "A100-80GB"
        assert report["total_steps"] == 1
        assert report["total_tokens"] == 2048
        assert "estimated_cost_usd" in report

    def test_default_pricing_lookup(self):
        tracker = TrainingCostTracker(gpu_name="NVIDIA H100")
        assert tracker.price_per_hour == 3.50

    def test_unknown_gpu_fallback(self):
        tracker = TrainingCostTracker(gpu_name="Some Unknown GPU")
        assert tracker.price_per_hour == 0.50  # fallback

    def test_zero_tokens_no_crash(self):
        tracker = TrainingCostTracker(gpu_name="RTX 3080")
        assert tracker.cost_per_token == 0
        assert tracker.cost_per_1k_tokens == 0

