"""
Tests for the evaluation metrics module.

Tests statistical comparison, bootstrap CIs, and pass@k estimation
without requiring GPU or model loading.
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_comparison_stats,
    bootstrap_confidence_interval,
)
from src.evaluation.benchmarks import _estimate_pass_at_k


class TestPassAtK:
    def test_all_correct(self):
        """If all samples are correct, pass@k should be 1.0."""
        assert _estimate_pass_at_k(n=10, c=10, k=1) == 1.0

    def test_none_correct(self):
        """If no samples are correct, pass@k should be 0.0."""
        assert _estimate_pass_at_k(n=10, c=0, k=1) == 0.0

    def test_partial(self):
        """pass@1 with 5/10 correct should be 0.5."""
        result = _estimate_pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 0.01

    def test_k_greater_than_failures(self):
        """If k > n-c, pass@k should be 1.0."""
        assert _estimate_pass_at_k(n=10, c=8, k=5) == 1.0

    def test_monotonic_in_k(self):
        """pass@k should increase with k."""
        p1 = _estimate_pass_at_k(n=20, c=5, k=1)
        p5 = _estimate_pass_at_k(n=20, c=5, k=5)
        p10 = _estimate_pass_at_k(n=20, c=5, k=10)
        assert p1 <= p5 <= p10


class TestComparisonStats:
    def test_improvement_detected(self):
        base = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.3}}}}
        finetuned = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.5}}}}
        result = compute_comparison_stats(base, finetuned)

        assert "humaneval" in result
        assert result["humaneval"]["pass@1"]["improved"] is True
        assert result["humaneval"]["pass@1"]["absolute_diff"] == pytest.approx(0.2)

    def test_regression_detected(self):
        base = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.5}}}}
        finetuned = {"benchmarks": {"humaneval": {"metrics": {"pass@1": 0.3}}}}
        result = compute_comparison_stats(base, finetuned)

        assert result["humaneval"]["pass@1"]["improved"] is False

    def test_perplexity_lower_is_better(self):
        """For perplexity, a decrease should count as improvement."""
        base = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 25.0}}}}
        finetuned = {"benchmarks": {"perplexity": {"metrics": {"perplexity": 20.0}}}}
        result = compute_comparison_stats(base, finetuned)

        assert result["perplexity"]["perplexity"]["improved"] is True

    def test_summary_generation(self):
        base = {"benchmarks": {
            "humaneval": {"metrics": {"pass@1": 0.3}},
            "perplexity": {"metrics": {"perplexity": 25.0}},
        }}
        finetuned = {"benchmarks": {
            "humaneval": {"metrics": {"pass@1": 0.5}},
            "perplexity": {"metrics": {"perplexity": 20.0}},
        }}
        result = compute_comparison_stats(base, finetuned)

        assert result["summary"]["total_metrics"] == 2
        assert result["summary"]["improvements"] == 2
        assert result["summary"]["overall_improved"] is True


class TestBootstrapCI:
    def test_contains_mean(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_confidence_interval(samples, confidence_level=0.95)
        assert lower <= np.mean(samples) <= upper

    def test_narrow_for_constant(self):
        samples = [5.0] * 100
        lower, upper = bootstrap_confidence_interval(samples)
        assert abs(upper - lower) < 0.01

    def test_wider_for_high_variance(self):
        low_var = [5.0, 5.1, 4.9, 5.0, 5.1]
        high_var = [1.0, 10.0, 2.0, 9.0, 5.0]
        _, upper_low = bootstrap_confidence_interval(low_var)
        lower_low, _ = bootstrap_confidence_interval(low_var)
        _, upper_high = bootstrap_confidence_interval(high_var)
        lower_high, _ = bootstrap_confidence_interval(high_var)

        width_low = upper_low - lower_low
        width_high = upper_high - lower_high
        assert width_high > width_low
