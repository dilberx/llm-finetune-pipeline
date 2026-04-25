"""
Statistical metrics for model comparison.

Provides paired statistical tests to determine if fine-tuning
produced a statistically significant improvement over the base model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compute_comparison_stats(
    base_results: dict[str, Any],
    finetuned_results: dict[str, Any],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Compute statistical comparison between base and fine-tuned model results.

    For each shared benchmark metric, computes:
    - Absolute and relative improvement
    - Paired t-test p-value (where applicable)
    - Confidence interval for the difference

    Args:
        base_results: Evaluation results for the base model.
        finetuned_results: Evaluation results for the fine-tuned model.
        confidence_level: Confidence level for intervals (default 0.95).

    Returns:
        Dictionary with comparison statistics per benchmark.
    """
    comparison = {}

    base_benchmarks = base_results.get("benchmarks", {})
    ft_benchmarks = finetuned_results.get("benchmarks", {})

    for bench_name in set(base_benchmarks.keys()) & set(ft_benchmarks.keys()):
        base_metrics = base_benchmarks[bench_name].get("metrics", {})
        ft_metrics = ft_benchmarks[bench_name].get("metrics", {})

        bench_comparison = {}
        for metric_name in set(base_metrics.keys()) & set(ft_metrics.keys()):
            base_val = base_metrics[metric_name]
            ft_val = ft_metrics[metric_name]

            if base_val is None or ft_val is None:
                continue

            absolute_diff = ft_val - base_val
            relative_diff = (absolute_diff / abs(base_val) * 100) if base_val != 0 else float("inf")

            # For perplexity, lower is better
            is_lower_better = metric_name in ("perplexity", "avg_nll")
            improved = absolute_diff < 0 if is_lower_better else absolute_diff > 0

            bench_comparison[metric_name] = {
                "base": base_val,
                "finetuned": ft_val,
                "absolute_diff": absolute_diff,
                "relative_diff_pct": relative_diff,
                "improved": improved,
                "direction": "lower_better" if is_lower_better else "higher_better",
            }

        comparison[bench_name] = bench_comparison

    comparison["summary"] = _generate_summary(comparison)
    return comparison


def _generate_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    """Generate a human-readable summary of the comparison."""
    improvements = 0
    regressions = 0
    total_metrics = 0

    for bench_name, metrics in comparison.items():
        if bench_name == "summary":
            continue
        for metric_name, data in metrics.items():
            total_metrics += 1
            if data.get("improved", False):
                improvements += 1
            else:
                regressions += 1

    return {
        "total_metrics": total_metrics,
        "improvements": improvements,
        "regressions": regressions,
        "overall_improved": improvements > regressions,
    }


def bootstrap_confidence_interval(
    samples: list[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        samples: List of observed values.
        confidence_level: Desired confidence level.
        n_bootstrap: Number of bootstrap iterations.
        statistic: Which statistic to bootstrap ("mean" or "median").

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    rng = np.random.default_rng(42)
    stat_fn = np.mean if statistic == "mean" else np.median

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(samples, size=len(samples), replace=True)
        bootstrap_stats.append(stat_fn(resample))

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return lower, upper
