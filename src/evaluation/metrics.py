"""
Statistical metrics for model comparison.

Provides paired statistical tests to determine whether fine-tuning produced a
real improvement over the base model, rather than a difference the evaluation
set is too small to distinguish from noise.

Aggregate metrics alone cannot answer that. Where benchmarks report per-example
results, the comparison pairs them by example so that example difficulty drops
out of the variance, and reports an interval on the difference alongside the
point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

# Aggregate metric -> the per-example field it is computed from. Most share a
# name; a rate over examples does not, since the aggregate is a proportion and
# the per-example value is the boolean it counts.
_PAIRABLE_FROM_RAW = {
    "exact_match": "exact_match",
    "normalized_exact_match": "normalized_exact_match",
    "ast_match": "ast_match",
    "ast_alpha_match": "ast_alpha_match",
    "token_f1": "token_f1",
    "edit_similarity": "edit_similarity",
    "parse_rate": "parses",
}


def compute_comparison_stats(
    base_results: dict[str, Any],
    finetuned_results: dict[str, Any],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Compute statistical comparison between base and fine-tuned model results.

    For each shared benchmark metric, computes absolute and relative
    improvement and the direction of improvement.

    Where a benchmark also returned `raw_results` for both models, a `paired`
    block is added per metric carrying a bootstrap confidence interval on the
    mean difference, a Wilcoxon signed-rank p-value, and the paired effect
    size. Benchmarks that report only aggregates (perplexity, pass@k) get the
    point estimates alone -- there is nothing to pair.

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

        base_raw = base_benchmarks[bench_name].get("raw_results")
        ft_raw = ft_benchmarks[bench_name].get("raw_results")

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

            entry = {
                "base": base_val,
                "finetuned": ft_val,
                "absolute_diff": absolute_diff,
                "relative_diff_pct": relative_diff,
                "improved": improved,
                "direction": "lower_better" if is_lower_better else "higher_better",
            }

            paired = _paired_block(
                base_raw, ft_raw, metric_name, confidence_level=confidence_level
            )
            if paired is not None:
                entry["paired"] = paired

            bench_comparison[metric_name] = entry

        comparison[bench_name] = bench_comparison

    comparison["summary"] = _generate_summary(comparison)
    return comparison


def _paired_block(
    base_raw: list[dict[str, Any]] | None,
    ft_raw: list[dict[str, Any]] | None,
    metric_name: str,
    confidence_level: float,
) -> dict[str, Any] | None:
    """Build the paired statistics for one metric, or None if not pairable."""
    if not base_raw or not ft_raw or metric_name not in _PAIRABLE_FROM_RAW:
        return None
    field = _PAIRABLE_FROM_RAW[metric_name]
    try:
        base_scores, ft_scores = align_raw_results(base_raw, ft_raw, field)
    except (KeyError, ValueError):
        return None
    if len(base_scores) < 2:
        return None
    return paired_difference(
        base_scores, ft_scores, confidence_level=confidence_level
    )


def align_raw_results(
    base_raw: Sequence[dict[str, Any]],
    ft_raw: Sequence[dict[str, Any]],
    field: str,
    key: str = "prompt",
) -> tuple[list[float], list[float]]:
    """
    Line up two models' per-example results on the same examples.

    Matches on `key` when both sides carry it, so a reordered or partially
    failed run still pairs correctly. Falls back to position only when the two
    runs are the same length and the key is unavailable.

    Args:
        base_raw: Per-example results for the base model.
        ft_raw: Per-example results for the fine-tuned model.
        field: Which per-example field to extract.
        key: Field identifying an example across runs.

    Returns:
        (base_scores, finetuned_scores), aligned and equal in length.

    Raises:
        ValueError: If the runs share no examples, or cannot be aligned.
    """
    have_key = all(key in row for row in base_raw) and all(key in row for row in ft_raw)

    if have_key:
        base_by_key = {str(row[key]): row for row in base_raw}
        ft_by_key = {str(row[key]): row for row in ft_raw}
        shared = sorted(set(base_by_key) & set(ft_by_key))
        if not shared:
            raise ValueError(f"base and fine-tuned runs share no examples on key '{key}'")
        base_scores = [float(base_by_key[k].get(field, 0.0)) for k in shared]
        ft_scores = [float(ft_by_key[k].get(field, 0.0)) for k in shared]
        return base_scores, ft_scores

    if len(base_raw) != len(ft_raw):
        raise ValueError(
            f"cannot align {len(base_raw)} base results with {len(ft_raw)} fine-tuned "
            f"results: no '{key}' field to match on and lengths differ"
        )
    return (
        [float(row.get(field, 0.0)) for row in base_raw],
        [float(row.get(field, 0.0)) for row in ft_raw],
    )


def paired_difference(
    base_scores: Sequence[float],
    finetuned_scores: Sequence[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compare two models on the same examples, pairing within each example.

    Reports a bootstrap interval on the mean difference plus a Wilcoxon
    signed-rank p-value. Wilcoxon rather than a t-test because scores like
    token F1 are bounded and skewed, and at the sample sizes involved here
    normality is an assumption rather than an observation.

    When every per-example difference is zero the comparison is flagged
    degenerate: the metric cannot distinguish these models on this set at any
    sample size, which is a statement about the metric, not the models.

    Args:
        base_scores: Base model score per example.
        finetuned_scores: Fine-tuned model score on the same examples, in order.
        confidence_level: Coverage for the interval on the difference.
        n_bootstrap: Bootstrap resamples.
        seed: Seed for the resampler.

    Returns:
        Dict with the mean difference, its interval, the p-value, the paired
        effect size, and whether the interval supports a directional claim.

    Raises:
        ValueError: If the inputs differ in length or hold fewer than 2 pairs.
    """
    base = np.asarray(base_scores, dtype=float)
    ft = np.asarray(finetuned_scores, dtype=float)

    if base.shape != ft.shape:
        raise ValueError(f"paired inputs must align: got {base.shape} and {ft.shape}")
    n = len(base)
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")

    diffs = ft - base
    mean_diff = float(np.mean(diffs))

    if np.allclose(diffs, 0.0):
        return {
            "n_pairs": n,
            "mean_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "confidence_level": confidence_level,
            "p_value": 1.0,
            "effect_size_dz": 0.0,
            "significant": False,
            "degenerate": True,
            "note": (
                "every example scored identically under both models; this metric "
                "cannot separate them on this set at any sample size"
            ),
        }

    ci_low, ci_high = bootstrap_confidence_interval(
        diffs.tolist(),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    try:
        p_value = float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        # scipy refuses some all-tied configurations; no evidence of a shift.
        p_value = 1.0

    spread = float(np.std(diffs, ddof=1))
    effect_size = mean_diff / spread if spread > 0 else 0.0
    alpha = 1 - confidence_level

    return {
        "n_pairs": n,
        "mean_difference": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence_level,
        "p_value": p_value,
        "effect_size_dz": float(effect_size),
        "significant": bool(p_value < alpha),
        "degenerate": False,
    }


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
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        samples: List of observed values.
        confidence_level: Desired confidence level.
        n_bootstrap: Number of bootstrap iterations.
        statistic: Which statistic to bootstrap ("mean" or "median").
        seed: Seed for the resampler, so a report is reproducible.

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    # Resample all iterations at once; the loop form dominated runtime at the
    # 10k resamples the paired comparison asks for.
    observations = np.asarray(samples, dtype=float)
    draws = rng.choice(observations, size=(n_bootstrap, len(observations)), replace=True)
    bootstrap_stats = stat_fn(draws, axis=1)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return lower, upper
