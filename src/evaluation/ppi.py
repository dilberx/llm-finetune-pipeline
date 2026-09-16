"""
Prediction-powered inference for small-sample evaluation.

Sixteen held-out examples cannot support a confident claim about a model
change. PPI recovers power by pairing those few gold-labelled examples with a
large pool of *unlabelled* examples scored by a cheap proxy, then correcting
the proxy's bias using the gold set.

The correction is what makes the estimate valid: the proxy may be
systematically wrong and the interval still covers, because the bias is
measured on the gold set rather than assumed away.

The power-tuned variant weights that correction by lambda, chosen to minimise
variance. It collapses to the classical gold-only mean when the proxy carries
no usable signal, so switching it on can never do worse than ignoring the
unlabelled pool. lambda is the control-variates coefficient from Monte Carlo
estimation, arrived at from a different direction.

References:
    Angelopoulos et al., "Prediction-Powered Inference" (2023).
    Angelopoulos et al., "PPI++: Efficient Prediction-Powered Inference" (2023).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class PPIEstimate:
    """A prediction-powered estimate of a mean, beside its classical counterpart."""

    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    lambda_weight: float
    n_gold: int
    n_unlabeled: int
    classical_estimate: float
    classical_std_error: float
    classical_ci_low: float
    classical_ci_high: float
    effective_gold_n: float
    confidence_level: float
    proxy_correlation: float = 0.0
    lambda_saturated: bool = False

    def warning(self) -> str | None:
        """
        Describe a proxy that is not earning its place, or None.

        The failure worth catching is a proxy pointing the wrong way. The
        tuned weight is clamped at zero, so a negatively-correlated proxy
        produces exactly the classical estimate and looks indistinguishable
        from a proxy that simply carried no signal.
        """
        if self.n_unlabeled == 0:
            return None
        if self.proxy_correlation < 0 and self.lambda_weight == 0.0:
            return (
                f"The proxy is negatively correlated with the measured metric "
                f"(r={self.proxy_correlation:+.2f}), so the pool was discarded and this is "
                f"the plain {self.n_gold}-example estimate. Check the proxy's orientation: "
                f"it must rise with quality."
            )
        if self.lambda_weight == 0.0:
            return (
                f"The proxy carried no usable signal (r={self.proxy_correlation:+.2f}); the "
                f"pool was discarded and this is the plain {self.n_gold}-example estimate."
            )
        if self.lambda_saturated:
            return (
                f"The tuned weight hit its bound, so the pool is being used less than the "
                f"data suggests it could be (r={self.proxy_correlation:+.2f}). The estimate "
                f"is valid but not as tight as a rescaled proxy would make it."
            )
        return None

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def classical_ci_width(self) -> float:
        return self.classical_ci_high - self.classical_ci_low

    @property
    def excludes_zero(self) -> bool:
        """True when the interval supports a directional claim."""
        return self.ci_low > 0 or self.ci_high < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_width,
            "lambda": self.lambda_weight,
            "n_gold": self.n_gold,
            "n_unlabeled": self.n_unlabeled,
            "classical": {
                "estimate": self.classical_estimate,
                "std_error": self.classical_std_error,
                "ci_low": self.classical_ci_low,
                "ci_high": self.classical_ci_high,
                "ci_width": self.classical_ci_width,
            },
            "effective_gold_n": self.effective_gold_n,
            "confidence_level": self.confidence_level,
            "excludes_zero": self.excludes_zero,
            "proxy_correlation": self.proxy_correlation,
            "lambda_saturated": self.lambda_saturated,
            "warning": self.warning(),
        }


def _z_score(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    return float(stats.norm.ppf(1 - (1 - confidence_level) / 2))


def optimal_lambda(
    gold_labels: Sequence[float],
    gold_proxy: Sequence[float],
    n_unlabeled: int,
) -> float:
    """
    Variance-minimising weight for the proxy correction.

    Derived by minimising Var(theta_lambda) in lambda:

        lambda* = Cov(Y, f) / (Var(f) * (1 + n/N))

    As N grows this approaches the regression coefficient of the gold label on
    the proxy. Clamped to [0, 1]: 0 discards the unlabelled pool entirely,
    1 is untuned PPI.

    Args:
        gold_labels: True scores on the gold set.
        gold_proxy: Proxy scores on the same gold examples, same order.
        n_unlabeled: Size of the unlabelled pool.

    Returns:
        lambda in [0, 1]. Returns 0.0 when the proxy has no variance.
    """
    raw = _unclamped_lambda(gold_labels, gold_proxy, n_unlabeled)
    return float(np.clip(raw, 0.0, 1.0))


def ppi_mean(
    gold_labels: Sequence[float],
    gold_proxy: Sequence[float],
    unlabeled_proxy: Sequence[float],
    confidence_level: float = 0.95,
    power_tune: bool = True,
) -> PPIEstimate:
    """
    Estimate a mean from a few gold labels plus many proxy-scored examples.

    The estimator is

        theta = mean(Y) + lambda * (mean(f_unlabeled) - mean(f_gold))

    which reads as: take the honest gold-set mean, then shift it by the proxy's
    view of how the wider pool differs from the gold sample. Because the shift
    is anchored on proxy scores measured in both places, a biased proxy cancels
    out to first order.

    Args:
        gold_labels: True scores for the gold examples.
        gold_proxy: Proxy scores for those same examples, in the same order.
        unlabeled_proxy: Proxy scores for the unlabelled pool.
        confidence_level: Coverage for the returned interval.
        power_tune: Tune lambda to minimise variance. When False, lambda = 1
            (untuned PPI), which can be less efficient than ignoring the pool.

    Returns:
        A PPIEstimate carrying both the PPI and classical intervals.

    Raises:
        ValueError: If the gold arrays differ in length or hold fewer than two
            examples, since no variance can be estimated from one point.
    """
    y = np.asarray(gold_labels, dtype=float)
    f = np.asarray(gold_proxy, dtype=float)
    f_pool = np.asarray(unlabeled_proxy, dtype=float)

    if y.shape != f.shape:
        raise ValueError(
            f"gold_labels and gold_proxy must align: got {y.shape} and {f.shape}"
        )
    n = len(y)
    if n < 2:
        raise ValueError(f"need at least 2 gold examples to estimate variance, got {n}")

    big_n = len(f_pool)
    z = _z_score(confidence_level)

    # Classical: gold labels only, the baseline this must beat.
    var_y = float(np.var(y, ddof=1))
    classical_est = float(np.mean(y))
    classical_se = float(np.sqrt(var_y / n))
    classical_lo = classical_est - z * classical_se
    classical_hi = classical_est + z * classical_se

    if big_n == 0:
        # Nothing to borrow strength from; PPI degenerates to classical.
        return PPIEstimate(
            estimate=classical_est,
            std_error=classical_se,
            ci_low=classical_lo,
            ci_high=classical_hi,
            lambda_weight=0.0,
            n_gold=n,
            n_unlabeled=0,
            classical_estimate=classical_est,
            classical_std_error=classical_se,
            classical_ci_low=classical_lo,
            classical_ci_high=classical_hi,
            effective_gold_n=float(n),
            confidence_level=confidence_level,
        )

    raw_lambda = _unclamped_lambda(y, f, big_n) if power_tune else 1.0
    lam = float(np.clip(raw_lambda, 0.0, 1.0)) if power_tune else 1.0
    correlation = _correlation(y, f)

    estimate = float(np.mean(y) + lam * (np.mean(f_pool) - np.mean(f)))

    # Var = lambda^2 * Var(f_pool)/N + Var(Y - lambda*f)/n
    var_pool = float(np.var(f_pool, ddof=1)) if big_n > 1 else 0.0
    var_resid = float(np.var(y - lam * f, ddof=1))
    variance = (lam**2) * var_pool / big_n + var_resid / n
    std_error = float(np.sqrt(max(variance, 0.0)))

    # How many gold labels the classical estimator would need to match this SE.
    effective_n = float(var_y / variance) if variance > 0 else float(n)

    return PPIEstimate(
        estimate=estimate,
        std_error=std_error,
        ci_low=estimate - z * std_error,
        ci_high=estimate + z * std_error,
        lambda_weight=float(lam),
        n_gold=n,
        n_unlabeled=big_n,
        classical_estimate=classical_est,
        classical_std_error=classical_se,
        classical_ci_low=classical_lo,
        classical_ci_high=classical_hi,
        effective_gold_n=effective_n,
        confidence_level=confidence_level,
        proxy_correlation=correlation,
        lambda_saturated=bool(power_tune and raw_lambda != lam),
    )


def _unclamped_lambda(
    gold_labels: Sequence[float],
    gold_proxy: Sequence[float],
    n_unlabeled: int,
) -> float:
    """Variance-minimising weight before clamping, so saturation is detectable."""
    y = np.asarray(gold_labels, dtype=float)
    f = np.asarray(gold_proxy, dtype=float)
    n = len(y)
    if n < 2 or n_unlabeled < 1:
        return 0.0
    var_f = float(np.var(f, ddof=1))
    if var_f <= 0:
        return 0.0
    cov_yf = float(np.cov(y, f, ddof=1)[0, 1])
    return cov_yf / (var_f * (1.0 + n / n_unlabeled))


def _correlation(gold_labels: np.ndarray, gold_proxy: np.ndarray) -> float:
    """Pearson correlation on the gold overlap, 0 when either side is constant."""
    if len(gold_labels) < 2:
        return 0.0
    if np.var(gold_labels) <= 0 or np.var(gold_proxy) <= 0:
        return 0.0
    return float(np.corrcoef(gold_labels, gold_proxy)[0, 1])


def ppi_mean_difference(
    gold_base: Sequence[float],
    gold_finetuned: Sequence[float],
    proxy_base: Sequence[float],
    proxy_finetuned: Sequence[float],
    pool_proxy_base: Sequence[float],
    pool_proxy_finetuned: Sequence[float],
    confidence_level: float = 0.95,
    power_tune: bool = True,
) -> PPIEstimate:
    """
    Estimate the base-to-fine-tuned difference, paired within each example.

    Pairing first and estimating the mean of the per-example differences
    removes example difficulty from the variance, which matters far more at
    n=16 than the choice of estimator does.

    Args:
        gold_base: True scores for the base model on gold examples.
        gold_finetuned: True scores for the fine-tuned model, same examples.
        proxy_base: Proxy scores for the base model on those gold examples.
        proxy_finetuned: Proxy scores for the fine-tuned model, same examples.
        pool_proxy_base: Proxy scores for the base model over the pool.
        pool_proxy_finetuned: Proxy scores for the fine-tuned model over the pool.
        confidence_level: Coverage for the returned interval.
        power_tune: Whether to variance-tune lambda.

    Returns:
        A PPIEstimate of the mean difference (fine-tuned minus base). An
        interval excluding zero is a directional claim you can defend.
    """
    gold_diff = np.asarray(gold_finetuned, dtype=float) - np.asarray(gold_base, dtype=float)
    proxy_diff = np.asarray(proxy_finetuned, dtype=float) - np.asarray(proxy_base, dtype=float)
    pool_diff = np.asarray(pool_proxy_finetuned, dtype=float) - np.asarray(
        pool_proxy_base, dtype=float
    )

    return ppi_mean(
        gold_labels=gold_diff,
        gold_proxy=proxy_diff,
        unlabeled_proxy=pool_diff,
        confidence_level=confidence_level,
        power_tune=power_tune,
    )
