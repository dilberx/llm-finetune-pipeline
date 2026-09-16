"""
Item response theory calibration for evaluation sets.

Psychometrics has spent a century on a problem this pipeline also has: score a
subject on a trait using as few questions as possible, given that questions
differ in difficulty and in how sharply they separate one subject from another.
Swap "subject" for "model checkpoint" and "question" for "held-out example"
and the machinery transfers unchanged.

The two-parameter logistic model gives each item a difficulty and a
discrimination:

    P(correct | theta) = sigmoid(a_j * (theta_i - b_j))

theta_i is a model's latent ability, b_j how hard item j is, and a_j how
sharply item j separates strong models from weak ones. Items with near-zero
discrimination contribute nothing but wall-clock time, and items every model
gets identically right or wrong contribute nothing at all -- they are not hard
or easy, they are simply not measuring.

Fitting needs a *population* of models. Two is not enough to identify anything.
Intermediate training checkpoints count, and are free: a run that saved every
2000 steps has already produced the population this needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Below this many models the 2PL parameters are not identifiable in practice.
MIN_MODELS_FOR_IRT = 5


@dataclass
class IRTFit:
    """A fitted 2PL model over an evaluation set."""

    item_ids: list[str]
    model_ids: list[str]
    difficulty: np.ndarray
    discrimination: np.ndarray
    ability: np.ndarray
    degenerate_items: list[str] = field(default_factory=list)
    log_likelihood: float = 0.0
    n_iterations: int = 0
    converged: bool = False

    def information(self, theta: float) -> np.ndarray:
        """
        Fisher information contributed by each fitted item at ability `theta`.

        I_j(theta) = a_j^2 * P * (1 - P)

        Information peaks where an item's difficulty matches the ability being
        measured, which is why a set calibrated against weak models measures
        strong ones poorly.
        """
        p = _sigmoid(self.discrimination * (theta - self.difficulty))
        return (self.discrimination**2) * p * (1.0 - p)

    def select_items(
        self,
        k: int,
        theta_grid: Sequence[float] | None = None,
    ) -> list[str]:
        """
        Choose the `k` most informative items across an ability range.

        Greedy on total Fisher information summed over the grid. Use the grid
        to say which models the set has to distinguish: a narrow grid around 0
        tunes for models near the population average, a wide one keeps the set
        useful as checkpoints improve.

        Args:
            k: How many items to keep.
            theta_grid: Ability points to integrate information over. Defaults
                to the fitted abilities, which targets the models you have.

        Returns:
            Item ids, most informative first.
        """
        if k <= 0:
            return []
        if theta_grid is None:
            theta_grid = self.ability if len(self.ability) else np.array([0.0])

        total = np.zeros(len(self.item_ids), dtype=float)
        for theta in theta_grid:
            total += self.information(float(theta))

        order = np.argsort(-total)[: min(k, len(self.item_ids))]
        return [self.item_ids[i] for i in order]

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "item_id": item,
                    "difficulty": float(b),
                    "discrimination": float(a),
                }
                for item, b, a in zip(self.item_ids, self.difficulty, self.discrimination)
            ],
            "models": [
                {"model_id": m, "ability": float(t)}
                for m, t in zip(self.model_ids, self.ability)
            ],
            "degenerate_items": self.degenerate_items,
            "n_degenerate": len(self.degenerate_items),
            "log_likelihood": self.log_likelihood,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def find_degenerate_items(
    responses: np.ndarray,
    item_ids: Sequence[str],
) -> list[str]:
    """
    Identify items every model scores identically.

    These carry zero information about model ability regardless of how many
    models you run. An evaluation set that is mostly degenerate cannot detect
    a change no matter how large the change is -- a measurement failure that
    looks exactly like a null result.

    Args:
        responses: Binary matrix, models on rows and items on columns.
        item_ids: Column labels.

    Returns:
        Ids of items with no variance across models.
    """
    if responses.ndim != 2:
        raise ValueError(f"responses must be 2-D (models x items), got {responses.ndim}-D")
    constant = responses.min(axis=0) == responses.max(axis=0)
    return [item_ids[j] for j in np.flatnonzero(constant)]


def fit_2pl(
    responses: np.ndarray | Sequence[Sequence[float]],
    item_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
    max_iterations: int = 2000,
    learning_rate: float = 0.05,
    tolerance: float = 1e-6,
    prior_strength: float = 0.1,
) -> IRTFit:
    """
    Fit a 2PL item response model by joint maximum likelihood.

    Abilities are standardised on every step to fix the location and scale that
    the 2PL leaves free. Weak priors pull discrimination toward 1 and difficulty
    toward 0, which keeps the fit stable at the small model counts available
    here -- without them a handful of items run away to extreme parameters.

    Degenerate items are dropped before fitting and reported separately; they
    would otherwise push their discrimination to zero and drag the optimiser.

    Args:
        responses: Binary matrix, models on rows and items on columns.
        item_ids: Column labels. Defaults to item_0, item_1, ...
        model_ids: Row labels. Defaults to model_0, model_1, ...
        max_iterations: Gradient ascent cap.
        learning_rate: Step size for gradient ascent.
        tolerance: Stop when the log-likelihood improves by less than this.
        prior_strength: Weight on the regularising priors. 0 disables them.

    Returns:
        An IRTFit over the non-degenerate items.

    Raises:
        ValueError: If fewer than MIN_MODELS_FOR_IRT models are supplied, or if
            every item is degenerate.
    """
    r_full = np.asarray(responses, dtype=float)
    if r_full.ndim != 2:
        raise ValueError(f"responses must be 2-D (models x items), got {r_full.ndim}-D")

    n_models, n_items_full = r_full.shape
    items_full = (
        list(item_ids) if item_ids is not None else [f"item_{j}" for j in range(n_items_full)]
    )
    models = (
        list(model_ids) if model_ids is not None else [f"model_{i}" for i in range(n_models)]
    )
    if len(items_full) != n_items_full:
        raise ValueError(
            f"item_ids length {len(items_full)} does not match {n_items_full} columns"
        )
    if len(models) != n_models:
        raise ValueError(f"model_ids length {len(models)} does not match {n_models} rows")

    if n_models < MIN_MODELS_FOR_IRT:
        raise ValueError(
            f"2PL needs at least {MIN_MODELS_FOR_IRT} models to be identifiable, got "
            f"{n_models}. Intermediate training checkpoints count as models -- "
            f"scoring checkpoint-2000, -4000, ... gives a usable population for free."
        )

    degenerate = find_degenerate_items(r_full, items_full)
    keep = [j for j, item in enumerate(items_full) if item not in set(degenerate)]
    if not keep:
        raise ValueError(
            f"all {n_items_full} items are degenerate: every model scored them "
            f"identically, so the set carries no information about model ability"
        )

    r = r_full[:, keep]
    items = [items_full[j] for j in keep]
    n_items = len(items)

    # Initialise difficulty from item pass rates, the classical starting point.
    pass_rate = np.clip(r.mean(axis=0), 0.01, 0.99)
    b = -np.log(pass_rate / (1.0 - pass_rate))
    a = np.ones(n_items, dtype=float)
    score = np.clip(r.mean(axis=1), 0.01, 0.99)
    theta = np.log(score / (1.0 - score))
    theta = _standardise(theta)

    prev_ll = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        resid = r - p

        grad_theta = (resid * a[None, :]).sum(axis=1)
        grad_b = -(resid * a[None, :]).sum(axis=0)
        grad_a = (resid * (theta[:, None] - b[None, :])).sum(axis=0)

        if prior_strength > 0:
            grad_b -= prior_strength * b
            grad_a -= prior_strength * (a - 1.0)

        theta = _standardise(theta + learning_rate * grad_theta / max(n_items, 1))
        b = b + learning_rate * grad_b / max(n_models, 1)
        # Discrimination must stay positive; a negative item is a mis-keyed one.
        a = np.clip(a + learning_rate * grad_a / max(n_models, 1), 0.01, 10.0)

        ll = _log_likelihood(r, theta, a, b)
        if abs(ll - prev_ll) < tolerance:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    return IRTFit(
        item_ids=items,
        model_ids=models,
        difficulty=b,
        discrimination=a,
        ability=theta,
        degenerate_items=degenerate,
        log_likelihood=float(prev_ll),
        n_iterations=iteration,
        converged=converged,
    )


def _standardise(theta: np.ndarray) -> np.ndarray:
    """Fix the 2PL's free location and scale by centring and scaling abilities."""
    centred = theta - theta.mean()
    spread = centred.std()
    if spread < 1e-8:
        return centred
    return centred / spread


def _log_likelihood(
    responses: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    p = np.clip(_sigmoid(a[None, :] * (theta[:, None] - b[None, :])), 1e-9, 1 - 1e-9)
    return float((responses * np.log(p) + (1 - responses) * np.log(1 - p)).sum())


def responses_from_raw_results(
    raw_results_by_model: dict[str, list[dict[str, Any]]],
    metric: str = "exact_match",
    key: str = "prompt",
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build an IRT response matrix from benchmark `raw_results` payloads.

    Only items present for every model are kept, so the matrix stays complete.

    Args:
        raw_results_by_model: Model id to that model's per-example results, as
            returned under `raw_results` by CodeCompletionBenchmark.
        metric: Which per-example field to treat as the binary response.
        key: Field identifying an example across models.

    Returns:
        (responses, item_ids, model_ids) ready for `fit_2pl`.

    Raises:
        ValueError: If no models are supplied, or no item is shared by all.
    """
    if not raw_results_by_model:
        raise ValueError("raw_results_by_model is empty")

    models = list(raw_results_by_model.keys())
    per_model: dict[str, dict[str, float]] = {}
    for model in models:
        per_model[model] = {
            str(row[key]): float(bool(row.get(metric, 0)))
            for row in raw_results_by_model[model]
            if key in row
        }

    shared = set.intersection(*(set(d.keys()) for d in per_model.values()))
    if not shared:
        raise ValueError(f"no examples shared across all {len(models)} models on key '{key}'")

    items = sorted(shared)
    responses = np.array(
        [[per_model[m][item] for item in items] for m in models], dtype=float
    )
    return responses, items, models
