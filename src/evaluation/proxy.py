"""
Cheap proxy scoring for prediction-powered evaluation.

Generation is what makes a code-completion eval expensive: every example needs
an autoregressive decode, which is why the held-out set ran to 16 examples
while thousands of curated examples sat unused. Scoring is not expensive.
The likelihood a model assigns to the *reference* completion, given its
prompt, takes a single forward pass and needs no decoding at all.

That gives prediction-powered inference the shape it wants. The expensive,
trustworthy measurement -- real generation quality -- exists on a handful of
examples. The cheap, biased measurement exists on all of them. PPI combines
them and corrects the bias using the overlap, so the estimate describes the
whole pool while staying anchored to what was actually generated.

Scores are emitted as exp(-nll), a per-token likelihood in (0, 1]. The
orientation matters: PPI's tuned weight is clamped to [0, 1] on the assumption
that the proxy rises with quality, as generation metrics do. A raw NLL falls
as quality rises, and would drive the weight to zero -- discarding the pool
while looking like it had simply found no signal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import track

console = Console()

# The assistant turn marker in the ChatML the synthesizer emits.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"
_TURN_END = "<|im_end|>"


@dataclass
class PoolExample:
    """A prompt and its reference completion, drawn from the unlabelled pool."""

    prompt: str
    completion: str


def split_chatml(text: str) -> PoolExample | None:
    """
    Recover the prompt and reference completion from a ChatML training row.

    The curated pool is stored as formatted chat turns rather than as pairs,
    so the assistant turn has to be split back out to be scored.

    Args:
        text: A ChatML-formatted example.

    Returns:
        The split example, or None when there is no assistant turn to score.
    """
    if _ASSISTANT_MARKER not in text:
        return None
    prompt, _, completion = text.partition(_ASSISTANT_MARKER)
    completion = completion.split(_TURN_END)[0]
    if not completion.strip():
        return None
    return PoolExample(prompt=prompt + _ASSISTANT_MARKER, completion=completion)


def load_pool(
    path: str | Path,
    limit: int | None = None,
    task: str | None = None,
) -> list[PoolExample]:
    """
    Load unlabelled examples to be proxy-scored.

    Accepts both shapes the pipeline produces: explicit prompt/completion pairs
    as in the code-completion eval files, and ChatML rows as in the curated
    splits. Rows that carry neither are skipped rather than failing the run --
    a pool is a convenience, and one malformed line should not cost an
    evaluation.

    `task` matters more than it looks. PPI estimates the mean over *the pool*,
    so a pool drawn from a different distribution than the gold set answers a
    different question. The curated splits mix completion, explanation and raw
    rows; leaving them mixed while the gold set is completions-only silently
    changes the estimand.

    Args:
        path: JSONL file to read.
        limit: Stop after this many usable examples. None reads all.
        task: Keep only rows whose `task` field equals this. None keeps all.

    Returns:
        Examples in file order.
    """
    examples: list[PoolExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if task is not None and row.get("task") != task:
                continue

            prompt = row.get("prompt", row.get("input"))
            completion = row.get("completion", row.get("output"))
            if prompt and completion:
                examples.append(PoolExample(prompt=prompt, completion=completion))
            elif "text" in row:
                split = split_chatml(row["text"])
                if split is not None:
                    examples.append(split)

            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_scoring_inputs(
    prompt_ids: list[int],
    completion_ids: list[int],
    max_length: int,
    prefix_tokens: int | None = None,
) -> tuple[list[int], list[int]] | None:
    """
    Assemble token ids and labels for teacher-forced scoring of a completion.

    Prompt tokens are masked with -100 so the loss covers only the completion;
    scoring the prompt too would mostly measure how predictable the instruction
    template is, which is identical across models.

    Tokenising the prompt and completion separately and concatenating avoids
    the boundary problem: re-tokenising the joined string can merge the last
    prompt character with the first completion character into one token, which
    silently shifts the mask by one position.

    Over-long inputs are truncated from the front, so the completion being
    scored always survives intact.

    Args:
        prompt_ids: Token ids of the prompt.
        completion_ids: Token ids of the reference completion.
        max_length: Maximum combined length.
        prefix_tokens: Score only the first N completion tokens. None scores
            all of them.

    Returns:
        (input_ids, labels), or None when there is nothing scoreable -- an
        empty completion, or a completion alone longer than max_length.
    """
    if not completion_ids or max_length <= 0:
        return None
    if prefix_tokens is not None:
        if prefix_tokens <= 0:
            return None
        completion_ids = completion_ids[:prefix_tokens]
    if len(completion_ids) >= max_length:
        return None

    budget = max_length - len(completion_ids)
    kept_prompt = prompt_ids[-budget:] if budget > 0 else []

    input_ids = list(kept_prompt) + list(completion_ids)
    labels = [-100] * len(kept_prompt) + list(completion_ids)
    return input_ids, labels


def likelihood_from_nll(nll: float) -> float:
    """
    Convert a mean negative log-likelihood to a per-token likelihood in (0, 1].

    Rises with quality, which is the orientation PPI's tuned weight assumes.
    """
    if not math.isfinite(nll):
        return 0.0
    return math.exp(-max(nll, 0.0))


class CompletionLikelihoodScorer:
    """
    Scores reference completions by teacher-forced likelihood.

    One forward pass per example and no decoding, so a pool of thousands costs
    roughly what generating on dozens does.
    """

    def __init__(self, max_length: int = 1024, prefix_tokens: int | None = None):
        self.max_length = max_length
        # Scoring only the opening tokens measures whether the model starts the
        # way the reference starts -- which is what format compliance is.
        # Teacher-forcing the whole reference cannot see that, because it
        # supplies the very tokens whose production is in question.
        self.prefix_tokens = prefix_tokens

    def score_one(self, model, tokenizer, example: PoolExample) -> float | None:
        """Return the per-token likelihood of one reference completion."""
        prompt_ids = tokenizer(example.prompt, add_special_tokens=False).input_ids
        completion_ids = tokenizer(example.completion, add_special_tokens=False).input_ids

        built = build_scoring_inputs(
            prompt_ids, completion_ids, self.max_length, self.prefix_tokens
        )
        if built is None:
            return None
        input_ids, labels = built

        input_tensor = torch.tensor([input_ids], device=model.device)
        label_tensor = torch.tensor([labels], device=model.device)
        with torch.no_grad():
            outputs = model(input_tensor, labels=label_tensor)
        return likelihood_from_nll(float(outputs.loss.item()))

    def score(
        self,
        model,
        tokenizer,
        examples: list[PoolExample],
        description: str = "  Proxy-scoring pool...",
    ) -> list[float | None]:
        """
        Score every example, preserving position.

        Unscoreable examples yield None rather than being dropped, so the two
        models' score lists stay aligned by index and can be paired later.
        """
        return [
            self.score_one(model, tokenizer, example)
            for example in track(examples, description=description, console=console)
        ]


def align_proxy_scores(
    base_scores: list[float | None],
    finetuned_scores: list[float | None],
) -> tuple[list[float], list[float]]:
    """
    Keep only positions both models scored.

    A completion too long for one model's window is too long for the other, so
    this normally drops nothing; it guards the case where it does.

    Args:
        base_scores: Base model proxy scores, position-aligned.
        finetuned_scores: Fine-tuned model proxy scores, same positions.

    Returns:
        (base, finetuned) with unscoreable positions removed from both.

    Raises:
        ValueError: If the two lists are not the same length.
    """
    if len(base_scores) != len(finetuned_scores):
        raise ValueError(
            f"proxy score lists must align: got {len(base_scores)} and "
            f"{len(finetuned_scores)}"
        )
    kept = [
        (b, f)
        for b, f in zip(base_scores, finetuned_scores)
        if b is not None and f is not None
    ]
    if not kept:
        return [], []
    base, finetuned = zip(*kept)
    return list(base), list(finetuned)


def summarize_pool(
    examples: list[PoolExample],
    base_scores: list[float | None],
    finetuned_scores: list[float | None],
) -> dict[str, Any]:
    """Report how much of the pool was usable, for the evaluation record."""
    base, _ = align_proxy_scores(base_scores, finetuned_scores)
    return {
        "pool_size": len(examples),
        "scored": len(base),
        "unscoreable": len(examples) - len(base),
    }


def align_gold_with_proxy(
    base_raw: list[dict[str, Any]],
    finetuned_raw: list[dict[str, Any]],
    gold_examples: list[PoolExample],
    gold_proxy_base: list[float | None],
    gold_proxy_finetuned: list[float | None],
    metric: str = "token_f1",
) -> dict[str, list[float]] | None:
    """
    Line up generated-metric values with proxy scores on the same examples.

    PPI's correction is measured on examples where both the expensive and the
    cheap measurement exist. Matching on the prompt rather than on position
    keeps that overlap correct even when a run skipped an example.

    Args:
        base_raw: Per-example generation results for the base model.
        finetuned_raw: Per-example generation results for the fine-tuned model.
        gold_examples: The examples proxy-scoring was run over, in order.
        gold_proxy_base: Base model proxy scores, aligned to gold_examples.
        gold_proxy_finetuned: Fine-tuned proxy scores, aligned to gold_examples.
        metric: Which generated metric to treat as the trustworthy label.

    Returns:
        Four equal-length lists keyed `gold_base`, `gold_finetuned`,
        `proxy_base` and `proxy_finetuned`, or None when fewer than two
        examples carry both measurements.
    """
    base_by_prompt = {str(row.get("prompt")): row for row in base_raw}
    ft_by_prompt = {str(row.get("prompt")): row for row in finetuned_raw}

    aligned: dict[str, list[float]] = {
        "gold_base": [],
        "gold_finetuned": [],
        "proxy_base": [],
        "proxy_finetuned": [],
    }

    for example, proxy_b, proxy_f in zip(gold_examples, gold_proxy_base, gold_proxy_finetuned):
        if proxy_b is None or proxy_f is None:
            continue
        key = str(example.prompt)
        base_row, ft_row = base_by_prompt.get(key), ft_by_prompt.get(key)
        if base_row is None or ft_row is None:
            continue
        if metric not in base_row or metric not in ft_row:
            continue
        aligned["gold_base"].append(float(base_row[metric]))
        aligned["gold_finetuned"].append(float(ft_row[metric]))
        aligned["proxy_base"].append(float(proxy_b))
        aligned["proxy_finetuned"].append(float(proxy_f))

    return aligned if len(aligned["gold_base"]) >= 2 else None
