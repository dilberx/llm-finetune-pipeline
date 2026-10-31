"""
Benchmark implementations for model evaluation.

Includes:
- HumanEval: code generation pass@k
- Perplexity: language modeling quality
- Code Completion: custom eval on held-out code
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import track

from .code_equivalence import alpha_equivalent, ast_equivalent, is_parseable

console = Console()


# ============================================================================
# HumanEval Benchmark
# ============================================================================

def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from n samples with c correct.

    Uses the formula from the Codex paper (Chen et al., 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# Minimal HumanEval problems for standalone evaluation
# (In production, you'd load the full dataset from openai/human-eval)
HUMANEVAL_SAMPLES = [
    {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": 'from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n',
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(()))(()()(()))((()))') == ['(()(()))', '(()()(()))', '((()))']\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Given a positive floating point number, it can be decomposed into an integer part\n    and decimals (the decimal part is always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n",
        "entry_point": "truncate_number",
    },
]


class HumanEvalBenchmark:
    """
    HumanEval code generation benchmark.

    Generates multiple completions per problem and estimates pass@k
    using the unbiased estimator from the Codex paper.
    """

    def __init__(
        self,
        num_samples: int = 10,
        k_values: list[int] | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ):
        self.num_samples = num_samples
        self.k_values = k_values or [1, 5, 10]
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def _generate_completions(self, model, tokenizer, prompt: str) -> list[str]:
        """Generate multiple completions for a prompt."""
        completions = []
        for _ in range(self.num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Truncate at the first function definition or class (stop at next function)
            lines = completion.split("\n")
            truncated = []
            for line in lines:
                if line.strip().startswith(("def ", "class ")) and truncated:
                    break
                truncated.append(line)
            completions.append("\n".join(truncated))

        return completions

    def _check_completion(self, prompt: str, completion: str, test: str, entry_point: str) -> bool:
        """Execute a completion against test cases to check correctness."""
        full_code = prompt + completion + "\n\n" + test + f"\ncheck({entry_point})\n"

        try:
            result = subprocess.run(
                ["python", "-c", full_code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def run(self, model, tokenizer) -> dict[str, Any]:
        """Run HumanEval benchmark and return pass@k metrics."""
        problems = HUMANEVAL_SAMPLES  # Use embedded samples
        all_pass_at_k = {k: [] for k in self.k_values}

        for problem in track(problems, description="  HumanEval problems...", console=console):
            completions = self._generate_completions(model, tokenizer, problem["prompt"])

            # Check each completion
            num_correct = sum(
                self._check_completion(
                    problem["prompt"], comp, problem["test"], problem["entry_point"]
                )
                for comp in completions
            )

            # Estimate pass@k for each k
            for k in self.k_values:
                if k <= self.num_samples:
                    pass_k = _estimate_pass_at_k(self.num_samples, num_correct, k)
                    all_pass_at_k[k].append(pass_k)

        # Average across problems
        metrics = {}
        for k in self.k_values:
            if all_pass_at_k[k]:
                metrics[f"pass@{k}"] = float(np.mean(all_pass_at_k[k]))

        return {"metrics": metrics, "num_problems": len(problems)}


# ============================================================================
# Perplexity Benchmark
# ============================================================================

class PerplexityBenchmark:
    """
    Perplexity evaluation using sliding window approach.

    Lower perplexity = better language modeling quality.
    """

    def __init__(self, dataset: str = "wikitext", stride: int = 512, max_length: int = 2048):
        self.dataset_name = dataset
        self.stride = stride
        self.max_length = max_length

    def run(self, model, tokenizer) -> dict[str, Any]:
        """Compute perplexity on the evaluation dataset."""
        from datasets import load_dataset

        # Load evaluation text
        if self.dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(dataset["text"])
        else:
            # Custom dataset path
            with open(self.dataset_name) as f:
                text = f.read()

        encodings = tokenizer(text, return_tensors="pt", verbose=False)
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end = 0

        for begin in track(
            range(0, seq_len, self.stride),
            description="  Computing perplexity...",
            console=console,
        ):
            end = min(begin + self.max_length, seq_len)
            target_len = end - prev_end

            input_ids = encodings.input_ids[:, begin:end].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-target_len] = -100  # Mask context tokens

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood.item())
            prev_end = end

            if end == seq_len:
                break

        perplexity = math.exp(sum(nlls) / len(nlls))

        return {
            "metrics": {
                "perplexity": perplexity,
                "avg_nll": sum(nlls) / len(nlls),
            },
            "num_tokens": seq_len,
        }


# ============================================================================
# Code Completion Benchmark
# ============================================================================

class CodeCompletionBenchmark:
    """
    Custom code completion evaluation on held-out examples.

    Scores each completion on a ladder of strictness, from character equality
    up to structural equivalence of the parse tree. The looser rungs exist
    because exact match saturates at zero on small sets: a model can be
    reliably close and still never reproduce a reference character for
    character, which reads as a null result when it is really a ceiling in the
    metric.

    Also reports `parse_rate`, separating wrong code from unparseable code, and
    `reference_parse_rate`, which describes the eval set rather than the model.
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        batch_size: int = 1,
    ):
        self.dataset_path = dataset_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Generation dominates the cost of this benchmark, and decoding one
        # sequence at a time leaves a 10 GB card almost idle: measured on one,
        # eight prompts per call raised throughput roughly 30-50x. The default
        # of 1 reproduces the original one-at-a-time behaviour.
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _batches_by_length(prompts: list[str], batch_size: int) -> list[list[int]]:
        """
        Group prompt indices into generation batches.

        Prompts are ordered shortest-first when batching, so each batch pads to
        a similar length instead of to its longest outlier. Indices, not
        prompts, are returned so the caller can put results back in the
        original order. With a batch size of 1 the original order is kept.
        """
        order = list(range(len(prompts)))
        if batch_size > 1:
            order.sort(key=lambda i: len(prompts[i]))
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def _generate_batch(self, model, tokenizer, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt in a single call."""
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        sampling = self.temperature > 0
        gen_kwargs = {"do_sample": sampling}
        if sampling:
            # Passing a temperature with sampling off is flagged by HF, and a
            # temperature of 0 is not a valid sampling temperature.
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        # Prompts are left-padded, so every row's new tokens start at the same
        # column. Right padding would put them after a run of pad tokens.
        start = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(row[start:], skip_special_tokens=True).strip()
            for row in outputs
        ]

    def _generate_all(self, model, tokenizer, prompts: list[str]) -> list[str]:
        """Generate a completion for every prompt, returned in the input order."""
        generated = [""] * len(prompts)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            batches = self._batches_by_length(prompts, self.batch_size)
            for indices in track(batches, description="  Code completions...", console=console):
                texts = self._generate_batch(model, tokenizer, [prompts[i] for i in indices])
                for index, text in zip(indices, texts):
                    generated[index] = text
        finally:
            tokenizer.padding_side = original_side
        return generated

    @staticmethod
    def _normalize_completion(text: str) -> str:
        """Normalize whitespace without changing code structure."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return "\n".join(line.rstrip() for line in lines).strip()

    @staticmethod
    def _tokenize_for_similarity(text: str) -> list[str]:
        """Tokenize code-like text while keeping punctuation as separate tokens."""
        return re.findall(r"\w+|[^\w\s]", text)

    def _compute_token_overlap(self, generated: str, reference: str) -> float:
        """Compute multiset token-level F1 between generated and reference text."""
        gen_tokens = Counter(self._tokenize_for_similarity(generated))
        ref_tokens = Counter(self._tokenize_for_similarity(reference))

        if not gen_tokens or not ref_tokens:
            return 0.0

        overlap = gen_tokens & ref_tokens
        overlap_count = sum(overlap.values())
        gen_count = sum(gen_tokens.values())
        ref_count = sum(ref_tokens.values())

        precision = overlap_count / gen_count if gen_count else 0
        recall = overlap_count / ref_count if ref_count else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _normalized_exact_match(self, generated: str, reference: str) -> bool:
        """Return whether generated text matches after conservative whitespace cleanup."""
        return self._normalize_completion(generated) == self._normalize_completion(reference)

    def _edit_similarity(self, generated: str, reference: str) -> float:
        """Return SequenceMatcher ratio after normalization."""
        return SequenceMatcher(
            None,
            self._normalize_completion(generated),
            self._normalize_completion(reference),
        ).ratio()

    def run(self, model, tokenizer) -> dict[str, Any]:
        """Run code completion evaluation."""
        if not self.dataset_path or not Path(self.dataset_path).exists():
            console.print("  [yellow]No code completion dataset found, skipping[/yellow]")
            return {"metrics": {}, "skipped": True}

        # Load eval examples
        examples = []
        with open(self.dataset_path) as f:
            for line in f:
                examples.append(json.loads(line))

        exact_matches = 0
        normalized_exact_matches = 0
        ast_matches = 0
        alpha_matches = 0
        parseable_generations = 0
        parseable_references = 0
        f1_scores = []
        edit_scores = []
        raw_results = []

        prompts = [ex.get("prompt", ex.get("input", "")) for ex in examples]
        generations = self._generate_all(model, tokenizer, prompts)

        for ex, prompt, generated in zip(examples, prompts, generations):
            reference = ex.get("completion", ex.get("output", ""))

            exact_match = generated.strip() == reference.strip()
            if exact_match:
                exact_matches += 1

            normalized_exact = self._normalized_exact_match(generated, reference)
            if normalized_exact:
                normalized_exact_matches += 1

            # Structural comparison: fences stripped and fragments parsed, so
            # naming and formatting stop masquerading as correctness errors.
            ast_match = ast_equivalent(generated, reference)
            alpha_match = alpha_equivalent(generated, reference)
            if ast_match:
                ast_matches += 1
            if alpha_match:
                alpha_matches += 1

            generation_parses = is_parseable(generated)
            if generation_parses:
                parseable_generations += 1
            if is_parseable(reference):
                parseable_references += 1

            token_f1 = self._compute_token_overlap(generated, reference)
            edit_similarity = self._edit_similarity(generated, reference)
            f1_scores.append(token_f1)
            edit_scores.append(edit_similarity)
            raw_results.append({
                "prompt": prompt,
                "reference": reference,
                "generated": generated,
                "exact_match": exact_match,
                "normalized_exact_match": normalized_exact,
                "ast_match": ast_match,
                "ast_alpha_match": alpha_match,
                "parses": generation_parses,
                "token_f1": token_f1,
                "edit_similarity": edit_similarity,
            })

        n = len(examples)
        return {
            "metrics": {
                "exact_match": exact_matches / n if n > 0 else 0,
                "normalized_exact_match": normalized_exact_matches / n if n > 0 else 0,
                "ast_match": ast_matches / n if n > 0 else 0,
                "ast_alpha_match": alpha_matches / n if n > 0 else 0,
                "parse_rate": parseable_generations / n if n > 0 else 0,
                "token_f1": float(np.mean(f1_scores)) if f1_scores else 0,
                "edit_similarity": float(np.mean(edit_scores)) if edit_scores else 0,
            },
            # A dataset property, not a model score: it is identical for every
            # model and flags a broken eval set rather than a weak model.
            "reference_parse_rate": parseable_references / n if n > 0 else 0,
            "num_examples": n,
            "raw_results": raw_results,
        }
