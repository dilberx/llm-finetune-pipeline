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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import track

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

        encodings = tokenizer(text, return_tensors="pt")
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

    Measures exact match and BLEU-like similarity between
    generated and reference completions.
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ):
        self.dataset_path = dataset_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _compute_token_overlap(self, generated: str, reference: str) -> float:
        """Simple token-level F1 score between generated and reference."""
        gen_tokens = set(generated.split())
        ref_tokens = set(reference.split())

        if not gen_tokens or not ref_tokens:
            return 0.0

        overlap = gen_tokens & ref_tokens
        precision = len(overlap) / len(gen_tokens) if gen_tokens else 0
        recall = len(overlap) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

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
        f1_scores = []

        for ex in track(examples, description="  Code completions...", console=console):
            prompt = ex.get("prompt", ex.get("input", ""))
            reference = ex.get("completion", ex.get("output", ""))

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Exact match
            if generated.strip() == reference.strip():
                exact_matches += 1

            # Token F1
            f1_scores.append(self._compute_token_overlap(generated, reference))

        n = len(examples)
        return {
            "metrics": {
                "exact_match": exact_matches / n if n > 0 else 0,
                "token_f1": float(np.mean(f1_scores)) if f1_scores else 0,
            },
            "num_examples": n,
        }
