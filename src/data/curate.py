"""
Data curation pipeline for code model fine-tuning.

Orchestrates the full data pipeline:
1. Ingest raw code from HuggingFace datasets
2. Quality filtering (AST validation, length, comment ratio)
3. MinHash deduplication
4. Instruction synthesis (convert raw code → instruction-following format)
5. Output clean JSONL for training

Designed for efficient processing of large datasets using streaming
and batch processing with Polars.
"""

from __future__ import annotations

import ast
import json
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import polars as pl
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .dedup import MinHashDeduplicator
from .synthesize import InstructionSynthesizer, validate_example

console = Console()


# ============================================================================
# Quality Filters
# ============================================================================

def is_ast_valid(code: str) -> bool:
    """Check if Python code is syntactically valid by attempting AST parse."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def compute_alpha_ratio(text: str) -> float:
    """Compute the ratio of alphanumeric characters to total characters."""
    if not text:
        return 0.0
    alpha_count = sum(1 for c in text if c.isalnum())
    return alpha_count / len(text)


def compute_comment_ratio(code: str) -> float:
    """
    Estimate the ratio of comment lines to total non-empty lines.

    Uses a heuristic approach (lines starting with #) rather than
    full AST analysis for speed on large datasets.
    """
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if not lines:
        return 0.0
    comment_lines = sum(1 for line in lines if line.startswith("#"))
    return comment_lines / len(lines)


def max_line_length(code: str) -> int:
    """Return the length of the longest line in the code."""
    if not code:
        return 0
    return max(len(line) for line in code.splitlines())


class QualityFilter:
    """
    Applies configurable quality filters to code samples.

    Filters include:
    - Line count bounds (min/max)
    - AST validity check
    - Alphanumeric ratio threshold
    - Maximum single-line length
    - Comment-to-code ratio cap
    """

    def __init__(
        self,
        min_lines: int = 5,
        max_lines: int = 500,
        ast_valid: bool = True,
        min_alpha_ratio: float = 0.25,
        max_line_len: int = 200,
        max_comment_ratio: float = 0.6,
    ):
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.check_ast = ast_valid
        self.min_alpha_ratio = min_alpha_ratio
        self.max_line_len = max_line_len
        self.max_comment_ratio = max_comment_ratio

        # Track filter statistics
        self.stats = {
            "total": 0,
            "passed": 0,
            "failed_line_count": 0,
            "failed_ast": 0,
            "failed_alpha_ratio": 0,
            "failed_line_length": 0,
            "failed_comment_ratio": 0,
        }

    def __call__(self, code: str) -> bool:
        """Return True if the code sample passes all quality filters."""
        self.stats["total"] += 1

        lines = code.splitlines()
        line_count = len(lines)

        # Line count filter
        if line_count < self.min_lines or line_count > self.max_lines:
            self.stats["failed_line_count"] += 1
            return False

        # AST validity filter
        if self.check_ast and not is_ast_valid(code):
            self.stats["failed_ast"] += 1
            return False

        # Alphanumeric ratio filter
        if compute_alpha_ratio(code) < self.min_alpha_ratio:
            self.stats["failed_alpha_ratio"] += 1
            return False

        # Max line length filter
        if max_line_length(code) > self.max_line_len:
            self.stats["failed_line_length"] += 1
            return False

        # Comment ratio filter
        if compute_comment_ratio(code) > self.max_comment_ratio:
            self.stats["failed_comment_ratio"] += 1
            return False

        self.stats["passed"] += 1
        return True

    def report(self) -> dict[str, Any]:
        """Return filter statistics as a dict."""
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / max(self.stats["total"], 1),
        }


# ============================================================================
# Data Pipeline
# ============================================================================

class DataCurationPipeline:
    """
    End-to-end data curation pipeline for code model training.

    Stages:
    1. Ingest: Load raw code from HuggingFace datasets (streaming)
    2. Filter: Apply quality filters
    3. Deduplicate: MinHash-based near-duplicate removal
    4. Synthesize: Convert to instruction-following format
    5. Export: Write clean JSONL splits
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.filter_config = config.get("filters", {})
        self.dedup_config = config.get("deduplication", {})
        self.synthesis_config = config.get("synthesis", {})
        self.output_config = config.get("output", {})
        self.processing_config = config.get("processing", {})

        # Initialize components
        self.quality_filter = QualityFilter(
            min_lines=self.filter_config.get("min_lines", 5),
            max_lines=self.filter_config.get("max_lines", 500),
            ast_valid=self.filter_config.get("ast_valid", True),
            min_alpha_ratio=self.filter_config.get("min_alpha_ratio", 0.25),
            max_line_len=self.filter_config.get("max_line_length", 200),
            max_comment_ratio=self.filter_config.get("max_comment_ratio", 0.6),
        )

        if self.dedup_config.get("enabled", True):
            self.deduplicator = MinHashDeduplicator(
                num_perm=self.dedup_config.get("num_perm", 128),
                threshold=self.dedup_config.get("threshold", 0.7),
                ngram_size=self.dedup_config.get("ngram_size", 5),
            )
        else:
            self.deduplicator = None

        if self.synthesis_config.get("enabled", True):
            self.synthesizer = InstructionSynthesizer(
                templates=self.synthesis_config.get("templates", {}),
            )
        else:
            self.synthesizer = None

        # Synthesis can silently emit unparseable examples when a template and
        # the indentation of its source disagree. Validating here turns that
        # into a failed run rather than a model trained on broken code.
        self.validate_synthesis = self.synthesis_config.get("validate", True)
        self.max_invalid_ratio = self.synthesis_config.get("max_invalid_ratio", 0.02)
        self.synthesis_stats: dict[str, Any] = {}

    def _iter_sources(self, sources: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield raw code samples from a list of HuggingFace dataset sources."""
        for source in sources:
            dataset_name = source["name"]
            subset = source.get("subset")
            split = source.get("split", "train")
            max_samples = source.get("max_samples")
            streaming = source.get("streaming", True)

            console.print(f"\n[bold blue]Loading dataset:[/bold blue] {dataset_name}")
            if subset:
                console.print(f"  Subset: {subset}, Split: {split}")

            try:
                ds = load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    streaming=streaming,
                )
            except Exception as e:
                console.print(f"[red]Failed to load {dataset_name}: {e}[/red]")
                continue

            count = 0
            for sample in ds:
                if max_samples and count >= max_samples:
                    break
                # Extract code content — handle different dataset schemas
                content = (
                    sample.get("content")
                    or sample.get("code")
                    or sample.get("text")
                    or ""
                )
                if content:
                    yield {"content": content, "source": dataset_name}
                    count += 1

            console.print(f"  [green]Loaded {count} samples[/green]")

    def _ingest_sources(self) -> Iterator[dict[str, Any]]:
        """Load raw code samples from configured primary sources or fallbacks."""
        yielded = 0
        for sample in self._iter_sources(self.config.get("sources", [])):
            yielded += 1
            yield sample

        if yielded == 0 and self.config.get("fallback_sources"):
            console.print("\n[yellow]No primary source samples loaded; trying fallbacks...[/yellow]")
            for sample in self._iter_sources(self.config["fallback_sources"]):
                yield sample

    def _filter_batch(self, samples: list[dict]) -> list[dict]:
        """Apply quality filters to a batch of samples."""
        return [s for s in samples if self.quality_filter(s["content"])]

    def _deduplicate(self, samples: list[dict]) -> list[dict]:
        """Remove near-duplicate samples using MinHash LSH."""
        if self.deduplicator is None:
            return samples

        console.print("\n[bold blue]Deduplicating samples...[/bold blue]")
        texts = [s["content"] for s in samples]
        unique_indices = self.deduplicator.deduplicate(texts)

        removed = len(samples) - len(unique_indices)
        console.print(f"  Removed {removed} near-duplicates ({removed/max(len(samples),1)*100:.1f}%)")

        return [samples[i] for i in unique_indices]

    def _synthesize(self, samples: list[dict]) -> list[dict]:
        """Convert raw code samples to instruction-following format."""
        if self.synthesizer is None:
            # Return raw format if synthesis is disabled
            return [{"text": s["content"]} for s in samples]

        console.print("\n[bold blue]Synthesizing instruction examples...[/bold blue]")
        synthesized = []
        rejected: Counter = Counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Synthesizing...", total=len(samples))

            for sample in samples:
                examples = self.synthesizer.synthesize(sample["content"])
                if self.validate_synthesis:
                    for example in examples:
                        if validate_example(example):
                            synthesized.append(example)
                        else:
                            rejected[example.get("task", "raw")] += 1
                else:
                    synthesized.extend(examples)
                progress.update(task, advance=1)

        console.print(
            f"  Generated {len(synthesized)} instruction examples from {len(samples)} samples"
        )

        if self.validate_synthesis:
            self._report_synthesis_validity(len(synthesized), rejected)

        return synthesized

    def _report_synthesis_validity(self, kept: int, rejected: Counter) -> None:
        """
        Report syntactic validity of synthesized examples and fail a bad run.

        Rejecting a few examples is normal: real corpora contain code this
        pipeline cannot cleanly split. A high rate is not noise, it means a
        template is producing code that does not parse, and every example it
        touched is teaching the model to write broken Python.

        Raises:
            RuntimeError: If the invalid share exceeds `max_invalid_ratio`.
        """
        n_rejected = sum(rejected.values())
        total = kept + n_rejected
        self.synthesis_stats = {
            "kept": kept,
            "rejected": n_rejected,
            "rejected_by_task": dict(rejected),
            "invalid_ratio": n_rejected / max(total, 1),
        }

        if not n_rejected:
            console.print("  [green]All synthesized examples are valid Python[/green]")
            return

        ratio = self.synthesis_stats["invalid_ratio"]
        breakdown = ", ".join(f"{task}={count}" for task, count in sorted(rejected.items()))
        console.print(
            f"  [yellow]Dropped {n_rejected} unparseable examples "
            f"({ratio:.2%}) — {breakdown}[/yellow]"
        )

        if ratio > self.max_invalid_ratio:
            raise RuntimeError(
                f"{ratio:.2%} of synthesized examples were not valid Python, above the "
                f"{self.max_invalid_ratio:.2%} limit ({breakdown}). A synthesis template is "
                f"likely emitting malformed code. Inspect src/data/synthesize.py, or set "
                f"synthesis.max_invalid_ratio in the data config if this rate is expected."
            )

    def _export(self, samples: list[dict]) -> tuple[Path, Path]:
        """Split and export samples as JSONL files."""
        train_path = Path(self.output_config.get("train_path", "./data/curated/train.jsonl"))
        eval_path = Path(self.output_config.get("eval_path", "./data/curated/eval.jsonl"))
        eval_ratio = self.output_config.get("eval_split_ratio", 0.05)
        seed = self.output_config.get("seed", 42)

        # Create output directories
        train_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.parent.mkdir(parents=True, exist_ok=True)

        # Shuffle
        if self.output_config.get("shuffle", True):
            random.seed(seed)
            random.shuffle(samples)

        # Split
        eval_size = max(1, int(len(samples) * eval_ratio))
        eval_samples = samples[:eval_size]
        train_samples = samples[eval_size:]

        # Write JSONL
        def _write_jsonl(path: Path, data: list[dict]) -> None:
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        _write_jsonl(train_path, train_samples)
        _write_jsonl(eval_path, eval_samples)

        console.print(f"\n[bold green]Exported:[/bold green]")
        console.print(f"  Train: {train_path} ({len(train_samples)} samples)")
        console.print(f"  Eval:  {eval_path} ({len(eval_samples)} samples)")

        return train_path, eval_path

    def run(self) -> tuple[Path, Path]:
        """
        Execute the full data curation pipeline.

        Returns:
            Tuple of (train_path, eval_path) for the exported JSONL files.
        """
        from rich.panel import Panel

        console.print(Panel("[bold]Data Curation Pipeline[/bold]", border_style="blue"))

        # Stage 1: Ingest
        console.print("\n[bold blue]Stage 1: Ingesting data...[/bold blue]")
        raw_samples = list(self._ingest_sources())
        console.print(f"  Total raw samples: {len(raw_samples)}")
        if not raw_samples:
            raise RuntimeError("No raw samples loaded from configured sources")

        # Stage 2: Quality Filter
        console.print("\n[bold blue]Stage 2: Quality filtering...[/bold blue]")
        filtered = self._filter_batch(raw_samples)
        stats = self.quality_filter.report()
        console.print(f"  Passed: {stats['passed']}/{stats['total']} ({stats['pass_rate']*100:.1f}%)")
        if not filtered:
            raise RuntimeError("No samples passed quality filters")

        # Stage 3: Deduplicate
        deduped = self._deduplicate(filtered)

        # Stage 4: Synthesize instruction examples
        synthesized = self._synthesize(deduped)
        if not synthesized:
            raise RuntimeError("No instruction examples were synthesized")

        # Stage 5: Export
        train_path, eval_path = self._export(synthesized)

        # Print summary
        console.print(Panel(
            f"[bold green]Pipeline Complete[/bold green]\n"
            f"Raw: {len(raw_samples)} → Filtered: {len(filtered)} → "
            f"Deduped: {len(deduped)} → Synthesized: {len(synthesized)}",
            border_style="green",
        ))

        return train_path, eval_path


def main():
    """CLI entry point for data curation."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Curate training data for code model fine-tuning")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                        help="Path to data curation config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = DataCurationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
