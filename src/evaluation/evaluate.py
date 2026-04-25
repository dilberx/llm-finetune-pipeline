"""
Model evaluation framework with multiple benchmarks.

Supports:
- HumanEval pass@k estimation for code generation quality
- Perplexity measurement for language modeling quality
- Custom code completion evaluation
- Statistical comparison between base and fine-tuned models
- Report generation with tables and plots
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .benchmarks import HumanEvalBenchmark, PerplexityBenchmark, CodeCompletionBenchmark
from .metrics import compute_comparison_stats

console = Console()


class ModelEvaluator:
    """
    Orchestrates model evaluation across multiple benchmarks.

    Compares base model vs. fine-tuned model performance and
    generates statistical comparison reports.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.benchmarks_config = config.get("benchmarks", {})
        self.comparison_config = config.get("comparison", {})
        self.output_config = config.get("output", {})
        self.hardware_config = config.get("hardware", {})

    def _load_model(self, model_config: dict) -> tuple:
        """Load a model (base or fine-tuned with adapter)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = self.config.get("quantization", {})
        bnb_config = None
        if quant_config.get("enabled", False):
            compute_dtype = getattr(torch, quant_config.get("compute_dtype", "bfloat16"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(quant_config.get("bits", 4) == 4),
                bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get("double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model_name = model_config["name"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=self.hardware_config.get("device_map", "auto"),
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        # Load LoRA adapter if specified
        adapter_path = model_config.get("adapter_path")
        if adapter_path:
            console.print(f"  Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        return model, tokenizer

    def evaluate_model(self, model, tokenizer, model_name: str) -> dict[str, Any]:
        """Run all configured benchmarks on a single model."""
        results = {"model": model_name, "benchmarks": {}}

        # HumanEval
        if self.benchmarks_config.get("humaneval", {}).get("enabled", False):
            console.print(f"\n[bold cyan]Running HumanEval on {model_name}...[/bold cyan]")
            he_config = self.benchmarks_config["humaneval"]
            benchmark = HumanEvalBenchmark(
                num_samples=he_config.get("num_samples_per_task", 10),
                k_values=he_config.get("k_values", [1, 5, 10]),
                temperature=he_config.get("temperature", 0.8),
                top_p=he_config.get("top_p", 0.95),
                max_new_tokens=he_config.get("max_new_tokens", 512),
            )
            results["benchmarks"]["humaneval"] = benchmark.run(model, tokenizer)

        # Perplexity
        if self.benchmarks_config.get("perplexity", {}).get("enabled", False):
            console.print(f"\n[bold cyan]Running Perplexity on {model_name}...[/bold cyan]")
            ppl_config = self.benchmarks_config["perplexity"]
            benchmark = PerplexityBenchmark(
                dataset=ppl_config.get("dataset", "wikitext"),
                stride=ppl_config.get("stride", 512),
                max_length=ppl_config.get("max_length", 2048),
            )
            results["benchmarks"]["perplexity"] = benchmark.run(model, tokenizer)

        # Code Completion
        if self.benchmarks_config.get("code_completion", {}).get("enabled", False):
            console.print(f"\n[bold cyan]Running Code Completion on {model_name}...[/bold cyan]")
            cc_config = self.benchmarks_config["code_completion"]
            benchmark = CodeCompletionBenchmark(
                dataset_path=cc_config.get("dataset"),
                max_new_tokens=cc_config.get("max_new_tokens", 256),
                temperature=cc_config.get("temperature", 0.2),
            )
            results["benchmarks"]["code_completion"] = benchmark.run(model, tokenizer)

        return results

    def run(self) -> dict[str, Any]:
        """
        Run evaluation on both base and fine-tuned models.

        Returns full comparison results with statistical tests.
        """
        console.print(Panel("[bold]Model Evaluation Pipeline[/bold]", border_style="blue"))
        models_config = self.config.get("models", {})
        all_results = {}

        # Evaluate base model
        if "base" in models_config:
            console.print("\n[bold blue]Loading base model...[/bold blue]")
            model, tokenizer = self._load_model(models_config["base"])
            all_results["base"] = self.evaluate_model(model, tokenizer, "Base Model")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Evaluate fine-tuned model
        if "finetuned" in models_config:
            console.print("\n[bold blue]Loading fine-tuned model...[/bold blue]")
            model, tokenizer = self._load_model(models_config["finetuned"])
            all_results["finetuned"] = self.evaluate_model(model, tokenizer, "Fine-tuned Model")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Statistical comparison
        if "base" in all_results and "finetuned" in all_results:
            all_results["comparison"] = compute_comparison_stats(
                all_results["base"],
                all_results["finetuned"],
                confidence_level=self.comparison_config.get("confidence_level", 0.95),
            )

        # Generate reports
        self._generate_reports(all_results)

        return all_results

    def _generate_reports(self, results: dict[str, Any]) -> None:
        """Generate evaluation reports in configured formats."""
        report_dir = Path(self.output_config.get("report_dir", "./reports"))
        report_dir.mkdir(parents=True, exist_ok=True)

        # Console summary
        self._print_summary(results)

        # JSON report
        if "json" in self.output_config.get("format", ["json"]):
            json_path = report_dir / "evaluation_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\n[dim]JSON report saved: {json_path}[/dim]")

        # Markdown report
        if "markdown" in self.output_config.get("format", ["markdown"]):
            md_path = report_dir / "evaluation_report.md"
            self._write_markdown_report(results, md_path)
            console.print(f"[dim]Markdown report saved: {md_path}[/dim]")

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print a rich console summary of evaluation results."""
        table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
        table.add_column("Benchmark", style="bold")
        table.add_column("Metric")

        if "base" in results:
            table.add_column("Base Model", justify="right")
        if "finetuned" in results:
            table.add_column("Fine-tuned", justify="right")
        if "comparison" in results:
            table.add_column("Δ Change", justify="right")

        # Populate table from results
        for benchmark_name in ["humaneval", "perplexity", "code_completion"]:
            base_bench = results.get("base", {}).get("benchmarks", {}).get(benchmark_name, {})
            ft_bench = results.get("finetuned", {}).get("benchmarks", {}).get(benchmark_name, {})

            if not base_bench and not ft_bench:
                continue

            metrics = set(list(base_bench.get("metrics", {}).keys()) +
                         list(ft_bench.get("metrics", {}).keys()))

            for metric_name in sorted(metrics):
                row = [benchmark_name, metric_name]
                base_val = base_bench.get("metrics", {}).get(metric_name)
                ft_val = ft_bench.get("metrics", {}).get(metric_name)

                if "base" in results:
                    row.append(f"{base_val:.4f}" if base_val is not None else "—")
                if "finetuned" in results:
                    row.append(f"{ft_val:.4f}" if ft_val is not None else "—")
                if "comparison" in results and base_val and ft_val:
                    delta = ft_val - base_val
                    color = "green" if delta > 0 else "red"
                    row.append(f"[{color}]{delta:+.4f}[/{color}]")

                table.add_row(*row)

        console.print(table)

    def _write_markdown_report(self, results: dict[str, Any], path: Path) -> None:
        """Write a detailed markdown evaluation report."""
        lines = [
            "# Model Evaluation Report\n",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        ]

        for model_key, model_name in [("base", "Base Model"), ("finetuned", "Fine-tuned Model")]:
            if model_key not in results:
                continue
            lines.append(f"\n## {model_name}\n")
            for bench_name, bench_data in results[model_key].get("benchmarks", {}).items():
                lines.append(f"\n### {bench_name}\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric, value in bench_data.get("metrics", {}).items():
                    lines.append(f"| {metric} | {value:.4f} |")

        if "comparison" in results:
            lines.append("\n## Statistical Comparison\n")
            for key, value in results["comparison"].items():
                lines.append(f"- **{key}**: {value}")

        with open(path, "w") as f:
            f.write("\n".join(lines))


def main():
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluator = ModelEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
