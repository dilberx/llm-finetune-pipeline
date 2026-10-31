"""
Score a population of checkpoints and calibrate the evaluation set against them.

Item response theory needs a *population* of models. Comparing a base model to
a fine-tuned one gives two, which is not enough to identify anything: with two
models every item is either passed by both, failed by both, or split one-one,
and none of those distinguish a hard item from a discriminating one.

Training checkpoints are that population, and they are free -- a run that saved
periodically has already produced a ladder of models of graded ability, which
is exactly what a calibration sample looks like.

What comes back is a description of the evaluation set rather than of any
model: which items separate strong checkpoints from weak ones, which are too
easy or too hard to inform anything, and which small subset carries most of the
measurement. That last one is the practical output -- an eval set that costs a
fraction of the full one and measures nearly as well.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from ..training.checkpoint import get_checkpoint_step, validate_checkpoint
from .benchmarks import CodeCompletionBenchmark
from .irt import MIN_MODELS_FOR_IRT, fit_2pl, responses_from_raw_results

console = Console()


@dataclass(frozen=True)
class CheckpointRef:
    """One member of the calibration population."""

    name: str
    adapter_path: str | None
    step: int | None

    @property
    def is_base(self) -> bool:
        return self.adapter_path is None


def discover_checkpoints(
    output_dir: str | Path,
    include_base: bool = True,
    stride: int = 1,
    limit: int | None = None,
) -> list[CheckpointRef]:
    """
    Find scoreable checkpoints, ordered by training step.

    Only directories named `checkpoint-<step>` are considered, and each must
    pass the same integrity check training uses on resume -- a partially
    written checkpoint would otherwise enter the population as a spuriously
    weak model. Smoke-test and diagnostic directories are ignored by name.

    The base model is included as the zero-step member. It anchors the low end
    of the ability scale, which matters because IRT only recovers ability
    differences, not absolute levels.

    Args:
        output_dir: Directory holding checkpoint-* subdirectories.
        include_base: Include the un-adapted base model as step 0.
        stride: Keep every nth checkpoint. Ability changes slowly late in a
            run, so consecutive checkpoints are near-duplicates for this
            purpose and spreading the sample is worth more than density.
        limit: Keep at most this many checkpoints, dropping the earliest.

    Returns:
        Checkpoints ordered by step, base model first when included.
    """
    directory = Path(output_dir)
    found: list[CheckpointRef] = []

    if directory.is_dir():
        for path in directory.glob("checkpoint-*"):
            if not path.is_dir() or not validate_checkpoint(path):
                continue
            step = get_checkpoint_step(path)
            if step is None:
                continue
            found.append(CheckpointRef(name=path.name, adapter_path=str(path), step=step))

    found.sort(key=lambda ref: ref.step or 0)

    if stride > 1:
        found = found[::stride]
    if limit is not None and len(found) > limit:
        found = found[-limit:]

    if include_base:
        found.insert(0, CheckpointRef(name="base", adapter_path=None, step=0))
    return found


def build_model_config(base_config: dict[str, Any], ref: CheckpointRef) -> dict[str, Any]:
    """Build the model config that loads one population member."""
    config = {
        "name": base_config["name"],
        "trust_remote_code": base_config.get("trust_remote_code", True),
    }
    if base_config.get("revision"):
        config["revision"] = base_config["revision"]
    if ref.adapter_path is not None:
        config["adapter_path"] = ref.adapter_path
    return config


def score_population(
    evaluator: Any,
    refs: list[CheckpointRef],
    benchmark_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Run the completion benchmark once per checkpoint.

    Models are loaded and released one at a time, since a consumer GPU holds
    exactly one.

    Args:
        evaluator: A ModelEvaluator, used for its model loading and quantization.
        refs: The population to score.
        benchmark_config: The `code_completion` benchmark settings.

    Returns:
        Checkpoint name to its per-example results.
    """
    import torch

    scored: dict[str, list[dict[str, Any]]] = {}
    for index, ref in enumerate(refs, start=1):
        console.print(
            f"\n[bold cyan]({index}/{len(refs)}) Scoring {ref.name}"
            f"{'' if ref.step is None else f' @ step {ref.step}'}...[/bold cyan]"
        )
        model, tokenizer = evaluator._load_model(
            build_model_config(evaluator.config["models"]["base"], ref)
        )
        try:
            benchmark = CodeCompletionBenchmark(
                dataset_path=benchmark_config.get("dataset"),
                max_new_tokens=benchmark_config.get("max_new_tokens", 256),
                temperature=benchmark_config.get("temperature", 0.2),
                batch_size=benchmark_config.get("batch_size", 1),
            )
            result = benchmark.run(model, tokenizer)
            raw = result.get("raw_results")
            if raw:
                scored[ref.name] = raw
            else:
                console.print(f"  [yellow]{ref.name} produced no per-example results[/yellow]")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return scored


def calibrate(
    scored: dict[str, list[dict[str, Any]]],
    metric: str = "ast_match",
    threshold: float | None = None,
    keep_items: int = 8,
) -> dict[str, Any]:
    """
    Fit the 2PL model and describe what the evaluation set is measuring.

    Args:
        scored: Checkpoint name to per-example results.
        metric: Per-example field to use as the response.
        threshold: Pass mark for a continuous metric; None if already boolean.
        keep_items: Size of the informative subset to recommend.

    Returns:
        The fit plus the recommended subset, ready to serialise.

    Raises:
        ValueError: If the population is too small or carries no information.
    """
    responses, item_ids, model_ids = responses_from_raw_results(
        scored, metric=metric, threshold=threshold
    )
    fit = fit_2pl(responses, item_ids=item_ids, model_ids=model_ids)

    payload = fit.to_dict()
    payload["metric"] = metric
    payload["threshold"] = threshold
    payload["n_items_total"] = len(item_ids)
    payload["recommended_items"] = fit.select_items(keep_items)
    return payload


def _render(payload: dict[str, Any]) -> None:
    """Print the calibration to the console."""
    table = Table(title="Item Calibration", header_style="bold cyan")
    table.add_column("Item")
    table.add_column("Difficulty", justify="right")
    table.add_column("Discrimination", justify="right")
    table.add_column("Keep", justify="center")

    keep = set(payload["recommended_items"])
    ranked = sorted(payload["items"], key=lambda item: -item["discrimination"])
    for item in ranked[:20]:
        table.add_row(
            item["item_id"][:48],
            f"{item['difficulty']:+.2f}",
            f"{item['discrimination']:.2f}",
            "yes" if item["item_id"] in keep else "",
        )
    console.print(table)

    ability = Table(title="Recovered Ability", header_style="bold cyan")
    ability.add_column("Checkpoint")
    ability.add_column("Ability", justify="right")
    for entry in sorted(payload["models"], key=lambda m: -m["ability"]):
        ability.add_row(entry["model_id"], f"{entry['ability']:+.2f}")
    console.print(ability)

    degenerate = payload["n_degenerate"]
    total = payload["n_items_total"]
    if degenerate:
        console.print(
            f"\n[yellow]{degenerate} of {total} items are degenerate: every checkpoint "
            f"scored them identically, so they carry no information about "
            f"the checkpoints scored here. A wider ability range may separate them."
            f"[/yellow]"
        )


def run_sweep(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Discover, score, and calibrate. Returns the serialisable payload."""
    from .evaluate import ModelEvaluator

    refs = discover_checkpoints(
        args.checkpoints, include_base=not args.no_base, stride=args.stride, limit=args.limit
    )
    if len(refs) < MIN_MODELS_FOR_IRT:
        raise SystemExit(
            f"Found {len(refs)} scoreable models in {args.checkpoints}, but item "
            f"calibration needs at least {MIN_MODELS_FOR_IRT}. Training saves a checkpoint "
            f"every save_steps; lower it or raise save_total_limit to build the population."
        )

    names = ", ".join(r.name for r in refs)
    console.print(f"[bold]Scoring {len(refs)} checkpoints:[/bold] {names}")
    evaluator = ModelEvaluator(config)
    scored = score_population(evaluator, refs, config["benchmarks"]["code_completion"])

    if len(scored) < MIN_MODELS_FOR_IRT:
        raise SystemExit(
            f"Only {len(scored)} checkpoints produced per-example results; "
            f"need {MIN_MODELS_FOR_IRT}."
        )

    payload = calibrate(
        scored, metric=args.metric, threshold=args.threshold, keep_items=args.keep
    )
    payload["checkpoints"] = [
        {"name": r.name, "step": r.step, "adapter_path": r.adapter_path} for r in refs
    ]
    return payload


def main() -> None:
    """CLI entry point for checkpoint sweep calibration."""
    parser = argparse.ArgumentParser(
        description="Score checkpoints and calibrate the eval set against them"
    )
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--metric", default="ast_match")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass mark for a continuous metric such as token_f1",
    )
    parser.add_argument("--stride", type=int, default=1, help="Keep every nth checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most this many")
    parser.add_argument("--keep", type=int, default=8, help="Size of the recommended subset")
    parser.add_argument("--no-base", action="store_true", help="Exclude the un-adapted base")
    parser.add_argument("--out", default="./reports/item_calibration.json")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    payload = run_sweep(config, args)
    _render(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    console.print(f"\n[dim]Calibration saved: {out_path}[/dim]")


if __name__ == "__main__":
    main()
