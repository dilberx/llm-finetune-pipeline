"""Sample the gold completion set and a disjoint PPI pool from the curated eval split.

The original data/eval/code_completion_200.jsonl was sampled from the curated
split minutes before it was regenerated with the fixed synthesizer, so 117 of
its 200 prompts carry the indentation defect. A model trained on the defective
data is scored on its own training format there, which flatters it against a
model trained on the fixed data. This script exists so the sets can be rebuilt
whenever the curated split changes, instead of being sampled by hand.

Usage:
    venv/bin/python scripts/build_eval_sets.py [--suffix _fixed]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.proxy import load_pool  # noqa: E402

# The defect: a docstring at the same indent as the `def` it belongs to.
_DEFECT = re.compile(r"```python\n( *)def [^\n]*\n( *)(\"\"\"|''')")


def has_indent_defect(prompt: str) -> bool:
    match = _DEFECT.search(prompt)
    return bool(match) and len(match.group(1)) == len(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="data/curated/eval.jsonl")
    parser.add_argument("--train", default="data/curated/train.jsonl",
                        help="Prompts found here are excluded from both sets")
    parser.add_argument("--gold", type=int, default=200)
    parser.add_argument("--pool", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--suffix", default="_fixed")
    args = parser.parse_args()

    rows = load_pool(args.source, task="code_completion")
    # MinHash dedup catches near-duplicate files, not a function whose signature
    # and docstring recur across files, so 14 of a 200 sample shared a prompt
    # with a training row (4 with the same completion too). Drop them.
    seen = {e.prompt for e in load_pool(args.train)}
    leaked = sum(e.prompt in seen for e in rows)
    rows = [e for e in rows if e.prompt not in seen]
    picked = random.Random(args.seed).sample(range(len(rows)), args.gold + args.pool)
    gold = [rows[i] for i in picked[: args.gold]]
    pool = [rows[i] for i in picked[args.gold :]]

    # Sampling without replacement guarantees disjoint indices; also check that
    # no prompt text repeats across the two, which duplicates in the split could cause.
    overlap = {e.prompt for e in gold} & {e.prompt for e in pool}
    defects = sum(has_indent_defect(e.prompt) for e in gold + pool)
    if defects:
        raise SystemExit(f"{defects} sampled prompts still carry the indent defect; "
                         f"is {args.source} from the fixed synthesizer?")

    out = Path("data/eval")
    for name, examples in [(f"code_completion_{args.gold}{args.suffix}", gold),
                           (f"ppi_pool_{args.pool}{args.suffix}", pool)]:
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for e in examples:
                handle.write(json.dumps({"prompt": e.prompt, "completion": e.completion}) + "\n")
        print(f"wrote {path} ({len(examples)} rows)")
    print(f"source rows: {len(rows)} (dropped {leaked} also in train)  prompt overlap gold/pool: {len(overlap)}  defects: {defects}")


if __name__ == "__main__":
    main()
