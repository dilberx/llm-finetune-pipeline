# Content Agent Handoff V2

Generated: 2026-06-15

## Scope

Use this file with `reports/docs_update_agent_handoff.md`.

This artifact is for a separate content agent. Do not treat it as a request to edit README or other repository prose in this pass.

## Current Git State

Staged code/config changes at the time of review:

- `Makefile`
- `configs/data_config.yaml`
- `configs/finetune_config.yaml`
- `pyproject.toml`
- `src/data/curate.py`
- `src/evaluation/benchmarks.py`
- `src/training/finetune.py`
- `src/utils/config.py`
- `src/utils/gpu.py`

Staged diff summary:

```text
9 files changed, 53 insertions(+), 20 deletions(-)
```

Unstaged file present:

- `README.md`

Do not stage the README unless the user asks for that.

## Final Artifacts To Use

Training:

- `checkpoints/best`
- `checkpoints/best/training_metadata.json`
- `checkpoints/checkpoint-50000`
- `checkpoints/checkpoint-50382`

Evaluation:

- `reports/main_eval_clean_retry/evaluation_report.md`
- `reports/main_eval_clean_retry/evaluation_results.json`
- `reports/target_domain_fixed/evaluation_report.md`
- `reports/target_domain_fixed/evaluation_results.json`
- `reports/target_domain_fixed/input_manifest.json`

Eval data:

- `data/eval/code_completion_fixed16.jsonl`
- `data/eval/target_perplexity_fixed128.txt`
- `data/eval/target_eval_fixed.yaml`

Older diagnostic runs exist under:

- `reports/main_eval_clean`
- `reports/target_domain`
- `reports/target_domain_clean`

Use those only for provenance.

## Run Facts

Base model:

```text
Qwen/Qwen2.5-Coder-1.5B
```

Fine-tuning:

```text
QLoRA, 4-bit quantization, LoRA r=16, alpha=32
```

Training metadata from `checkpoints/best/training_metadata.json`:

```text
run_id: fd7c6ca8
train_runtime: 78435.6452 seconds
train_samples_per_second: 5.139
train_steps_per_second: 0.642
train_loss: 0.31003253989986096
trainable_params: 18464768
total_params: 1562179072
resumed_from: checkpoints/checkpoint-32000
```

Trainer state:

```text
best eval loss: 0.9893123507499695
best checkpoint: checkpoints/checkpoint-50000
final checkpoint: checkpoints/checkpoint-50382
final global step: 50382
```

## Evaluation Facts

Target-domain eval:

```text
source: data/curated/eval.jsonl
code completion examples: 16
perplexity text examples: 128
seed: 123
max_new_tokens: 96
```

Target-domain results:

```text
Target perplexity: base 3.2652, fine-tuned 3.0355
Target avg_nll: base 1.1833, fine-tuned 1.1104
Code completion exact_match: base 0.0000, fine-tuned 0.0000
Code completion token_f1: base 0.1006, fine-tuned 0.2299
```

Generic eval results:

```text
HumanEval pass@1: base 0.0000, fine-tuned 0.0000
HumanEval pass@5: base 0.0000, fine-tuned 0.0000
HumanEval pass@10: base 0.0000, fine-tuned 0.0000
Wikitext perplexity: base 10.3337, fine-tuned 12.5419
Wikitext avg_nll: base 2.3354, fine-tuned 2.5291
```

## Safe Claim

Use this claim if a summary is needed:

```text
The fine-tuned adapter improved on a small target-domain held-out evaluation, while generic Wikitext perplexity regressed and the small embedded HumanEval sample stayed at zero for both models.
```

## Superseded

The exact-match findings below are a metric artifact, not a result. References
in the eval set are markdown-fenced, so `exact_match` has a ceiling of zero:
scoring the references against themselves gives 0/16, while AST equivalence
gives 16/16. Separately, a synthesis bug left ~70% of generated prompts
syntactically invalid. Both are fixed; the numbers in this file predate the
fixes and must be regenerated before reuse.

## Do Not Claim

Do not write that:

- the model improved overall
- HumanEval improved
- general language modeling improved
- code completion passed
- exact-match completion quality improved
- exact match staying at zero says anything about the model

## Content Agent Tasks

Recommended next pass:

1. Reconcile README with `reports/docs_update_agent_handoff.md`.
2. Keep target-domain and generic eval results separate.
3. Add caveats near every metric table.
4. Do not present exact-match at zero as a model finding; it is a metric ceiling.
5. Keep generated artifacts separate from code/config changes unless the user asks.

## Engineering Follow-Up

Keep this short unless the user asks for more:

1. Expand target-domain code completion beyond 16 examples.
2. Run a larger code benchmark such as full HumanEval or MBPP.
3. Add confidence intervals for target-domain metrics.
4. Review qualitative completions before planning another training run.

## Suggested Commit Message

```text
Prepare fine-tuning pipeline and target-domain evaluation
```
