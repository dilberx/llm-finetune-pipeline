# Working on this repository

Operational notes for anyone — human or agent — running the GPU pipeline here.
These are things that cost real hours to learn; the code does not make them
obvious.

## Before any GPU load test

```bash
make gpu-preflight            # or: scripts/gpu_preflight.sh 9000
```

`train`, `train-resume`, `eval` and `sweep` depend on this and refuse to start
when free VRAM is below the threshold. Override per invocation when you know
what you are doing:

```bash
make train PREFLIGHT_MIN_FREE=8000
```

**Never skip it on this machine.** The card is a 10 GB RTX 3080 and the
training config has no headroom to spare. When a run starts without enough free
memory it does not fail cleanly — the paged optimizer thrashes, and that looks
like *100% GPU utilization with the step counter frozen*. A run was left in that
state for seven minutes before anyone noticed, because every liveness check said
it was fine.

### Finding what holds the GPU

`nvidia-smi --query-compute-apps` **returns nothing under WSL2**, even while a
job holds 9 GB. It is not that the GPU is free; the process list simply is not
visible. `scripts/gpu_preflight.sh` instead scans `/proc/*/fd` for open GPU
device handles. **Under WSL2 the GPU is `/dev/dxg`** — `/dev/nvidia*` does not
exist there. An earlier version of the script scanned only `/dev/nvidia*`, so
it reported "no GPU processes" while other jobs held the card. Test any change
to this detection against a process that is actually running, not an idle GPU.

Each process is listed with its working directory and uptime, because the
command line alone does not say whose it is: every project on the machine uses
`./venv/bin/python`. **Another project on this machine shares the GPU** (for
example `llm-inference-benchmark`), so check the listed project directories
before assuming a process is yours.

The script never stops anything. Whether a process is needed is the owner's
decision.

Memory in use with *no* process visible in WSL belongs to the Windows side —
desktop compositor, a browser with hardware acceleration, another CUDA app.
Nothing inside WSL can release it. Earlier notes here put that at "~1.8 GB
routinely"; that figure came from the broken detection above and is not
established.

**Use strict mode for anything whose speed you will quote:**

```bash
PREFLIGHT_STRICT=1 scripts/gpu_preflight.sh 6000
```

A second process can slow a run by taking compute while leaving plenty of
memory free, and the free-memory check cannot see that. Strict mode exits 3
when any other process holds the GPU.

## Monitoring a long run

Watch the **step counter**, not the process. A hung run stays alive, keeps the
GPU at 100%, and advances nothing. Compare the step number between ticks and
say so when it does not move:

```bash
CUR=$(tr '\r' '\n' < train.log | grep -aoE "[0-9]+/[0-9]+ \[" | tail -1 | cut -d/ -f1)
[ "$CUR" = "$PREV" ] && echo "STALLED at step $CUR"
```

Two more traps in the logs:

- Rich and tqdm write progress with `\r`. Pipe through `tr '\r' '\n'` or the
  last line looks frozen when it is not.
- Rich's `track()` suppresses itself on a non-TTY, so a redirected log shows the
  stage banner and then nothing at all for the whole stage. Absence of output is
  not absence of progress; check accruing CPU time instead.

## Training gotchas

**`make train` auto-resumes.** With no `--resume-from`, `finetune.py` calls
`find_latest_checkpoint()` and silently picks up whatever is in `checkpoints/`.
To start fresh, move existing `checkpoint-*` directories aside first — a
subdirectory such as `checkpoints/_archive_<date>/` is enough, since the glob is
not recursive. Resuming a finished run on a new corpus with an exhausted LR
schedule looks like it is working and produces nonsense.

**Training off this machine.** `scripts/cloud_train.sh` with
`configs/finetune_cloud.yaml` runs a pilot (20k examples, one epoch, ~2,500
steps) on a rented 24 GB card. The header of the script has the copy-up and
copy-back commands. It keeps 4-bit, LoRA and the effective batch of 8 identical
to the local config so the adapters are comparable, refuses to start over
existing checkpoints, and runs the preflight in strict mode. Evaluate the
returned adapters locally; the eval fits here.

**`save_steps` must be a multiple of `eval_steps`** while
`load_best_model_at_end` is set, which `finetune.py` sets unconditionally. A
violation fails at Trainer construction, minutes into a run, after model
loading. `tests/test_config.py` pins this against the shipped config.

**Cap the eval split.** `max_samples` scales train and eval together;
`max_eval_samples` caps eval alone. Uncapped, mid-training evaluation consumed
25% of a 21.8 h run (25 evals × 7,070 samples × ~772 s). Mid-training eval
exists to pick a checkpoint, not to benchmark.

## Measured on this hardware

A 1.5B model, 4-bit NF4, LoRA r=16, gradient checkpointing, `paged_adamw_8bit`.

**Treat every figure in this section as taken under unknown concurrent GPU
load.** The process detection was broken when these were measured, and another
project's job was sharing the card at least some of the time. The training
thrash, the rollout throughput and the batching speedup may all include that
contention. Re-measure with `PREFLIGHT_STRICT=1` before relying on a number.

| config | free VRAM | rate | outcome |
|--------|-----------|------|---------|
| batch 2, unpacked, seq 1024 | 226 MiB | 7.2 s/it | froze at step 24 |
| batch 1, unpacked, seq 1024 | 2.4 GB | 6.27 s/it | ran, ~87 h for 3 epochs |
| batch 1, **packed**, seq 1024 | ~400 MiB | 58 s/it | thrashed, ~200 h |

Packing cuts the step count roughly 4× (median example is ~157 tokens against a
1024 window) but fills every sequence to the full window, so per-step memory
rises and on this card that costs far more than the step count saves. **Memory
is the binding constraint here; evaluate any speed change for its memory effect
first.**

### Rollout throughput (sampling)

Same 1.5B model on the same card, HF `generate`, 96 new tokens, temperature 0.8,
64 sequences per call (8 prompts x 8 samples), 2 timed calls per setting:

| config | s/call | sequences/s | peak VRAM |
|--------|--------|-------------|-----------|
| 4-bit | 12.2 | 5.3 | 2.7 GB |
| 4-bit + LoRA | 19.3 | 3.3 | 3.8 GB |
| bf16 | 6.3 | 10.2 | 4.7 GB |
| bf16 + LoRA | 12.8 | 5.0 | 5.7 GB |

**Batching is the lever.** One sequence at a time ran 0.09-0.31 sequences/s, so
batched sampling is roughly 30-50x faster. An unbatched estimate overstates
rollout cost by that factor: 8,000 sequences take about 27 minutes in bf16 +
LoRA, not 17 hours. An unmerged LoRA adapter roughly doubles per-step latency
(~134 ms against ~66 ms per step in bf16), and 4-bit is about half the speed of
bf16 at the cost of ~2 GB less memory.

Caveats: this is static batching, not continuous batching (vLLM is not
installed), so a sequence that ends early still occupies its slot until the
longest one in the batch finishes. The single-sequence rows are noisy (two
calls). About 2.4 GB of the card is held Windows-side throughout. The adapter
used was the one trained on the pre-fix data, so its output lengths (mean ~50
tokens) will differ from a new policy's.

### Batched evaluation (`batch_size` in the completion benchmark)

`configs/eval_config.yaml` sets 16; the code default is 1. Results and sweep
payloads record the batch size under `generation`. Reports written before this
was set used 1, so do not compare them item by item with newer ones. Measured on
the 200-example completion set, greedy, 256 new tokens, `checkpoints/best` (the
adapter trained on pre-fix data), preflight showing no other GPU process:

| batch size | seconds | speedup | identical to bs 1 | token_f1 | parse_rate | ast_match |
|------------|---------|---------|-------------------|----------|------------|-----------|
| 1  | 2235 | 1.0x | 200/200 | 0.5051 | 0.930 | 0.075 |
| 2  | 1929 | 1.2x | 116/200 | 0.5049 | 0.925 | 0.075 |
| 4  | 1362 | 1.6x | 115/200 | 0.5014 | 0.910 | 0.075 |
| 8  | 866  | 2.6x | 117/200 | 0.5039 | 0.930 | 0.070 |
| 16 | 559  | 4.0x | 117/200 | 0.5027 | 0.930 | 0.070 |

It is not 30-50x: that figure is for sampling throughput, not for this
benchmark. An earlier 24-example run showed 4.2x at batch 8, so the speedup
depends on the prompt-length mix.

**Batched greedy output is not identical to unbatched, but the scores are.**
Any batch size above 1 changes about 42% of generations. A repeated batch-8 run
was byte-identical, so this is batch shape, not run-to-run randomness. The
aggregate spread across batch sizes is small next to sampling error: 0.12 of
the 95% CI half-width for token_f1, 0.13 for ast_match, but **0.57 for
parse_rate**, the fragile one. No batch-size change gave a significant paired
difference. (An n=24 run had suggested a parse_rate gap of 0.75 vs 0.875; at
n=200 it is 0.02. That was small-sample noise.)

So: compare scores only within one batch size, and do not treat per-example
outputs as stable across batch sizes — that matters for anything built on
per-item responses, such as the IRT calibration. Raw runs and the analysis
script are in `reports/batch_size_sweep/` (gitignored, local only).

The cause is numerical, not a masking bug. Teacher-forced logits against the
unbatched run, 24 sequences, argmax flips at positions:

| variant | argmax flips | largest flip margin |
|---------|--------------|---------------------|
| left-padded batch, correct mask | 1.10% | 0.63 logits |
| no padding at all, batch shape only | 0.98% | 0.63 logits |
| pad tokens attended to (deliberate bug) | 17.0% | 17.3 logits |

Padding adds nothing over batch shape alone, and every flip is a near-tie. A
real masking error looks very different (17x the flip rate, margins in the
tens). If you change the batching code, rerun that bug control rather than
trusting a high identical-output count. The control script lived in a scratch
directory and is gone; rebuild it from the three rows above (teacher-forced
logits, one sequence alone against the same sequence inside a padded batch).

The cost of a policy update has not been measured. SFT on this card ran 6.27 s
per 8 sequences (~0.78 s per sequence), which would put 8,000 sequences at
about 1.7 hours of forward/backward per pass; that is an extrapolation, but it
suggests the update, not sampling, is what dominates.

Beware `train_steps_per_second` in `training_metadata.json`: HF computes it as
`max_steps / train_runtime`, so for a *resumed* run it divides the full step
count by the runtime of only the resumed segment. The 0.642 recorded there
implies a far faster rate than the ~0.25 steps/s actually achieved.

## Makefile

Targets use `$(PYTHON)`, which resolves to `venv/bin/python` when present, then
`python`, then `python3`. Do not write a bare `python` — it does not exist on
Debian/Ubuntu and silently breaks every target with "No such file".

## Data

`bigcode/starcoderdata` is **gated** and fails to load; curation falls back to
`codeparrot/codeparrot-clean`. The README's architecture diagram names
StarCoder, which is not what actually runs.

`make curate` overwrites `data/curated/{train,eval}.jsonl` in place. Back them
up first — a subdirectory of `data/curated/` survives the run and stays
gitignored.
