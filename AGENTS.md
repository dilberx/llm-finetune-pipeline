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
visible. `scripts/gpu_preflight.sh` instead scans `/proc/*/fd` for open
`/dev/nvidia*` handles, which does work inside the WSL namespace.

Memory in use with *no* WSL process holding a handle belongs to the Windows
side — desktop compositor, a browser with hardware acceleration, another CUDA
app. Roughly 1.8 GB sits there routinely, which is ~18% of the card gone before
training starts. Nothing inside WSL can release it. The script reports it and
stops; closing those applications is a human decision, not something a script
should do on its own.

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

**`save_steps` must be a multiple of `eval_steps`** while
`load_best_model_at_end` is set, which `finetune.py` sets unconditionally. A
violation fails at Trainer construction, minutes into a run, after model
loading. `tests/test_config.py` pins this against the shipped config.

**Cap the eval split.** `max_samples` scales train and eval together;
`max_eval_samples` caps eval alone. Uncapped, mid-training evaluation consumed
25% of a 21.8 h run (25 evals × 7,070 samples × ~772 s). Mid-training eval
exists to pick a checkpoint, not to benchmark.

## Measured on this hardware

A 1.5B model, 4-bit NF4, LoRA r=16, gradient checkpointing, `paged_adamw_8bit`:

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
