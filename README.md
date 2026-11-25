# LLM Fine-Tuning & Evaluation Pipeline

A production-grade ML training platform for fine-tuning, evaluating, and serving code LLMs. Built with infrastructure-first principles: containerized, observable, fault-tolerant, and GPU-efficient.

> **Not a tutorial.** This is an infrastructure tool — it handles the data pipeline, GPU resource management, checkpoint recovery, cost tracking, and monitoring so you can focus on the model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Docker Compose Orchestration                     │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  curate      │  train       │  eval        │  serve                 │
│  (CPU)       │  (GPU)       │  (GPU)       │  (GPU)                 │
│              │              │              │                        │
│  HuggingFace │  QLoRA       │  HumanEval   │  vLLM Engine           │
│  → Polars    │  BitsAndBytes│  Perplexity  │  Continuous Batching   │
│  → MinHash   │  PEFT        │  Bootstrap   │  LoRA Hot-Loading      │
│  → AST Filter│  Accelerate  │  CI          │  FastAPI + Prometheus  │
│  → ChatML    │  DeepSpeed   │              │  /metrics /health      │
│              │              │              │  /ready                │
├──────────────┴──────────────┴──────────────┴────────────────────────┤
│                     Infrastructure Layer                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │
│  │ GPU Manager  │ Structured   │ Checkpoint   │ Cost Tracker  │     │
│  │ VRAM Track   │ JSON Logging │ Fault        │ $/token       │     │
│  │ OOM Predict  │ Correlation  │ Tolerance    │ GPU-hours     │     │
│  │ Device Map   │ IDs          │ Integrity    │ Prometheus    │     │
│  └──────────────┴──────────────┴──────────────┴──────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│  CI/CD: GitHub Actions (lint → test → Docker build)                  │
│  Config: YAML + env overrides + dataclass validation                 │
│  Tests: 285 unit tests (CPU-compatible, no GPU required)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Status

Two QLoRA adapters have been trained on this pipeline, both on a single 10 GB RTX 3080
under WSL2:

| | Old adapter | Pilot adapter |
|---|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B | same |
| Method | QLoRA, 4-bit NF4, LoRA r=16, α=32, effective batch 8 | same |
| Training data | ~134K curated examples, **before** the synthesis fix below | 20K of the ~134K, **after** the fix |
| Data source | `codeparrot/codeparrot-clean` (`bigcode/starcoderdata` is gated and fails to load) | same |
| Length | ~50K steps (~3 epochs); resumed after several crashes | 2,500 steps (1 epoch), ~4.7 h |
| Compute (Trainer FLOPs) | 1.22e18 | 4.1e16 (~30x less) |
| Location | `checkpoints/_archive_20261111-232116_prefix_best/best` | `checkpoints/best` |

`checkpoints/` is gitignored; the adapters are local only.

The old adapter's run crashed and resumed several times, and checkpoint recovery picked
each one up. Its logged `train_steps_per_second` covers only the last resumed segment, so
it overstates the real rate (see `AGENTS.md`).

### Evaluation results

All on `data/eval/code_completion_200_fixed.jsonl`: 200 held-out completions from the
fixed curated split, with no prompt shared with train. Greedy decoding, batch 16, 256 new
tokens. CIs are 95% paired bootstrap.

**The base model needs a fair prompt.** Zero-shot, it repeats the instruction and rewrites
the whole function (~1,091 characters against ~300 for an adapter), so most of a naive
base-vs-adapter gap is output format. The fair baseline gives the base three worked
examples and cuts every output, base and adapter alike, to its first fenced code block:

| Paired against the few-shot base | Token F1 Δ [95% CI] | Parse rate Δ [95% CI] |
|---|---|---|
| Pilot adapter | +0.058 [+0.034, +0.083] | +0.080 [+0.035, +0.130] |
| Old adapter | +0.124 [+0.094, +0.155] | +0.050 [−0.005, +0.105] |

Absolute scores with no output cut, for reference:

| Metric | Zero-shot base | Old adapter | Pilot adapter |
|---|---|---|---|
| Token F1 | 0.161 | 0.534 | 0.469 |
| Parse rate | 0.400 | 0.905 | 0.935 |
| AST match | 0.000 | 0.090 | 0.015 |
| Exact match | 0.000 | 0.000 | 0.000 |

**What this means**: fine-tuning helps beyond format, but by much less than the raw table
suggests. The old adapter used ~30x the compute and different data, so its larger gain
cannot be put down to either one. On the pilot, 93% of the final token-F1 gain over the
zero-shot base is already there at step 350 of 2,500. Structural correctness (`ast_match`)
stays rare. Specializing on code costs general language modeling: with the old adapter, Wikitext
perplexity rose from 10.33 to 12.54 (not re-measured for the pilot).

`exact_match` is confounded rather than broken: every reference is wrapped in markdown
code fences, so it scores correct code as a miss unless the fences are reproduced too.

### Superseded results

Earlier versions of this README reported a 16-example evaluation and then a 200-example
one (parse rate 0.235 → 0.810, token F1 0.214 → 0.492). **Both are withdrawn.**

- A defect in the synthesis templates indented docstrings to a fixed 4 spaces whether or
  not the function was a method, which left roughly 70% of synthesized training prompts
  syntactically invalid. It is fixed, and curation now aborts when too many synthesized
  examples fail to parse.
- The old 200-example set (`data/eval/code_completion_200.jsonl`) was sampled before that
  fix. 117 of its 200 prompts carry the defect, which flatters a model trained on the
  defective data, and 14 of its prompts also appear in train.
- Its baseline was the zero-shot base, so the headline parse-rate gain was mostly format.

`scripts/build_eval_sets.py` rebuilds the gold and PPI sets from the current split. It
drops prompts that also appear in train, and stops if any sampled prompt still carries the
defect.

### Evaluation methods in the repo

- **Prediction-powered inference** (`src/evaluation/ppi.py`): generates on a small gold
  set and carries the estimate onto a larger pool using a cheap teacher-forced proxy.
- **IRT item calibration** (`src/evaluation/sweep.py`, `make sweep`): fits a 2PL model
  over training checkpoints. On two independent early-checkpoint ladders, item difficulty
  agreed only moderately across runs (Spearman 0.51), and a plain per-item pass rate
  predicted held-out models nearly as well. Treat it as exploratory; details are in
  `AGENTS.md`.
- **Batched evaluation**: batch 16 is 4x faster than 1 with no significant change in any
  aggregate score, but it changes about 42% of individual greedy outputs (numerical
  near-ties, not a masking bug). Compare per-item results only within one batch size.

Measured numbers, hardware gotchas and the reasoning behind them are in
[AGENTS.md](AGENTS.md).

## Quick Start

```bash
# Clone
git clone https://github.com/dilberx/llm-finetune-pipeline.git
cd llm-finetune-pipeline

# Install
make install

# Run the full pipeline
make curate    # Ingest and filter training data
make train     # Fine-tune with QLoRA
make eval      # Evaluate base vs. fine-tuned
make serve     # Start inference server with Prometheus metrics
```

### Docker

```bash
# Build
make build

# Run all stages (curate → train → eval → serve)
docker compose up

# Or run individual stages
docker compose up train
docker compose up serve
```

## Infrastructure Features

### GPU Resource Management

Real-time VRAM monitoring with OOM prediction. The GPU manager tracks memory allocation trajectories and warns before crashes — essential for long training runs on consumer hardware.

```python
from src.utils.gpu import GPUResourceManager

mgr = GPUResourceManager()

# Check all GPUs
for snap in mgr.get_all_snapshots():
    print(f"{snap.device_name}: {snap.used_mb:.0f}/{snap.total_mb:.0f} MB")

# Track memory during an operation
with mgr.track_operation("forward_pass") as metrics:
    output = model(input_ids)
print(f"Peak: {metrics['peak_mb']} MB, Delta: {metrics['delta_mb']} MB")

# OOM prediction
warning = mgr.record_usage()
if warning:
    print(f"⚠️ OOM risk: {warning['prediction']['risk']}")

# Device placement recommendation
strategy = mgr.recommend_device_map(model_size_mb=3000)
# → {"strategy": "single_gpu", "device": 0, "free_mb": 9240.0}
```

### Structured Logging & Observability

JSON-structured logging with correlation IDs for tracing across pipeline stages. Production-ready: switch between human-readable (dev) and JSON (production) via environment variable.

```bash
# Production mode
LOG_FORMAT=json LOG_FILE=/var/log/training.log make train

# Example JSON log output:
# {"timestamp": "2026-04-25T14:30:00", "level": "INFO", "component": "training",
#  "message": "Epoch 1/3 started", "correlation_id": "a1b2c3d4",
#  "data": {"epoch": 1, "total_epochs": 3}}
```

### Checkpoint Fault Tolerance

Training crashes at epoch 2.5 on a 72-hour run? The checkpoint manager automatically finds and validates the latest checkpoint, detecting corrupted saves before resuming.

```bash
# Auto-resume from latest valid checkpoint
make train-resume

# Or explicitly
python -m src.training.finetune --resume-from checkpoints/checkpoint-500
```

### Prometheus Metrics Endpoint

The serving endpoint exposes `/metrics` in Prometheus text format for scraping by Grafana/Datadog/CloudWatch.

```
# HELP gpu_memory_used_mb GPU memory currently in use (MB)
# TYPE gpu_memory_used_mb gauge
gpu_memory_used_mb{device="0",gpu="NVIDIA GeForce RTX 3080"} 8523.0

# HELP serving_tokens_per_second Inference throughput in tokens/sec
# TYPE serving_tokens_per_second gauge
serving_tokens_per_second 142.5
```

### Training Cost Tracker

Compute GPU-hours and estimate training cost in USD. Supports custom pricing for cloud vs. on-prem comparisons.

```python
from src.utils.metrics_export import TrainingCostTracker

tracker = TrainingCostTracker(gpu_name="RTX 3080", price_per_hour=0.35)
tracker.start()
# ... training loop ...
tracker.record_step(tokens=2048)
# ... more steps ...
tracker.stop()

print(tracker.report())
# {"gpu_hours": 2.5, "estimated_cost_usd": 0.875, "cost_per_1k_tokens_usd": 0.00012}
```

### Training Profiler

Wraps PyTorch profiler to capture GPU kernel timing, memory allocation timelines, and bottleneck classification. Exports Chrome traces for visual debugging.

```python
from src.utils.profiler import TrainingProfiler

profiler = TrainingProfiler(output_dir="./reports/profile")
with profiler.profile_steps(warmup=1, active=3) as prof:
    for batch in dataloader:
        with profiler.step("forward"):
            output = model(batch)
        with profiler.step("backward"):
            loss.backward()
        prof.step()

report = profiler.analyze()
# → {"bottleneck": "compute_bound", "gpu_utilization_pct": 85.2, "peak_memory_mb": 8500}
report.save("./reports/profile/training_profile.json")
```

## Project Structure

```
llm-finetune-pipeline/
├── src/
│   ├── data/
│   │   └── curate.py         # Data ingestion: HF → Polars → MinHash → AST → ChatML
│   ├── training/
│   │   ├── finetune.py       # QLoRA training with GPU monitoring + auto-resume
│   │   └── checkpoint.py     # Fault-tolerant checkpoint management
│   ├── evaluation/
│   │   ├── evaluate.py       # Orchestrator: base vs. fine-tuned comparison
│   │   ├── benchmarks.py     # HumanEval (pass@k), perplexity, code completion
│   │   ├── code_equivalence.py # Parse and AST-equivalence scoring
│   │   ├── metrics.py        # Statistical comparison with bootstrap CIs
│   │   ├── ppi.py            # Prediction-powered inference
│   │   ├── proxy.py          # Teacher-forced proxy scores for PPI
│   │   ├── irt.py            # 2PL item response model
│   │   └── sweep.py          # Score checkpoints and calibrate the eval set
│   ├── serving/
│   │   └── server.py         # vLLM server with Prometheus /metrics endpoint
│   └── utils/
│       ├── config.py         # YAML + env + dataclass config system
│       ├── gpu.py            # GPU resource manager + OOM prediction
│       ├── logging.py        # Structured JSON logging with correlation IDs
│       ├── profiler.py       # Training profiler (PyTorch profiler wrapper)
│       └── metrics_export.py # Prometheus exporter + cost tracker
├── configs/
│   ├── finetune_config.yaml  # Full run on the local card
│   ├── finetune_pilot_local.yaml  # 20K-example, one-epoch pilot (local)
│   ├── finetune_cloud.yaml   # Same pilot for a rented 24 GB card
│   ├── finetune_early_ladder*.yaml # First 350 steps, checkpoint every 10
│   ├── data_config.yaml
│   ├── eval_config.yaml
│   └── deepspeed_config.json # DeepSpeed ZeRO Stage 2 for multi-GPU
├── scripts/
│   ├── gpu_preflight.sh      # Refuse GPU jobs when VRAM is short (WSL2-aware)
│   ├── build_eval_sets.py    # Rebuild gold and PPI eval sets from the split
│   ├── early_ladder.sh       # Train the first N steps, then stop
│   └── cloud_train.sh        # Pilot run on a rented GPU
├── tests/                    # 285 unit tests (CPU-compatible)
├── Dockerfile                # Multi-stage CUDA 12.1 container
├── docker-compose.yaml       # 4-service GPU orchestration
├── Makefile                  # Developer workflow automation
└── .github/workflows/ci.yml # Lint → Test → Docker Build
```

## Configuration

All pipeline stages are driven by YAML configs with environment variable overrides:

```yaml
# configs/finetune_config.yaml
model:
  name: "Qwen/Qwen2.5-Coder-1.5B"
  max_length: 2048

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

quantization:
  bits: 4
  quant_type: "nf4"
  compute_dtype: "bfloat16"
```

Override any value via environment variable: `LFP_MODEL_NAME=deepseek-ai/deepseek-coder-1.3b make train`

## Distributed Training

Multi-GPU training via DeepSpeed ZeRO Stage 2 with CPU optimizer offloading:

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
  -m src.training.finetune --config configs/finetune_config.yaml
```

## Testing

```bash
make test       # 285 tests, all CPU-compatible
make lint       # ruff lint + format check
make gpu-status # Check GPU availability and VRAM
make gpu-preflight # Run before any GPU job; see AGENTS.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Training | PyTorch, HuggingFace (Transformers, PEFT, TRL), BitsAndBytes |
| Distributed | Accelerate, DeepSpeed ZeRO Stage 2 |
| Data | Polars, Datasketch (MinHash LSH), AST validation |
| Serving | vLLM, FastAPI, Continuous Batching |
| Monitoring | Prometheus metrics, structured JSON logging, PyTorch Profiler |
| Container | Docker (CUDA 12.1), Docker Compose, NVIDIA Container Toolkit |
| CI/CD | GitHub Actions (ruff + mypy + pytest + Docker build) |
| GPU | pynvml/NVML, VRAM tracking, OOM prediction, cost tracking |

## License

MIT
