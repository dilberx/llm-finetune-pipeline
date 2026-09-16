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
│  Tests: 80 unit tests (CPU-compatible, no GPU required)              │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Status

A full QLoRA fine-tuning run has been completed on this pipeline.

- **Base model**: Qwen/Qwen2.5-Coder-1.5B
- **Method**: QLoRA (4-bit NF4, LoRA r=16, α=32)
- **Training**: ~50K steps, ~22 hours on a single RTX 3080 (10 GB VRAM)
- **Training data**: ~134K curated code instruction examples (ChatML format)
- **Adapter**: saved to `checkpoints/best/`
- **Training crashed 3 times** during the run — checkpoint fault tolerance handled all three resumes without data loss

### Evaluation Results

Two evaluations were run: one on the target code distribution, one on generic benchmarks.

**Target-domain evaluation** (16 held-out code completions + 128 perplexity samples from the curated distribution):

| Metric | Base | Fine-tuned | Change |
|--------|------|-----------|--------|
| Perplexity | 3.27 | 3.04 | −7.0% (better) |
| Code completion token F1 | 0.10 | 0.23 | +128% (better) |
| Code completion exact match | 0.00 | 0.00 | metric ceiling — see below |

**Generic benchmark** (3-problem HumanEval sample + Wikitext perplexity):

| Metric | Base | Fine-tuned | Change |
|--------|------|-----------|--------|
| Wikitext perplexity | 10.33 | 12.54 | +21% (worse) |
| HumanEval pass@1 | 0.00 | 0.00 | no change |

**What this means**: The adapter improved on the target code distribution — lower perplexity and better token overlap on completions. Generic prose perplexity got worse, which is expected when you specialize a model on code. The HumanEval sample is too small (3 problems) to draw conclusions from. This is a directional signal on a small eval set, not a definitive benchmark result.

> **Superseded — see the 200-example results below.** The table above comes from a
> 16-example evaluation, which was too small to resolve the effects that matter. Every
> metric it reported as zero-for-both turned out to be rare rather than absent: `ast_match`
> read 0/16 for both models, then 15/200 at p=0.0001 on a larger set.
>
> `exact_match` is also confounded rather than broken. Every reference is wrapped in
> markdown code fences, so it scores correct code as a miss unless the fence markers are
> reproduced too — it measures format alongside correctness. It is not a hard ceiling:
> the fine-tuned model produced 2 exact matches in 200.
>
> A separate defect in the synthesis templates left roughly 70% of generated training
> prompts syntactically invalid (a docstring indented to a fixed 4 spaces regardless of
> whether the function was a method). That is fixed, and curation now aborts when too many
> synthesized examples fail to parse. **The numbers above predate that fix.**

### 200-example evaluation

Run on 200 held-out completions with a disjoint 600-example proxy pool. Same adapter, same
base model; a larger and more informative eval set.

| Metric | Base | Fine-tuned | Δ | 95% CI | p |
|--------|------|-----------|---|--------|---|
| Parse rate | 0.235 | **0.810** | +0.575 | [+0.500, +0.650] | <0.0001 |
| Token F1 | 0.214 | 0.492 | +0.278 | [+0.244, +0.313] | <0.0001 |
| Edit similarity | 0.088 | 0.357 | +0.269 | [+0.231, +0.308] | <0.0001 |
| AST match | 0.000 | 0.075 | +0.075 | [+0.040, +0.115] | 0.0001 |
| Exact match | 0.000 | 0.010 | +0.010 | [+0.000, +0.025] | 0.157 (n.s.) |

**What this means**: the dominant effect is that the fine-tuned model produces *parseable,
correctly formatted Python*. The base model manages it under a quarter of the time —
answering in prose, or in one case in Java. Structural correctness improves too, but
remains rare at 7.5%. Generic Wikitext perplexity still regresses; specializing on code
costs general language modeling.

A prediction-powered estimate over the disjoint pool puts the token-F1 effect at +0.303
[+0.276, +0.331], with the cheap proxy correlating at r=0.69 — the 200 generated examples
carry the statistical weight of about 308.

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
│   │   └── metrics.py        # Statistical comparison with bootstrap CIs
│   ├── serving/
│   │   └── server.py         # vLLM server with Prometheus /metrics endpoint
│   └── utils/
│       ├── config.py         # YAML + env + dataclass config system
│       ├── gpu.py            # GPU resource manager + OOM prediction
│       ├── logging.py        # Structured JSON logging with correlation IDs
│       ├── profiler.py       # Training profiler (PyTorch profiler wrapper)
│       └── metrics_export.py # Prometheus exporter + cost tracker
├── configs/
│   ├── finetune_config.yaml
│   ├── data_config.yaml
│   ├── eval_config.yaml
│   └── deepspeed_config.json # DeepSpeed ZeRO Stage 2 for multi-GPU
├── tests/                    # 80 unit tests (CPU-compatible)
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
make test       # 80 tests, all CPU-compatible
make lint       # ruff lint + format check
make gpu-status # Check GPU availability and VRAM
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
