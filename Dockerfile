# ============================================================================
# Multi-stage Dockerfile for LLM Fine-Tuning & Evaluation Pipeline
#
# Stage 1 (base): CUDA runtime + Python + system deps
# Stage 2 (app):  Application code + pip dependencies
#
# Supports GPU passthrough via NVIDIA Container Toolkit
# Usage:
#   docker build -t llm-finetune-pipeline .
#   docker run --gpus all llm-finetune-pipeline python -m src.training.finetune
# ============================================================================

# --- Stage 1: CUDA base with Python ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# --- Stage 2: Application ---
FROM base AS app

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    torch>=2.1.0 \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 \
    trl>=0.8.0 \
    wandb>=0.16.0 \
    tensorboard>=2.16.0 \
    pyyaml>=6.0 \
    rich>=13.0 \
    polars>=0.20.0 \
    datasketch>=1.6.0 \
    evaluate>=0.4.0 \
    scipy>=1.12.0 \
    matplotlib>=3.8.0 \
    pytest>=8.0 \
    ruff>=0.3.0

# Copy application code
COPY . .

# Create directories for outputs
RUN mkdir -p /app/data/curated /app/checkpoints /app/reports /app/logs

# Health check — verifies PyTorch + CUDA availability
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" || exit 1

# Default: run the full pipeline
ENTRYPOINT ["python"]
CMD ["-m", "src.training.finetune", "--config", "configs/finetune_config.yaml"]
