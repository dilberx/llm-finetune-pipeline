#!/usr/bin/env bash
# ============================================================================
# End-to-end pipeline: curate data → fine-tune → evaluate → serve
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "  LLM Fine-Tuning Pipeline - Full Run"
echo "================================================"

# --- Stage 1: Data Curation ---
echo ""
echo "[1/4] Curating training data..."
python -m src.data.curate --config configs/data_config.yaml

# --- Stage 2: Fine-Tuning ---
echo ""
echo "[2/4] Fine-tuning model with QLoRA..."
python -m src.training.finetune --config configs/finetune_config.yaml

# --- Stage 3: Evaluation ---
echo ""
echo "[3/4] Evaluating base vs fine-tuned model..."
python -m src.evaluation.evaluate --config configs/eval_config.yaml

# --- Stage 4: Serve (optional) ---
if [[ "${SERVE:-false}" == "true" ]]; then
    echo ""
    echo "[4/4] Starting model server..."
    python -m src.serving.server \
        --model "Qwen/Qwen2.5-Coder-1.5B" \
        --adapter "./checkpoints/best" \
        --port 8000
else
    echo ""
    echo "[4/4] Skipping serving (set SERVE=true to enable)"
fi

echo ""
echo "================================================"
echo "  Pipeline complete! Check ./reports/ for results"
echo "================================================"
