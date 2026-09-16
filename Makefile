.PHONY: help install lint test test-gpu build up train eval serve profile clean

# ============================================================================
# LLM Fine-Tuning Pipeline — Makefile
# ============================================================================

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- Development ---

install: ## Install all dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cu121
	pip install transformers peft datasets accelerate bitsandbytes trl
	pip install vllm fastapi uvicorn
	pip install polars datasketch scipy evaluate matplotlib
	pip install pytest ruff mypy rich pyyaml tensorboard

lint: ## Run linting and format check
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/

test: ## Run tests (CPU-only)
	pytest tests/ -v --tb=short

test-gpu: ## Run tests including GPU-dependent tests
	pytest tests/ -v --tb=short -m "not slow"

# --- Docker ---

build: ## Build Docker image
	docker build -t llm-finetune-pipeline .

up: ## Start all services (train → eval → serve)
	docker compose up --build

# Prefer the project virtualenv; fall back to whatever python is on PATH.
# A bare `python` does not exist on many systems (Debian/Ubuntu ship only
# python3), which silently broke every target below with "No such file".
PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; \
	elif command -v python >/dev/null 2>&1; then echo python; \
	else echo python3; fi)

# --- Pipeline Stages ---

curate: ## Run data curation pipeline
	$(PYTHON) -m src.data.curate --config configs/data_config.yaml

train: ## Run fine-tuning
	$(PYTHON) -m src.training.finetune --config configs/finetune_config.yaml

train-resume: ## Resume training from latest checkpoint
	$(PYTHON) -m src.training.finetune --config configs/finetune_config.yaml --resume-from auto

eval: ## Run evaluation
	$(PYTHON) -m src.evaluation.evaluate --config configs/eval_config.yaml

sweep: ## Score checkpoints and calibrate the eval set (IRT)
	$(PYTHON) -m src.evaluation.sweep --config configs/eval_config.yaml

serve: ## Start model server
	$(PYTHON) -m src.serving.server --config configs/serving_config.yaml

# --- Profiling & Monitoring ---

profile: ## Profile training for 5 steps and generate report
	LOG_LEVEL=DEBUG python -c " \
		from src.utils.profiler import TrainingProfiler; \
		p = TrainingProfiler(); \
		print('Profiler ready. Run training with profiler integration.') \
	"

gpu-status: ## Show GPU status and memory
	$(PYTHON) -c " \
		from src.utils.gpu import GPUResourceManager; \
		mgr = GPUResourceManager(); \
		for s in mgr.get_all_snapshots(): \
			print(f'GPU {s.device_id}: {s.device_name} | {s.used_mb:.0f}/{s.total_mb:.0f} MB ({s.used_pct:.1f}%) | {s.temperature_c}°C | {s.power_w:.0f}W'); \
		if not mgr.get_all_snapshots(): print('No GPUs detected') \
	"

# --- Cleanup ---

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
