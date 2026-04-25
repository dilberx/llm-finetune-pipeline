"""
Model serving with vLLM for optimized inference.

Features:
- High-throughput serving with continuous batching
- KV-cache management
- LoRA adapter hot-loading
- Basic monitoring (latency, throughput, GPU utilization)
- FastAPI REST endpoint
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

import yaml
from rich.console import Console

console = Console()


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    adapter_path: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64
    quantization: str | None = "awq"  # None, "awq", "gptq", "squeezellm"
    trust_remote_code: bool = True
    dtype: str = "auto"
    seed: int = 42


@dataclass
class RequestMetrics:
    """Tracks per-request serving metrics."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    _start_time: float = field(default_factory=time.time)

    def record(self, tokens_generated: int, latency_ms: float) -> None:
        """Record metrics for a completed request."""
        self.total_requests += 1
        self.total_tokens_generated += tokens_generated
        self.total_latency_ms += latency_ms

        elapsed = time.time() - self._start_time
        self.requests_per_second = self.total_requests / elapsed if elapsed > 0 else 0
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        self.avg_tokens_per_second = (
            self.total_tokens_generated / (self.total_latency_ms / 1000)
            if self.total_latency_ms > 0 else 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as a dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "requests_per_second": round(self.requests_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
        }


class ModelServer:
    """
    High-throughput model server using vLLM.

    Supports serving base models and models with LoRA adapters,
    with automatic continuous batching and KV-cache optimization.
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.metrics = RequestMetrics()
        self.engine = None

    def _init_engine(self) -> None:
        """Initialize the vLLM inference engine."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError:
            console.print("[red]vLLM not installed. Install with: pip install vllm[/red]")
            raise

        engine_kwargs = {
            "model": self.config.model_name,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "trust_remote_code": self.config.trust_remote_code,
            "dtype": self.config.dtype,
            "seed": self.config.seed,
        }

        if self.config.quantization:
            engine_kwargs["quantization"] = self.config.quantization

        # Enable LoRA if adapter path is specified
        if self.config.adapter_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = 64

        console.print(f"[bold blue]Initializing vLLM engine:[/bold blue] {self.config.model_name}")
        self.engine = LLM(**engine_kwargs)
        console.print("[bold green]Engine ready![/bold green]")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate completions for a batch of prompts.

        Uses vLLM's continuous batching for optimal throughput.
        """
        if self.engine is None:
            self._init_engine()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        # Build LoRA request if adapter is configured
        lora_request = None
        if self.config.adapter_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("finetuned", 1, self.config.adapter_path)

        start_time = time.time()
        outputs = self.engine.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        results = []
        total_tokens = 0
        for output in outputs:
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            total_tokens += num_tokens
            results.append({
                "generated_text": generated_text,
                "num_tokens": num_tokens,
                "finish_reason": output.outputs[0].finish_reason,
            })

        self.metrics.record(total_tokens, elapsed_ms)
        return results

    def start_api_server(self) -> None:
        """Start the FastAPI REST server for model serving."""
        try:
            import uvicorn
            from fastapi import FastAPI
            from pydantic import BaseModel
        except ImportError:
            console.print("[red]FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn[/red]")
            raise

        if self.engine is None:
            self._init_engine()

        app = FastAPI(title="LLM Fine-Tune Pipeline - Model Server")

        class CompletionRequest(BaseModel):
            prompt: str | list[str]
            max_tokens: int = 512
            temperature: float = 0.7
            top_p: float = 0.95
            stop: list[str] | None = None

        class CompletionResponse(BaseModel):
            completions: list[dict[str, Any]]
            metrics: dict[str, Any]

        @app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest):
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            results = self.generate(
                prompts=prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            return CompletionResponse(
                completions=results,
                metrics=self.metrics.to_dict(),
            )

        @app.get("/v1/metrics")
        async def get_metrics():
            return self.metrics.to_dict()

        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus-compatible metrics endpoint for monitoring systems."""
            from fastapi.responses import PlainTextResponse
            from src.utils.metrics_export import MetricsRegistry
            from src.utils.gpu import GPUResourceManager

            registry = MetricsRegistry()
            gpu_mgr = GPUResourceManager()

            # Export serving metrics
            m = self.metrics
            registry.counter("serving_requests_total", m.total_requests)
            registry.counter("serving_tokens_generated_total", m.total_tokens_generated)
            registry.gauge("serving_latency_ms", m.avg_latency_ms)
            registry.gauge("serving_tokens_per_second", m.avg_tokens_per_second)
            registry.gauge("serving_requests_per_second", m.requests_per_second)

            # Export GPU metrics
            for snap in gpu_mgr.get_all_snapshots():
                labels = {"device": str(snap.device_id), "gpu": snap.device_name}
                registry.gauge("gpu_memory_used_mb", snap.used_mb, labels)
                registry.gauge("gpu_memory_total_mb", snap.total_mb, labels)
                registry.gauge("gpu_utilization_pct", snap.utilization_pct, labels)
                registry.gauge("gpu_temperature_celsius", snap.temperature_c, labels)
                registry.gauge("gpu_power_watts", snap.power_w, labels)

            return PlainTextResponse(registry.to_prometheus(), media_type="text/plain")

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": self.config.model_name}

        @app.get("/ready")
        async def readiness():
            """Kubernetes readiness probe — only returns OK when model is loaded."""
            if self.engine is None:
                from fastapi.responses import JSONResponse
                return JSONResponse({"status": "not_ready"}, status_code=503)
            return {"status": "ready", "model": self.config.model_name}

        console.print(f"\n[bold green]Starting server on {self.config.host}:{self.config.port}[/bold green]")
        uvicorn.run(app, host=self.config.host, port=self.config.port)


def main():
    """CLI entry point for model serving."""
    parser = argparse.ArgumentParser(description="Serve a fine-tuned code model via REST API")
    parser.add_argument("--config", type=str, default=None, help="Path to serving config YAML")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        config = ServingConfig(**{k: v for k, v in raw.items() if hasattr(ServingConfig, k)})
    else:
        config = ServingConfig()

    if args.model:
        config.model_name = args.model
    if args.adapter:
        config.adapter_path = args.adapter
    if args.port:
        config.port = args.port

    server = ModelServer(config)
    server.start_api_server()


if __name__ == "__main__":
    main()
