"""Benchmark HF vs llama.cpp inference throughput and memory."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.inference import generate_response
from src.logging_utils import get_logger, setup_logging
from src.model import load_model_for_inference
from src.utils import load_config

logger = get_logger(__name__)


def get_gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB if CUDA is available, else 0."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def get_cpu_memory_mb() -> float:
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024**2


def benchmark_hf(
    model_path: str,
    prompts: list[str],
    config: dict[str, Any],
    load_in_4bit: bool = False,
    num_runs: int = 3,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Benchmark full-precision Hugging Face inference.

    Args:
        model_path: Path to HF model.
        prompts: List of prompts.
        config: Inference config.
        load_in_4bit: Use 4-bit loading.
        num_runs: Number of benchmark repetitions.
        max_tokens: Generation length.

    Returns:
        Dictionary of benchmark metrics.
    """
    inference_cfg = config.get("inference", {})
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": inference_cfg.get("temperature", 0.7),
        "top_p": inference_cfg.get("top_p", 0.9),
        "top_k": inference_cfg.get("top_k", 50),
        "repetition_penalty": inference_cfg.get("repetition_penalty", 1.0),
        "do_sample": inference_cfg.get("do_sample", True),
    }

    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        max_seq_length=config.get("model", {}).get("max_seq_length", 2048),
        load_in_4bit=load_in_4bit,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    tokens_generated: list[int] = []

    for _ in range(num_runs):
        start = time.perf_counter()
        for prompt in prompts:
            response = generate_response(model, tokenizer, prompt, generation_kwargs)
            tokens_generated.append(len(tokenizer.encode(response)))
        latencies.append(time.perf_counter() - start)

    metrics = {
        "backend": "huggingface",
        "model_path": model_path,
        "num_prompts": len(prompts),
        "num_runs": num_runs,
        "total_time_s": mean(latencies),
        "avg_latency_s": mean(latencies) / num_runs,
        "tokens_per_second": sum(tokens_generated) / sum(latencies),
        "peak_gpu_memory_mb": get_gpu_memory_mb(),
        "peak_cpu_memory_mb": get_cpu_memory_mb(),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def benchmark_llamacpp(
    gguf_path: str,
    prompts: list[str],
    config: dict[str, Any],
    num_runs: int = 3,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Benchmark quantized llama.cpp inference.

    Args:
        gguf_path: Path to GGUF file.
        prompts: List of prompts.
        config: Inference config.
        num_runs: Number of benchmark repetitions.
        max_tokens: Generation length.

    Returns:
        Dictionary of benchmark metrics.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is required for llama.cpp benchmarking") from exc

    n_gpu_layers = -1 if torch.cuda.is_available() else 0
    model = Llama(
        model_path=gguf_path,
        n_ctx=config.get("model", {}).get("max_seq_length", 2048),
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    latencies: list[float] = []
    tokens_generated: list[int] = []

    for _ in range(num_runs):
        start = time.perf_counter()
        for prompt in prompts:
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=config.get("inference", {}).get("temperature", 0.7),
                top_p=config.get("inference", {}).get("top_p", 0.9),
                repeat_penalty=config.get("inference", {}).get("repetition_penalty", 1.0),
            )
            tokens_generated.append(output["usage"]["completion_tokens"])
        latencies.append(time.perf_counter() - start)

    metrics = {
        "backend": "llama.cpp",
        "model_path": gguf_path,
        "num_prompts": len(prompts),
        "num_runs": num_runs,
        "total_time_s": mean(latencies),
        "avg_latency_s": mean(latencies) / num_runs,
        "tokens_per_second": sum(tokens_generated) / sum(latencies),
        "peak_gpu_memory_mb": get_gpu_memory_mb(),
        "peak_cpu_memory_mb": get_cpu_memory_mb(),
    }
    return metrics


def load_prompts(prompts_file: str | None) -> list[str]:
    """Load prompts from a JSONL file or return defaults.

    Args:
        prompts_file: Path to JSONL file.

    Returns:
        List of prompt strings.
    """
    default_prompts = [
        "Explain quantum computing in one paragraph.",
        "Write a Python function to compute factorial.",
        "Summarize the key principles of machine learning.",
    ]
    if not prompts_file:
        return default_prompts

    prompts = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts or default_prompts


def main() -> None:
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark HF vs llama.cpp inference")
    parser.add_argument("hf_model", help="Path to full-precision HF model")
    parser.add_argument("gguf_model", help="Path to quantized GGUF model")
    parser.add_argument("--config", default="configs/inference_config.yaml", help="Config file")
    parser.add_argument("--num_runs", type=int, default=3, help="Benchmark repetitions")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--output_dir", default="outputs/benchmarks", help="Output directory")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    prompts = load_prompts(config.get("benchmark", {}).get("prompts_file"))

    logger.info("Benchmarking Hugging Face model: %s", args.hf_model)
    hf_metrics = benchmark_hf(
        model_path=args.hf_model,
        prompts=prompts,
        config=config,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
    )
    logger.info("HF metrics: %s", hf_metrics)

    logger.info("Benchmarking llama.cpp model: %s", args.gguf_model)
    gguf_metrics = benchmark_llamacpp(
        gguf_path=args.gguf_model,
        prompts=prompts,
        config=config,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
    )
    logger.info("llama.cpp metrics: %s", gguf_metrics)

    report = {
        "huggingface": hf_metrics,
        "llamacpp": gguf_metrics,
        "speedup": hf_metrics["total_time_s"] / gguf_metrics["total_time_s"],
        "memory_ratio": hf_metrics["peak_gpu_memory_mb"] / max(gguf_metrics["peak_gpu_memory_mb"], 1.0),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Benchmark report saved to %s", output_file)


if __name__ == "__main__":
    main()
