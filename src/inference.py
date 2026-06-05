"""Hugging Face inference engine for fine-tuned Qwen2.5 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.logging_utils import get_logger, setup_logging
from src.model import load_model_for_inference
from src.utils import load_config

logger = get_logger(__name__)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generation_kwargs: dict[str, Any],
) -> str:
    """Generate a single response from the model.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: Input prompt string.
        generation_kwargs: Generation hyperparameters.

    Returns:
        Decoded response string.
    """
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_inference(
    model_path: str,
    prompts: list[str],
    config: dict[str, Any] | None = None,
    dtype: Any = None,
    load_in_4bit: bool = False,
) -> list[dict[str, str]]:
    """Run batch inference over a list of prompts.

    Args:
        model_path: Path to model checkpoint.
        prompts: List of input prompts.
        config: Optional inference config.
        dtype: Optional dtype override.
        load_in_4bit: Load model in 4-bit.

    Returns:
        List of result dicts with prompt and response.
    """
    config = config or {}
    inference_cfg = config.get("inference", {})
    generation_kwargs = {
        "max_new_tokens": inference_cfg.get("max_new_tokens", 512),
        "temperature": inference_cfg.get("temperature", 0.7),
        "top_p": inference_cfg.get("top_p", 0.9),
        "top_k": inference_cfg.get("top_k", 50),
        "repetition_penalty": inference_cfg.get("repetition_penalty", 1.0),
        "do_sample": inference_cfg.get("do_sample", True),
    }

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        max_seq_length=config.get("model", {}).get("max_seq_length", 2048),
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    results = []
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt, generation_kwargs)
        results.append({"prompt": prompt, "response": response})
    return results


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Qwen2.5")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/inference_config.yaml", help="Inference config")
    parser.add_argument("--prompt", help="Single prompt string")
    parser.add_argument("--prompts_file", help="JSONL file with prompts")
    parser.add_argument("--output", help="Output JSONL file")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit inference")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    else:
        prompts.append("Explain the importance of reproducibility in machine learning.")

    results = run_inference(
        model_path=args.model_path,
        prompts=prompts,
        config=config,
        load_in_4bit=args.load_in_4bit,
    )

    for r in results:
        print(f"Prompt: {r['prompt']}\nResponse: {r['response']}\n{'-' * 40}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d results to %s", len(results), args.output)


if __name__ == "__main__":
    main()
