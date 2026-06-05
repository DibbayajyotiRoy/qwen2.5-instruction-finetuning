"""Merge LoRA adapters and export fine-tuned Qwen2.5 to GGUF."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.logging_utils import get_logger, setup_logging
from src.model import load_base_model
from src.utils import load_config

logger = get_logger(__name__)


def merge_and_save_adapter(
    checkpoint_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Merge LoRA adapters into the base model and save the full weights.

    Args:
        checkpoint_dir: Directory containing adapter weights.
        output_dir: Directory to write merged model.
        config: Full project configuration.

    Returns:
        Path to the merged model directory.
    """
    logger.info("Loading base model for merge: %s", config["model"]["base_model"])
    model, tokenizer = load_base_model(
        model_name=config["model"]["base_model"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=config["model"].get("dtype"),
        load_in_4bit=False,  # merged model must be full precision
        use_gradient_checkpointing=False,
        trust_remote_code=config["model"].get("trust_remote_code", False),
        revision=config["model"].get("revision", "main"),
    )

    logger.info("Loading adapter weights from %s", checkpoint_dir)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    logger.info("Merging adapters into base model")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def export_to_gguf(
    merged_model_dir: Path,
    output_dir: Path,
    quantization_methods: list[str],
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> list[Path]:
    """Export a merged model to GGUF using llama.cpp conversion scripts.

    Args:
        merged_model_dir: Path to merged HF model.
        output_dir: Directory to write GGUF files.
        quantization_methods: List of quant methods, e.g. ["q4_k_m", "q8_0"].
        push_to_hub: Whether to push to Hugging Face Hub.
        hub_model_id: Hub model id if pushing.

    Returns:
        List of generated GGUF file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Attempt to use Unsloth's native GGUF export if available.
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(merged_model_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        for method in quantization_methods:
            save_path = output_dir / f"{merged_model_dir.name}-{method}.gguf"
            logger.info("Exporting GGUF with Unsloth: %s", save_path)
            model.save_pretrained_gguf(
                str(save_path.parent),
                tokenizer,
                quantization_method=method,
            )
            generated.append(save_path)
        if push_to_hub and hub_model_id:
            model.push_to_hub_gguf(
                hub_model_id,
                tokenizer,
                quantization_method=quantization_methods[0],
            )
        return generated
    except Exception as exc:  # pragma: no cover
        logger.warning("Unsloth GGUF export failed (%s); falling back to llama.cpp", exc)

    # Fallback: use llama.cpp convert script.
    llama_cpp_dir = Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp"
    if not llama_cpp_dir.exists():
        logger.info("Cloning llama.cpp for GGUF conversion")
        llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)],
            check=True,
        )

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert.py"

    base_gguf = output_dir / f"{merged_model_dir.name}-f16.gguf"
    logger.info("Converting HF model to F16 GGUF: %s", base_gguf)
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_model_dir), "--outfile", str(base_gguf), "--outtype", "f16"],
        check=True,
    )
    generated.append(base_gguf)

    quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_binary.exists():
        logger.info("Building llama.cpp quantize binary")
        build_dir = llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", "..", "-DLLAMA_CUDA=OFF"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"], cwd=build_dir, check=True)

    for method in quantization_methods:
        if method == "f16":
            continue
        quant_path = output_dir / f"{merged_model_dir.name}-{method}.gguf"
        logger.info("Quantizing to %s: %s", method, quant_path)
        subprocess.run(
            [str(quantize_binary), str(base_gguf), str(quant_path), method],
            check=True,
        )
        generated.append(quant_path)

    return generated


def main() -> None:
    """CLI entry point for exporting to GGUF."""
    parser = argparse.ArgumentParser(description="Export fine-tuned Qwen2.5 to GGUF")
    parser.add_argument("checkpoint", help="Path to adapter checkpoint directory")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--output", help="Override output directory for merged model")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()

    checkpoint_dir = Path(args.checkpoint)
    merged_output = Path(args.output) if args.output else checkpoint_dir.parent / "final_merged"

    merge_and_save_adapter(checkpoint_dir, merged_output, config)

    gguf_files = export_to_gguf(
        merged_model_dir=merged_output,
        output_dir=Path(config["export"]["output_dir"]),
        quantization_methods=config["export"]["quantization_methods"],
        push_to_hub=config["export"].get("push_to_hub", False),
        hub_model_id=config["export"].get("hub_model_id"),
    )

    logger.info("Generated GGUF files:")
    for path in gguf_files:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
