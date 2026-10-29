"""Model loading helpers for Unsloth Qwen2.5 fine-tuning."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel, PreTrainedTokenizer

# Guard Unsloth import so tests can run without it installed.
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except Exception:  # pragma: no cover - unsloth may not be installed on CPU-only runners
    FastLanguageModel = None  # type: ignore[assignment, misc]
    UNSLOTH_AVAILABLE = False


def load_base_model(
    model_name: str,
    max_seq_length: int,
    dtype: Any = None,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: bool | str = True,
    trust_remote_code: bool = False,
    revision: str = "main",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a Qwen2.5 base model and tokenizer via Unsloth.

    Args:
        model_name: Hugging Face model id or local path.
        max_seq_length: Maximum sequence length.
        dtype: Optional dtype override.
        load_in_4bit: Use 4-bit quantization.
        use_gradient_checkpointing: Gradient checkpointing strategy.
        trust_remote_code: Allow custom model code.
        revision: Model revision to pin.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        RuntimeError: If Unsloth is not installed.
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "Unsloth is not installed or failed to import. "
            "Install it with `pip install unsloth`."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    return model, tokenizer


def build_lora_model(
    model: PreTrainedModel,
    lora_config: dict[str, Any],
) -> PreTrainedModel:
    """Apply LoRA adapters to a loaded model.

    Args:
        model: Base model returned by Unsloth.
        lora_config: LoRA configuration dictionary.

    Returns:
        Model with LoRA adapters applied.
    """
    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config.get("lora_dropout", 0.0),
        bias=lora_config.get("bias", "none"),
        task_type="CAUSAL_LM",
        use_rslora=lora_config.get("use_rslora", False),
        init_lora_weights=lora_config.get("init_lora_weights", True),
    )

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", False),
        )

    model = get_peft_model(model, config)
    return model


def load_model_for_inference(
    model_path: str,
    max_seq_length: int = 2048,
    dtype: Any = None,
    load_in_4bit: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a merged/full model for inference.

    Args:
        model_path: Path to the merged model checkpoint.
        max_seq_length: Maximum sequence length.
        dtype: Optional dtype.
        load_in_4bit: Use 4-bit quantization for inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    if UNSLOTH_AVAILABLE:
        return FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype or "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer
