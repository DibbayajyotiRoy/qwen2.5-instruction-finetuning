"""Dataset loading, formatting, and splitting for instruction tuning."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.utils import resolve_path


def format_alpaca_example(
    example: dict[str, Any],
    template: str,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    eos_token: str = "",
    target_field: str | None = None,
) -> dict[str, str]:
    """Format a single example using an instruction template.

    Args:
        example: Raw dataset example.
        template: Jinja-like template string with {instruction}, {input}, {output} placeholders.
        instruction_field: Field name for the instruction.
        input_field: Field name for optional context.
        output_field: Field name for the expected response.
        eos_token: End-of-sequence token to append.
        target_field: Optional alternate field to use as the response (e.g. teacher_output).

    Returns:
        Dictionary with the formatted text.
    """
    instruction = str(example.get(instruction_field, "")).strip()
    input_text = str(example.get(input_field, "")).strip()
    response_field = target_field if target_field else output_field
    output_text = str(example.get(response_field, "")).strip()
    if not output_text:
        # Fall back to original output if target field is empty/missing.
        output_text = str(example.get(output_field, "")).strip()

    text = template.format(
        instruction=instruction,
        input=input_text,
        output=output_text,
    )
    if eos_token:
        text += eos_token

    return {"text": text}


def load_instruction_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool = False,
    max_samples: int | None = None,
) -> Dataset:
    """Load a Hugging Face instruction dataset.

    Args:
        dataset_name: Hugging Face dataset name or local path.
        config_name: Dataset config name.
        split: Dataset split to load.
        cache_dir: Cache directory for datasets.
        streaming: Whether to stream the dataset.
        max_samples: Optional cap on number of examples.

    Returns:
        Loaded dataset.
    """
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "name": config_name,
        "split": split,
        "cache_dir": resolve_path(cache_dir) if cache_dir else None,
        "streaming": streaming,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    dataset = load_dataset(**kwargs)
    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def build_dataset(
    data_config: dict[str, Any],
    tokenizer: Any,
    eval_size: float = 0.05,
    seed: int = 42,
    eos_token: str = "",
    target_field: str | None = None,
) -> DatasetDict:
    """Build a formatted train/validation dataset for instruction tuning.

    Args:
        data_config: Data configuration dictionary.
        tokenizer: Tokenizer (used for EOS token if not provided).
        eval_size: Fraction of training data to hold out for validation.
        seed: Random seed for splitting.
        eos_token: EOS token to append to each example.
        target_field: Optional alternate response field for formatting.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    dataset_name = data_config["dataset_name"]
    config_name = data_config.get("dataset_config")
    split = data_config.get("split", "train")
    cache_dir = data_config.get("cache_dir")
    streaming = data_config.get("streaming", False)
    max_samples = data_config.get("max_samples")

    instruction_field = data_config.get("instruction_field", "instruction")
    input_field = data_config.get("input_field", "input")
    output_field = data_config.get("output_field", "output")
    template = data_config.get("template", "{instruction}\n{output}")

    if not eos_token and tokenizer is not None:
        eos_token = tokenizer.eos_token or ""

    dataset = load_instruction_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        max_samples=max_samples,
    )

    dataset = dataset.map(
        lambda ex: format_alpaca_example(
            ex,
            template=template,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
            eos_token=eos_token,
            target_field=target_field,
        ),
        remove_columns=dataset.column_names,
        desc="Formatting instructions",
    )

    if eval_size and eval_size > 0 and not streaming:
        split_result = dataset.train_test_split(test_size=eval_size, seed=seed)
        return DatasetDict({"train": split_result["train"], "validation": split_result["test"]})

    return DatasetDict({"train": dataset})
