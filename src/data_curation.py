"""Data curation utilities for efficient instruction tuning.

Implements ideas from recent work on data-centric efficiency:
- Concision-aware selection (inference efficiency through brevity)
- Perplexity / quality scoring
- Influence-based filtering
- Teacher soft-label generation for knowledge distillation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def concision_score(text: str, tokenizer: PreTrainedTokenizer) -> float:
    """Score how concise a text is while preserving information.

    Lower is more concise. Returns token count normalized by word count to
    penalise redundancy and filler. This aligns with the finding that training
    on concise, correct data reduces inference Cost-of-Pass.

    Args:
        text: Response text.
        tokenizer: Tokenizer used for the model.

    Returns:
        Concision score (tokens per word), lower is better.
    """
    words = [w for w in text.split() if w.strip()]
    if not words:
        return float("inf")
    num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return num_tokens / len(words)


def quality_score(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """Compute perplexity-based quality score for a response.

    Lower perplexity => higher quality. This is a lightweight proxy for
    value-centric data curation.

    Args:
        text: Response text.
        model: Reference language model.
        tokenizer: Tokenizer.
        device: Torch device.
        max_length: Max tokens to consider.

    Returns:
        Perplexity score (lower is better).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def compute_example_scores(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizer,
    response_field: str = "output",
) -> dict[str, list[float]]:
    """Compute concision scores for a batch of examples.

    Args:
        examples: Batch dict from datasets.
        tokenizer: Tokenizer.
        response_field: Field containing the response.

    Returns:
        Dictionary with 'concision_score' list.
    """
    scores = [concision_score(str(text), tokenizer) for text in examples[response_field]]
    return {"concision_score": scores}


def curate_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    response_field: str = "output",
    keep_top_fraction: float = 0.8,
    max_tokens_per_word: float | None = None,
    min_tokens_per_example: int = 4,
) -> Dataset:
    """Curate a dataset by selecting concise, non-degenerate examples.

    Args:
        dataset: Input dataset.
        tokenizer: Tokenizer.
        response_field: Field holding the response text.
        keep_top_fraction: Fraction of examples to keep by concision.
        max_tokens_per_word: Hard cutoff for tokens-per-word (verbosity).
        min_tokens_per_example: Minimum token length to avoid empty answers.

    Returns:
        Curated dataset.
    """
    scored = dataset.map(
        lambda ex: compute_example_scores(ex, tokenizer, response_field),
        batched=True,
        desc="Scoring concision",
    )

    # Filter out empty / too-short responses.
    def is_valid(ex: dict[str, Any]) -> bool:
        text = str(ex.get(response_field, "")).strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < min_tokens_per_example:
            return False
        return not (max_tokens_per_word is not None and ex["concision_score"] > max_tokens_per_word)

    valid = scored.filter(is_valid, desc="Filtering degenerate examples")

    if len(valid) == 0:
        return valid

    # Keep the most concise fraction.
    scores = np.array(valid["concision_score"])
    threshold = np.percentile(scores, 100 * (1 - keep_top_fraction))
    curated = valid.filter(lambda ex: ex["concision_score"] <= threshold, desc="Keeping concise data")

    # Drop temporary score column.
    if "concision_score" in curated.column_names:
        curated = curated.remove_columns(["concision_score"])

    return curated


@torch.inference_mode()
def generate_teacher_labels(
    dataset: Dataset,
    teacher_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: torch.device | None = None,
) -> Dataset:
    """Generate teacher soft labels / chain-of-thought responses for distillation.

    Uses a larger teacher model to produce high-quality targets. The student
    can then be trained with a mixture of hard labels and KL divergence to
    the teacher distribution, matching the distillation paradigm used by
    OpenAI/Anthropic for efficient small-model training.

    Args:
        dataset: Input dataset.
        teacher_model: Larger teacher language model.
        tokenizer: Tokenizer shared by teacher and student.
        prompt_template: Template string for formatting instructions.
        instruction_field: Instruction field name.
        input_field: Optional input/context field name.
        output_field: Output field name.
        max_new_tokens: Max teacher generation length.
        temperature: Sampling temperature.
        device: Torch device.

    Returns:
        Dataset augmented with 'teacher_output' and optionally 'teacher_logits'.
    """
    device = device or teacher_model.device
    teacher_model.eval()

    def format_prompt(example: dict[str, Any]) -> str:
        return prompt_template.format(
            instruction=str(example.get(instruction_field, "")).strip(),
            input=str(example.get(input_field, "")).strip(),
            output="",
        ).rstrip()

    teacher_outputs = []
    for example in dataset:
        prompt = format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        output_ids = teacher_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        teacher_text = tokenizer.decode(generated, skip_special_tokens=True)
        teacher_outputs.append(teacher_text)

    # Add teacher output alongside the original human output.
    augmented = dataset.add_column("teacher_output", teacher_outputs)
    return augmented


def build_distillation_dataset(
    dataset: Dataset,
    teacher_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_config: dict[str, Any],
) -> Dataset:
    """Convenience wrapper to curate and then augment with teacher labels.

    Args:
        dataset: Raw instruction dataset.
        teacher_model: Teacher model.
        tokenizer: Tokenizer.
        data_config: Data configuration dict.

    Returns:
        Curated dataset with teacher outputs.
    """
    curated = dataset
    curation_cfg = data_config.get("curation", {})
    if curation_cfg.get("enabled", False):
        curated = curate_dataset(
            dataset=curated,
            tokenizer=tokenizer,
            response_field=data_config.get("output_field", "output"),
            keep_top_fraction=curation_cfg.get("keep_top_fraction", 0.8),
            max_tokens_per_word=curation_cfg.get("max_tokens_per_word"),
            min_tokens_per_example=curation_cfg.get("min_tokens_per_example", 4),
        )

    distillation_cfg = data_config.get("distillation", {})
    if distillation_cfg.get("enabled", False):
        curated = generate_teacher_labels(
            dataset=curated,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            prompt_template=data_config.get("template", "{instruction}\n{input}\n{output}"),
            instruction_field=data_config.get("instruction_field", "instruction"),
            input_field=data_config.get("input_field", "input"),
            output_field=data_config.get("output_field", "output"),
            max_new_tokens=distillation_cfg.get("max_new_tokens", 256),
            temperature=distillation_cfg.get("temperature", 0.7),
        )

    return curated
