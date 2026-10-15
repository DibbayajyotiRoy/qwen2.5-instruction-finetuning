"""Tests for dataset formatting and loading."""

from __future__ import annotations

from typing import Any

import pytest

from src.data import build_dataset, format_alpaca_example, format_dataset, load_instruction_dataset


class DummyTokenizer:
    """Minimal tokenizer stand-in for tests."""

    eos_token = "</s>"


def test_format_alpaca_example() -> None:
    """Test instruction template formatting."""
    example = {"instruction": "Hello", "input": "World", "output": "Hi"}
    template = "I:{instruction}\nIN:{input}\nO:{output}"
    result = format_alpaca_example(example, template, eos_token="<eos>")
    assert result["text"] == "I:Hello\nIN:World\nO:Hi<eos>"


def test_format_alpaca_example_missing_fields() -> None:
    """Formatting should coerce missing fields to empty strings."""
    example = {"instruction": "Say hi"}
    result = format_alpaca_example(example, "{instruction} {input} -> {output}")
    assert result["text"] == "Say hi  -> "


def test_load_instruction_dataset_streaming() -> None:
    """Test that streaming dataset loading returns an iterable dataset."""
    ds = load_instruction_dataset(
        dataset_name="yahma/alpaca-cleaned",
        config_name=None,
        split="train",
        cache_dir=None,
        streaming=True,
        max_samples=None,
    )
    sample = next(iter(ds))
    assert "instruction" in sample


@pytest.mark.slow
@pytest.mark.requires_model
def test_build_dataset_splits(sample_data_config: Any) -> None:
    """Test that build_dataset produces train and validation splits."""
    sample_data_config["max_samples"] = 20
    tokenizer = DummyTokenizer()
    dataset = build_dataset(sample_data_config, tokenizer=tokenizer, eval_size=0.2, seed=42)
    assert "train" in dataset
    assert "validation" in dataset
    assert len(dataset["validation"]) > 0
    assert "text" in dataset["train"].column_names


def test_format_dataset_preserves_text_column() -> None:
    """format_dataset should convert raw columns into a single text column."""
    from datasets import Dataset

    data = {
        "instruction": ["q1", "q2"],
        "input": ["", "ctx"],
        "output": ["a", "b"],
    }
    config = {
        "instruction_field": "instruction",
        "input_field": "input",
        "output_field": "output",
        "template": "I:{instruction}\nIN:{input}\nO:{output}",
    }
    raw = Dataset.from_dict(data)
    formatted = format_dataset(raw, config, tokenizer=DummyTokenizer())
    assert "text" in formatted.column_names
    assert "instruction" not in formatted.column_names
    assert formatted[0]["text"] == "I:q1\nIN:\nO:a</s>"
