"""Tests for data curation and distillation utilities."""

from __future__ import annotations

import pytest

from src.data_curation import concision_score, curate_dataset


class DummyTokenizer:
    """Minimal tokenizer stand-in for curation tests."""

    eos_token = "</s>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def test_concision_score() -> None:
    """Concision score should be lower for shorter responses."""
    tokenizer = DummyTokenizer()
    verbose = concision_score("This is a very verbose and unnecessarily long response", tokenizer)
    concise = concision_score("Short answer.", tokenizer)
    assert concise < verbose


def test_curate_dataset_filters_verbose() -> None:
    """Curation should drop the most verbose examples."""
    from datasets import Dataset

    tokenizer = DummyTokenizer()
    data = {
        "instruction": ["q1", "q2", "q3", "q4"],
        "input": ["", "", "", ""],
        "output": [
            "a",  # 1 token / 1 word
            "a b",  # 2 / 2 = 1.0
            "a b c d e",  # 5 / 5 = 1.0
            "a a a a a a a a a a",  # 10 / 10 = 1.0  (all words same, tokenizer still counts)
        ],
    }
    dataset = Dataset.from_dict(data)
    curated = curate_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        response_field="output",
        keep_top_fraction=0.5,
        max_tokens_per_word=1.1,
        min_tokens_per_example=1,
    )
    assert len(curated) <= 2
