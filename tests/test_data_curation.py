"""Tests for data curation and distillation utilities."""

from __future__ import annotations

from typing import Any

import torch

from src.data_curation import (
    build_distillation_dataset,
    concision_score,
    curate_dataset,
    deduplicate_exact,
    deduplicate_near,
    score_dataset_quality,
)


class DummyEncoding(dict):
    """Tiny batch encoding with a .to() method like Transformers BatchEncoding."""

    def to(self, device: Any) -> DummyEncoding:
        return DummyEncoding({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()})


class DummyTokenizer:
    """Minimal tokenizer stand-in for curation tests."""

    eos_token = "</s>"
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Approximate BPE: one token per word plus an inter-word overhead so
        # longer sentences get a higher tokens-per-word ratio.
        words = text.split()
        tokens: list[int] = []
        for i, _word in enumerate(words):
            tokens.append(i)
            if i > 0:
                tokens.append(i + len(words))
        return tokens

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs: Any) -> DummyEncoding:
        input_ids = self.encode(text)
        if return_tensors == "pt":
            return DummyEncoding({"input_ids": torch.tensor([input_ids])})
        return DummyEncoding({"input_ids": [input_ids]})


class DummyModelOutput:
    """Fake model forward output carrying a scalar loss."""

    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class DummyModel:
    """Tiny language-model stand-in that returns a fixed perplexity."""

    def __init__(self, perplexity: float = 10.0) -> None:
        self.device = torch.device("cpu")
        self._perplexity = perplexity

    def eval(self) -> DummyModel:
        return self

    def __call__(self, **kwargs: Any) -> DummyModelOutput:
        return DummyModelOutput(torch.log(torch.tensor(self._perplexity)))


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


def test_deduplicate_exact_removes_duplicates() -> None:
    """Exact dedup should keep the first occurrence and drop later copies."""
    from datasets import Dataset

    data = {
        "instruction": ["q1", "q2", "q1"],
        "input": ["", "ctx", ""],
        "output": ["a", "b", "a"],
    }
    dataset = Dataset.from_dict(data)
    deduped = deduplicate_exact(dataset)
    assert len(deduped) == 2
    assert list(deduped["instruction"]) == ["q1", "q2"]


def test_deduplicate_exact_ignores_formatting_case() -> None:
    """Exact dedup should treat case/whitespace-normalised copies as duplicates."""
    from datasets import Dataset

    data = {
        "instruction": ["Q1", "q1"],
        "input": ["  some context  ", "some context"],
        "output": ["A", "a"],
    }
    dataset = Dataset.from_dict(data)
    deduped = deduplicate_exact(dataset)
    assert len(deduped) == 1


def test_deduplicate_near_removes_near_duplicates() -> None:
    """Near-dedup should drop examples that differ only by a few words."""
    from datasets import Dataset

    base = " ".join(["The quick brown fox jumps over the lazy dog"] * 20)
    variant = base + " indeed"
    different = " ".join(["Astronomy is the scientific study of celestial objects and phenomena"] * 20)

    data = {
        "instruction": ["q1", "q1", "q2"],
        "input": ["", "", ""],
        "output": [base, variant, different],
    }
    dataset = Dataset.from_dict(data)
    deduped = deduplicate_near(dataset, threshold=0.8, num_perm=128)
    assert len(deduped) == 2
    assert different in list(deduped["output"])


def test_curate_dataset_applies_deduplication() -> None:
    """curate_dataset should apply deduplication when configured."""
    from datasets import Dataset

    tokenizer = DummyTokenizer()
    data = {
        "instruction": ["q1", "q1", "q2"],
        "input": ["", "", ""],
        "output": ["a", "a", "b"],
    }
    dataset = Dataset.from_dict(data)
    curated = curate_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        response_field="output",
        keep_top_fraction=1.0,
        min_tokens_per_example=1,
        deduplication={"enabled": True, "exact": True, "near": False},
    )
    assert len(curated) == 2


def test_build_distillation_dataset_curates_raw_dataset() -> None:
    """build_distillation_dataset should work on raw, unformatted columns."""
    from datasets import Dataset

    tokenizer = DummyTokenizer()
    data = {
        "instruction": ["q1", "q1", "q2", "q3"],
        "input": ["", "", "", ""],
        "output": ["a", "a", "b", "c c c c c c c c c c"],
    }
    dataset = Dataset.from_dict(data)
    data_config = {
        "instruction_field": "instruction",
        "input_field": "input",
        "output_field": "output",
        "template": "{instruction}\n{input}\n{output}",
        "curation": {
            "enabled": True,
            "keep_top_fraction": 1.0,
            "max_tokens_per_word": 1.5,
            "min_tokens_per_example": 1,
            "deduplication": {"enabled": True, "exact": True, "near": False},
        },
        "distillation": {"enabled": False},
    }
    curated = build_distillation_dataset(
        dataset=dataset,
        teacher_model=None,
        tokenizer=tokenizer,
        data_config=data_config,
    )
    # One exact duplicate removed, verbose example removed.
    assert len(curated) == 2
    assert "instruction" in curated.column_names


def test_score_dataset_quality_adds_perplexity() -> None:
    """score_dataset_quality should add a quality_score column with expected perplexity."""
    from datasets import Dataset

    data = {
        "instruction": ["q1", "q2"],
        "input": ["", ""],
        "output": ["a", "b c d"],
    }
    dataset = Dataset.from_dict(data)
    tokenizer = DummyTokenizer()
    model = DummyModel(perplexity=7.0)

    scored = score_dataset_quality(dataset, model, tokenizer, response_field="output")
    assert "quality_score" in scored.column_names
    assert all(abs(score - 7.0) < 1e-6 for score in scored["quality_score"])


def test_curate_dataset_filters_by_max_perplexity() -> None:
    """curate_dataset should drop examples above max_perplexity when quality_score exists."""
    from datasets import Dataset

    data = {
        "instruction": ["q1", "q2", "q3"],
        "input": ["", "", ""],
        "output": ["a", "b", "c"],
        "quality_score": [5.0, 15.0, 10.0],
    }
    dataset = Dataset.from_dict(data)
    tokenizer = DummyTokenizer()

    curated = curate_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        response_field="output",
        keep_top_fraction=1.0,
        min_tokens_per_example=1,
        max_perplexity=10.0,
    )
    assert len(curated) == 2
    assert 15.0 not in list(curated["quality_score"])
