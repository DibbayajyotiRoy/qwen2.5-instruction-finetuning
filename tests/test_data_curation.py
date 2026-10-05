"""Tests for data curation and distillation utilities."""

from __future__ import annotations

from src.data_curation import concision_score, curate_dataset, deduplicate_exact, deduplicate_near


class DummyTokenizer:
    """Minimal tokenizer stand-in for curation tests."""

    eos_token = "</s>"

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
