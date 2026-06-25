"""Tests for training orchestration."""

from __future__ import annotations

import pytest

from src.train import build_training_arguments
from src.utils import get_output_dir, load_config, save_config


def test_load_and_save_config(tmp_path) -> None:
    """Config round-trip should preserve values."""
    config = {"a": 1, "b": {"c": [1, 2, 3]}}
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded == config


def test_get_output_dir(sample_training_config) -> None:
    """Output directory helper should create the directory."""
    out = get_output_dir(sample_training_config)
    assert out.exists()
    assert "outputs" in str(out)


def test_build_training_arguments(sample_training_config) -> None:
    """TrainingArguments should reflect config values."""
    args = build_training_arguments(sample_training_config)
    assert args.output_dir.endswith("checkpoints")
    assert args.per_device_train_batch_size == 1
    assert args.num_train_epochs == 1
    assert args.seed == 42


@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.gpu
def test_train_smoke(sample_training_config) -> None:
    """Run a two-step training smoke test with Unsloth."""
    from src.train import train

    trainer = train(sample_training_config)
    assert trainer is not None
    assert (get_output_dir(sample_training_config) / "final_checkpoint").exists()
