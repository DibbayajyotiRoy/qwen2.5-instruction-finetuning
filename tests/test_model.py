"""Tests for model loading and LoRA setup."""

from __future__ import annotations

import torch

from src.model import build_lora_model
from src.utils import count_trainable_parameters


class DummyModel(torch.nn.Module):
    """Tiny transformer-like model for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)
        self.is_loaded_in_4bit = False


def test_count_trainable_parameters() -> None:
    """Test parameter counting utility."""
    model = DummyModel()
    trainable, total = count_trainable_parameters(model)
    assert trainable == total
    assert total > 0

    for p in model.q_proj.parameters():
        p.requires_grad = False
    trainable, total = count_trainable_parameters(model)
    assert trainable < total


def test_build_lora_model(sample_lora_config) -> None:
    """Test LoRA adapter application on a dummy model."""
    model = DummyModel()
    lora_model = build_lora_model(model, sample_lora_config)
    assert "lora" in str(type(lora_model)).lower()


def test_build_lora_model_4bit_prepare(sample_lora_config) -> None:
    """Test 4-bit preparation path does not crash."""
    model = DummyModel()
    model.is_loaded_in_4bit = True
    # prepare_model_for_kbit_training expects HF model attrs, so we mock them.
    model.config = type("Config", (), {"use_cache": True})()
    model.gradient_checkpointing_enable = lambda **kwargs: None
    model.enable_input_require_grads = lambda: None
    lora_model = build_lora_model(model, sample_lora_config)
    assert lora_model is not None
