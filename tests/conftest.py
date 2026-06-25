"""Pytest fixtures and shared helpers."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_data_config() -> dict:
    """Return a minimal data config for tests."""
    return {
        "dataset_name": "yahma/alpaca-cleaned",
        "dataset_config": None,
        "split": "train",
        "eval_size": 0.1,
        "max_samples": 5,
        "instruction_field": "instruction",
        "input_field": "input",
        "output_field": "output",
        "template": "{instruction}\n{input}\n{output}",
        "cache_dir": None,
        "streaming": False,
    }


@pytest.fixture
def sample_lora_config() -> dict:
    """Return a minimal LoRA config for tests."""
    return {
        "r": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": False,
        "use_rslora": False,
    }


@pytest.fixture
def sample_training_config(tmp_path) -> dict:
    """Return a minimal training config pointing at temp dirs."""
    return {
        "model": {
            "base_model": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
            "revision": "main",
            "max_seq_length": 128,
            "dtype": None,
            "load_in_4bit": True,
            "use_gradient_checkpointing": False,
        },
        "lora": {
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
        },
        "training": {
            "output_dir": str(tmp_path / "outputs"),
            "run_name": "test-run",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "max_steps": 2,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "lr_scheduler_type": "linear",
            "logging_steps": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "seed": 42,
            "bf16": False,
            "fp16": False,
            "optim": "adamw_torch",
            "report_to": "none",
            "remove_unused_columns": False,
            "dataset_text_field": "text",
        },
        "data": {
            "dataset_name": "yahma/alpaca-cleaned",
            "eval_size": 0.0,
            "max_samples": 4,
            "instruction_field": "instruction",
            "input_field": "input",
            "output_field": "output",
            "template": "{instruction}\n{input}\n{output}",
            "cache_dir": str(tmp_path / "cache"),
        },
    }
