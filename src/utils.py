"""Shared utilities for reproducibility and path handling."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to YAML.

    Args:
        config: Configuration dictionary.
        path: Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_output_dir(config: dict[str, Any]) -> Path:
    """Return the output directory from a training config and create it.

    Args:
        config: Training configuration dictionary.

    Returns:
        Path object for the output directory.
    """
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if not absolute.

    Args:
        path: Relative or absolute path.

    Returns:
        Absolute Path object.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    # Assume project root is two levels up from src/ (src -> project -> cwd)
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable vs total parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
