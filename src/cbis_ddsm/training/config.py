"""Training configuration: a frozen dataclass loaded from YAML plus CLI overrides."""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class TrainConfig:
    patch_root: str
    output_root: str = "experiments"
    run_name: str = "unet_smoke"
    architecture: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = 2
    batch_size: int = 8
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    amp: bool = True
    num_workers: int = 2
    seed: int = 0
    device: str = "auto"
    augment: bool = True
    limit_train: int | None = None
    limit_val: int | None = None
    threshold: float = 0.5


_FIELD_NAMES = frozenset(field.name for field in fields(TrainConfig))


def config_to_dict(cfg: TrainConfig) -> dict:
    """Serialize a TrainConfig to a plain dict (for YAML/JSON persistence)."""
    return dataclasses.asdict(cfg)


def load_config(path: Path | str, overrides: Mapping[str, object] | None = None) -> TrainConfig:
    """Load a TrainConfig from a YAML file, then apply `overrides` on top.

    Raises:
        ValueError: If the YAML file or overrides contain a key that is not a TrainConfig field.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    merged = dict(raw)
    if overrides:
        merged.update(overrides)
    unknown = sorted(set(merged) - _FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    return TrainConfig(**merged)


def seed_everything(seed: int) -> None:
    """Seed Python's random, numpy, and torch (CPU + all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    """Resolve a device name; "auto" picks cuda when available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
