"""Training configuration: a frozen dataclass loaded from YAML plus CLI overrides."""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml


SCHEDULERS = ("none", "cosine", "plateau")
# Both selection metrics are maximized; `val_dice_lesion` ignores background patches, whose Dice
# is 1.0 whenever the model correctly predicts nothing, and so matches `mean_dice_lesion` as
# reported by `cbis_ddsm.inference.predict.evaluate_patches`.
SELECTION_METRICS = ("val_dice", "val_dice_lesion")
# Checked for their type at construction: YAML 1.1 reads `3e-4` as a *string* (a float literal
# needs a dot, as in `3.0e-4`), and such a value would otherwise only fail deep inside the
# optimizer, after data preparation has already run.
_REQUIRED_NUMERIC_FIELDS = (
    "epochs", "batch_size", "accumulation_steps", "learning_rate", "weight_decay", "bce_weight",
    "num_workers", "seed", "threshold", "min_learning_rate", "plateau_factor", "plateau_patience",
)
_OPTIONAL_NUMERIC_FIELDS = ("early_stopping_patience", "max_hours", "limit_train", "limit_val")


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
    scheduler: str = "none"
    min_learning_rate: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    early_stopping_patience: int | None = None
    max_hours: float | None = None
    selection_metric: str = "val_dice"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        for name in _REQUIRED_NUMERIC_FIELDS:
            self._require_number(name, getattr(self, name))
        for name in _OPTIONAL_NUMERIC_FIELDS:
            value = getattr(self, name)
            if value is not None:
                self._require_number(name, value)
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.scheduler!r}; expected one of {SCHEDULERS}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"unknown selection_metric {self.selection_metric!r}; expected one of {SELECTION_METRICS}"
            )
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1 when set, got {self.early_stopping_patience}")
        if self.max_hours is not None and self.max_hours < 0:
            raise ValueError(f"max_hours must be >= 0 when set, got {self.max_hours}")

    @staticmethod
    def _require_number(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a number, got {value!r}; YAML reads 3e-4 as a string, "
                "so write 3.0e-4 (with a dot) or 0.0003"
            )


_FIELD_NAMES = frozenset(field.name for field in fields(TrainConfig))


def parse_overrides(pairs: Sequence[str]) -> dict[str, object]:
    """Parse `--set key=value` pairs; each value is decoded with `yaml.safe_load`.

    Args:
        pairs: Sequence of "key=value" strings.

    Returns:
        Dictionary mapping keys to parsed values.

    Raises:
        ValueError: If a pair is malformed (missing '=' or empty key).
    """
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got: {pair}")
        key, _, value = pair.partition("=")
        if not key:
            raise ValueError(f"--set expects key=value, got: {pair}")
        overrides[key] = yaml.safe_load(value)
    return overrides


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
