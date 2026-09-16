"""Reproducible training loop, checkpointing, and the overfit-a-small-batch gate."""

from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import PatchDataset
from ..data.transforms import train_transforms
from ..evaluation.metrics import dice_score, threshold_probabilities
from ..models.factory import build_model
from .config import TrainConfig, config_to_dict, resolve_device, seed_everything
from .losses import BCEDiceLoss


@dataclass
class RunResult:
    run_dir: Path
    best_epoch: int
    best_val_dice: float
    history: list[dict]


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random per DataLoader worker from the worker's torch seed."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _git_commit() -> str | None:
    """Return the current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _make_run_dir(output_root: Path, run_name: str) -> Path:
    """Create `<output_root>/<run_name>_<UTC timestamp>[_2, _3, ...]`."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{run_name}_{timestamp}"
    candidate = output_root / base
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _build_loader(dataset: PatchDataset, cfg: TrainConfig, shuffle: bool, generator: torch.Generator | None) -> DataLoader:
    worker_init = _worker_init_fn if cfg.num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=generator,
        worker_init_fn=worker_init,
        persistent_workers=False,
    )


def _batch_dice(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> list[float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = threshold_probabilities(probs, threshold)
    masks_np = masks.detach().cpu().numpy()
    return [dice_score(pred_item, mask_item) for pred_item, mask_item in zip(preds, masks_np)]


def train(cfg: TrainConfig) -> RunResult:
    """Train a segmentation model and write a self-contained run directory.

    Writes `config.yaml`, `metrics.jsonl` (one JSON line per epoch), `checkpoint_best.pt`
    (weights of the best epoch by validation Dice, or the last epoch when validation is
    empty), and `run.json` (environment + dataset provenance) under a fresh timestamped
    directory inside `cfg.output_root`.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"

    train_dataset = PatchDataset(
        cfg.patch_root, "train", train_transforms() if cfg.augment else None,
        seed=cfg.seed, limit=cfg.limit_train,
    )
    val_dataset = PatchDataset(cfg.patch_root, "val", None, seed=cfg.seed, limit=cfg.limit_val)

    model = build_model(cfg.architecture, cfg.encoder, cfg.encoder_weights, in_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=cfg.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = _build_loader(train_dataset, cfg, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = _build_loader(val_dataset, cfg, shuffle=False, generator=None) if len(val_dataset) > 0 else None

    run_dir = _make_run_dir(Path(cfg.output_root), cfg.run_name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"

    history: list[dict] = []
    best_epoch = 0
    best_val_dice = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        start = time.monotonic()
        running_loss, n_batches = 0.0, 0
        n_train_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)
            running_loss += loss.item()
            n_batches += 1

            scaled_loss = loss / cfg.accumulation_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_accumulation_boundary = (batch_index + 1) % cfg.accumulation_steps == 0
            is_last_batch = batch_index == n_train_batches - 1
            if is_accumulation_boundary or is_last_batch:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        train_loss = running_loss / max(n_batches, 1)

        val_loss: float | None = None
        val_dice: float | None = None
        if val_loader is not None:
            model.eval()
            val_running_loss, val_batches = 0.0, 0
            dice_values: list[float] = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images_dev, masks_dev = images.to(device), masks.to(device)
                    logits = model(images_dev)
                    val_running_loss += criterion(logits, masks_dev).item()
                    val_batches += 1
                    dice_values.extend(_batch_dice(logits, masks_dev, cfg.threshold))
            val_loss = val_running_loss / max(val_batches, 1)
            val_dice = float(np.mean(dice_values)) if dice_values else 0.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - start,
        }
        history.append(epoch_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        # With no validation set, every epoch overwrites the checkpoint, so the last one wins.
        is_best = val_dice is None or epoch == 0 or val_dice > best_val_dice
        if is_best:
            best_epoch = epoch
            best_val_dice = val_dice if val_dice is not None else 0.0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    torch.save(
        {"model": best_state, "config": config_to_dict(cfg), "epoch": best_epoch, "val_dice": best_val_dice},
        run_dir / "checkpoint_best.pt",
    )
    run_info = {
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "seed": cfg.seed,
        "patch_root": cfg.patch_root,
        "n_train_patches": len(train_dataset),
        "n_val_patches": len(val_dataset),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    return RunResult(run_dir=run_dir, best_epoch=best_epoch, best_val_dice=best_val_dice, history=history)


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild the model architecture from a checkpoint's saved config and load its weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    model = build_model(
        saved_config["architecture"], saved_config["encoder"], encoder_weights=None, in_channels=1,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def overfit_batch(cfg: TrainConfig, n: int = 8, steps: int = 300) -> float:
    """Fit a fixed, unaugmented batch of `n` train patches for `steps` steps; return mean Dice.

    A model that cannot drive Dice up on a handful of memorized patches signals a pipeline bug
    (e.g. misaligned image/mask pairs) before spending time on real training.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dataset = PatchDataset(cfg.patch_root, "train", None, seed=cfg.seed, limit=n)

    model = build_model(cfg.architecture, cfg.encoder, cfg.encoder_weights, in_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=cfg.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    try:
        fixed_batches = [next(iter(DataLoader(dataset, batch_size=n, shuffle=False)))]
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fixed_batches = list(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False))
    fixed_batches = [(images.to(device), masks.to(device)) for images, masks in fixed_batches]

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for images, masks in fixed_batches:
            loss = criterion(model(images), masks) / len(fixed_batches)
            loss.backward()
        optimizer.step()

    model.eval()
    dice_values: list[float] = []
    with torch.no_grad():
        for images, masks in fixed_batches:
            dice_values.extend(_batch_dice(model(images), masks, cfg.threshold))
    return float(np.mean(dice_values)) if dice_values else 0.0
