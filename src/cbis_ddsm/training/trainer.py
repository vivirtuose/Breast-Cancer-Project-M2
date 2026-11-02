"""Reproducible training loop, checkpointing, and the overfit-a-small-batch gate."""

from __future__ import annotations

import json
import os
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
from ..data.patches import PatchRecord
from ..data.transforms import train_transforms
from ..evaluation.metrics import dice_score, threshold_probabilities
from ..models.factory import build_model
from .config import TrainConfig, config_to_dict, resolve_device, seed_everything
from .losses import BCEDiceLoss


BEST_CHECKPOINT_NAME = "checkpoint_best.pt"
LAST_CHECKPOINT_NAME = "checkpoint_last.pt"
# A resumed run must continue the same experiment, so these fields may not change across a resume.
RESUME_INVARIANT_FIELDS = ("patch_root", "architecture", "encoder", "batch_size", "accumulation_steps")


@dataclass
class RunResult:
    run_dir: Path
    best_epoch: int
    best_val_dice: float
    history: list[dict]
    completed_epochs: int = 0
    stopped_early: bool = False
    stop_reason: str | None = None


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


def stop_reason(
    *, epochs_since_best: int, patience: int | None, elapsed_seconds: float, max_hours: float | None
) -> str | None:
    """Why training should stop after the epoch just finished, or None to keep going.

    An exhausted time budget outranks early stopping: on a session-limited runtime (Kaggle) the
    budget is the reason the run must end now, whatever the validation curve is doing.
    """
    if max_hours is not None and elapsed_seconds >= max_hours * 3600:
        return "time_budget"
    if patience is not None and epochs_since_best >= patience:
        return "early_stopping"
    return None


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """Build the per-epoch LR scheduler named by `cfg.scheduler`, or None for "none"."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_learning_rate
        )
    if cfg.scheduler == "plateau":
        # The monitored metric is a Dice score, so the plateau scheduler maximizes it.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.plateau_factor, patience=cfg.plateau_patience,
            min_lr=cfg.min_learning_rate,
        )
    return None


def _save_atomic(payload: dict, path: Path) -> None:
    """Write a checkpoint through a temporary file, so a killed process cannot truncate it.

    A Kaggle session can be stopped at any moment; `torch.save` straight onto the destination
    would leave an unreadable file and make the run unresumable.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_best_checkpoint(path: Path, model: torch.nn.Module, cfg: TrainConfig, epoch: int, score: float) -> None:
    """Write the weights-only checkpoint that inference reloads (`load_checkpoint`, `predict.py`)."""
    _save_atomic(
        {
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "config": config_to_dict(cfg),
            "epoch": epoch,
            "val_dice": score,
        },
        path,
    )


def _save_last_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    scheduler, cfg: TrainConfig, epoch: int, best_epoch: int, best_score: float | None,
) -> None:
    """Write everything needed to resume after the epoch that just finished."""
    _save_atomic(
        {
            "model": model.state_dict(),
            "config": config_to_dict(cfg),
            "epoch": epoch,
            "val_dice": best_score if best_score is not None else 0.0,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        path,
    )


def _read_history(metrics_path: Path, before_epoch: int) -> list[dict]:
    """Re-read the epoch records of a resumed run, keeping those strictly before `before_epoch`."""
    if not metrics_path.is_file():
        return []
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [record for record in records if record["epoch"] < before_epoch]


def _load_resume_state(cfg: TrainConfig) -> tuple[Path, dict]:
    """Validate `cfg.resume_from` and return its run directory plus the last checkpoint payload.

    Raises:
        ValueError: If the directory or its last checkpoint is missing, or if the current config
            changes a field that must stay fixed across a resume.
    """
    run_dir = Path(cfg.resume_from)
    checkpoint_path = run_dir / LAST_CHECKPOINT_NAME
    if not run_dir.is_dir():
        raise ValueError(f"resume_from is not a run directory: {run_dir}")
    if not checkpoint_path.is_file():
        raise ValueError(f"resume_from has no {LAST_CHECKPOINT_NAME}: {run_dir}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = state["config"]
    current = config_to_dict(cfg)
    mismatched = [
        f"{field}: run has {saved.get(field)!r}, config has {current[field]!r}"
        for field in RESUME_INVARIANT_FIELDS
        if saved.get(field) != current[field]
    ]
    if mismatched:
        raise ValueError("cannot resume with a changed configuration — " + "; ".join(mismatched))
    return run_dir, state


def _validate_epoch(
    model: torch.nn.Module, val_loader: DataLoader, val_records: tuple[PatchRecord, ...],
    criterion: torch.nn.Module, device: torch.device, threshold: float,
) -> tuple[float, float, float | None]:
    """Score one validation pass: `(loss, dice over every patch, dice over lesion patches only)`.

    The lesion-only Dice is what `evaluate_patches` reports as `mean_dice_lesion`; the all-patch
    Dice includes background patches, whose Dice is 1.0 whenever the model predicts nothing.
    `None` is returned for the lesion-only Dice when the partition holds no lesion patch.
    """
    model.eval()
    running_loss, n_batches = 0.0, 0
    all_dice: list[float] = []
    lesion_dice: list[float] = []
    index = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images_dev, masks_dev = images.to(device), masks.to(device)
            logits = model(images_dev)
            running_loss += criterion(logits, masks_dev).item()
            n_batches += 1
            for value in _batch_dice(logits, masks_dev, threshold):
                all_dice.append(value)
                if val_records[index].kind == "lesion":
                    lesion_dice.append(value)
                index += 1
    return (
        running_loss / max(n_batches, 1),
        float(np.mean(all_dice)) if all_dice else 0.0,
        float(np.mean(lesion_dice)) if lesion_dice else None,
    )


def train(cfg: TrainConfig) -> RunResult:
    """Train a segmentation model and write a self-contained run directory.

    Writes `config.yaml`, `metrics.jsonl` (one JSON line per epoch), `checkpoint_best.pt` (the
    weights of the best epoch by `cfg.selection_metric`, or of the last epoch when validation is
    empty), `checkpoint_last.pt` (those weights plus optimizer, scaler and scheduler state, for
    `resume_from`) and `run.json` (environment + dataset provenance) under a fresh timestamped
    directory inside `cfg.output_root`. Both checkpoints and `run.json` are rewritten after every
    epoch, so a run killed by a session limit keeps everything it had reached.

    With `cfg.resume_from` set to an existing run directory, training continues in that directory
    from the epoch after its `checkpoint_last.pt` and up to `cfg.epochs`, appending to its
    `metrics.jsonl`; the fields in `RESUME_INVARIANT_FIELDS` must be unchanged.

    Raises:
        ValueError: If `cfg.resume_from` cannot be resumed, or if `cfg.selection_metric` is
            `"val_dice_lesion"` while the validation partition holds no lesion patch.
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
    scheduler = _build_scheduler(optimizer, cfg)

    # Reseeded from (seed, epoch) at the top of every epoch rather than once for the whole run, so
    # the batch order of epoch N is a function of N alone. A resumed run then shuffles exactly as
    # the uninterrupted run would have, which a generator advanced in place could not guarantee.
    train_generator = torch.Generator()
    train_loader = _build_loader(train_dataset, cfg, shuffle=True, generator=train_generator)
    val_loader = _build_loader(val_dataset, cfg, shuffle=False, generator=None) if len(val_dataset) > 0 else None
    if (
        cfg.selection_metric == "val_dice_lesion"
        and val_loader is not None
        and not any(record.kind == "lesion" for record in val_dataset.records)
    ):
        raise ValueError(
            "selection_metric='val_dice_lesion' needs at least one lesion patch in the validation "
            f"partition of {cfg.patch_root}"
        )

    start_epoch = 0
    best_epoch = 0
    best_score: float | None = None
    resumed_from: str | None = None
    if cfg.resume_from:
        run_dir, state = _load_resume_state(cfg)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
            if cfg.scheduler == "cosine":
                # `CosineAnnealingLR.state_dict()` carries `T_max`, so a resume that extends
                # `epochs` would otherwise keep the interrupted run's horizon and send the rate
                # back up to its starting value. The config owns the horizon.
                scheduler.T_max = max(cfg.epochs, 1)
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_score = state["best_score"]
        resumed_from = str(run_dir)
    else:
        run_dir = _make_run_dir(Path(cfg.output_root), cfg.run_name)

    config_name = "config.yaml" if start_epoch == 0 else f"config_resume_{start_epoch}.yaml"
    (run_dir / config_name).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"

    history: list[dict] = _read_history(metrics_path, start_epoch)
    completed_epochs = start_epoch
    reason: str | None = None
    started = time.monotonic()

    for epoch in range(start_epoch, cfg.epochs):
        train_dataset.set_epoch(epoch)
        train_generator.manual_seed(cfg.seed + epoch)
        model.train()
        start = time.monotonic()
        epoch_lr = optimizer.param_groups[0]["lr"]
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
        val_dice_lesion: float | None = None
        if val_loader is not None:
            val_loss, val_dice, val_dice_lesion = _validate_epoch(
                model, val_loader, val_dataset.records, criterion, device, cfg.threshold
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_dice_lesion": val_dice_lesion,
            "lr": epoch_lr,
            "seconds": time.monotonic() - start,
        }
        history.append(epoch_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
        completed_epochs = epoch + 1

        score = epoch_record[cfg.selection_metric] if val_loader is not None else None
        # With no validation set, every epoch overwrites the checkpoint, so the last one wins.
        is_best = score is None or best_score is None or score > best_score
        if is_best:
            best_epoch = epoch
            best_score = score
            _save_best_checkpoint(
                run_dir / BEST_CHECKPOINT_NAME, model, cfg, epoch, score if score is not None else 0.0
            )
        # The scheduler steps before the checkpoint is written, so the state saved for a resume is
        # the one the *next* epoch must start from; saving first would replay this step and
        # stretch the schedule by one epoch on every resume.
        if scheduler is not None:
            if cfg.scheduler == "plateau":
                if score is not None:
                    scheduler.step(score)
            else:
                scheduler.step()

        _save_last_checkpoint(
            run_dir / LAST_CHECKPOINT_NAME, model, optimizer, scaler, scheduler, cfg,
            epoch, best_epoch, best_score,
        )

        reason = stop_reason(
            epochs_since_best=epoch - best_epoch,
            patience=cfg.early_stopping_patience,
            elapsed_seconds=time.monotonic() - started,
            max_hours=cfg.max_hours,
        )
        _write_run_info(run_dir, cfg, device, train_dataset, val_dataset, completed_epochs, reason, resumed_from)
        if reason is not None:
            break

    if not (run_dir / BEST_CHECKPOINT_NAME).is_file():
        # Nothing ran (epochs=0, or a resume that was already finished before any checkpoint):
        # keep the run directory self-contained so predict.py still has weights to load.
        _save_best_checkpoint(run_dir / BEST_CHECKPOINT_NAME, model, cfg, best_epoch, best_score or 0.0)
    _write_run_info(run_dir, cfg, device, train_dataset, val_dataset, completed_epochs, reason, resumed_from)

    return RunResult(
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_val_dice=best_score if best_score is not None else 0.0,
        history=history,
        completed_epochs=completed_epochs,
        stopped_early=reason is not None,
        stop_reason=reason,
    )


def _write_run_info(
    run_dir: Path, cfg: TrainConfig, device: torch.device, train_dataset: PatchDataset,
    val_dataset: PatchDataset, completed_epochs: int, reason: str | None, resumed_from: str | None,
) -> None:
    """Rewrite `run.json` so a run killed mid-training still carries its provenance."""
    run_info = {
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "seed": cfg.seed,
        "patch_root": cfg.patch_root,
        "n_train_patches": len(train_dataset),
        "n_val_patches": len(val_dataset),
        "n_val_lesion_patches": sum(1 for record in val_dataset.records if record.kind == "lesion"),
        "selection_metric": cfg.selection_metric,
        "scheduler": cfg.scheduler,
        "requested_epochs": cfg.epochs,
        "completed_epochs": completed_epochs,
        "stop_reason": reason,
        "resumed_from": resumed_from,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")


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
