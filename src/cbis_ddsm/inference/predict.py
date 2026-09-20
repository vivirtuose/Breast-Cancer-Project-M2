"""Patch inference: run a trained checkpoint over a partition and score its predictions.

`evaluate_patches` is the schema Task 12 must match exactly when importing nnU-Net
predictions, so both models can be compared on the same CSV columns and summary keys.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import PatchDataset
from ..data.patches import PatchRecord
from ..evaluation.metrics import dice_score, iou_score, threshold_probabilities
from ..evaluation.overlays import make_overlay
from ..training.config import resolve_device
from ..training.trainer import load_checkpoint

METRICS_FIELDS = ("patch_id", "patient_id", "kind", "lesion_pixels", "dice", "iou", "predicted_pixels")


def predict_dataset(
    model: torch.nn.Module, dataset: PatchDataset, device: torch.device | str, batch_size: int = 8,
) -> Iterator[tuple[PatchRecord, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield `(record, image, mask, probability)` for every patch in `dataset`, in dataset order.

    `image`/`mask`/`probability` are HxW arrays with the channel dimension squeezed out: `image`
    is float32 (as loaded by `PatchDataset`, i.e. in [0, 1]), `mask` is boolean, and `probability`
    is float32 in [0, 1]. Runs `model.eval()` under `torch.no_grad()`, batching with a plain
    `DataLoader(..., shuffle=False, num_workers=0)` so the yield order matches `dataset.records`.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    index = 0
    with torch.no_grad():
        for images, masks in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            images_np = images.numpy()
            masks_np = masks.numpy()
            for item in range(images_np.shape[0]):
                record = dataset.records[index]
                image = images_np[item, 0]
                mask = masks_np[item, 0] >= 0.5
                probability = probs[item, 0].astype(np.float32)
                yield record, image, mask, probability
                index += 1


def evaluate_patches(
    run_dir: Path | str,
    patch_root: Path | str,
    partition: str = "val",
    threshold: float = 0.5,
    limit: int | None = None,
    n_figures: int = 8,
    device: str = "auto",
) -> Path:
    """Score `run_dir`'s best checkpoint on `partition`; write a metrics CSV, a summary JSON, and figures.

    Writes `run_dir/predictions/<partition>_metrics.csv` (columns: `patch_id, patient_id, kind,
    lesion_pixels, dice, iou, predicted_pixels`; `dice`/`iou` formatted with `f"{value:.6f}"`),
    `run_dir/predictions/<partition>_summary.json` (keys: `n_patches`, `mean_dice_lesion`,
    `mean_iou_lesion`, `background_false_positive_rate`), and up to `n_figures` overlay PNGs
    (named `<patch_id>.png`) in `run_dir/figures/`, one per lesion patch in dataset order.
    Returns the summary JSON path.
    """
    run_dir = Path(run_dir)
    resolved_device = resolve_device(device)
    model, _ = load_checkpoint(run_dir / "checkpoint_best.pt", device=str(resolved_device))
    dataset = PatchDataset(patch_root, partition=partition, limit=limit)

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    lesion_dice: list[float] = []
    lesion_iou: list[float] = []
    n_background = 0
    n_background_false_positive = 0
    n_figures_written = 0

    for record, image, mask, probability in predict_dataset(model, dataset, resolved_device):
        prediction = threshold_probabilities(probability, threshold)
        dice = dice_score(prediction, mask)
        iou = iou_score(prediction, mask)
        predicted_pixels = int(prediction.sum())

        rows.append({
            "patch_id": record.patch_id,
            "patient_id": record.patient_id,
            "kind": record.kind,
            "lesion_pixels": record.lesion_pixels,
            "dice": f"{dice:.6f}",
            "iou": f"{iou:.6f}",
            "predicted_pixels": predicted_pixels,
        })

        if record.kind == "lesion":
            lesion_dice.append(dice)
            lesion_iou.append(iou)
            if n_figures_written < n_figures:
                title = f"{record.patch_id} dice={dice:.3f}"
                panel = make_overlay(image, mask, prediction, title)
                cv2.imwrite(str(figures_dir / f"{record.patch_id}.png"), panel)
                n_figures_written += 1
        elif record.kind == "background":
            n_background += 1
            if predicted_pixels > 0:
                n_background_false_positive += 1

    metrics_path = predictions_dir / f"{partition}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_patches": len(rows),
        "mean_dice_lesion": float(np.mean(lesion_dice)) if lesion_dice else None,
        "mean_iou_lesion": float(np.mean(lesion_iou)) if lesion_iou else None,
        "background_false_positive_rate": (
            n_background_false_positive / n_background if n_background else None
        ),
    }
    summary_path = predictions_dir / f"{partition}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
