"""Native-resolution patch extraction and re-projection geometry."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .targets import ImageRecord, fuse_masks


@dataclass(frozen=True)
class PatchWindow:
    """A square window in source-image pixel coordinates; may extend beyond the image."""

    x0: int
    y0: int
    side: int


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    image_key: str
    patient_id: str
    partition: str
    kind: str
    abnormality_id: str
    x0: int
    y0: int
    side: int
    out_size: int
    source_height: int
    source_width: int
    image_file: str
    mask_file: str
    lesion_pixels: int


PATCH_INDEX_FIELDS = (
    "patch_id", "image_key", "patient_id", "partition", "kind", "abnormality_id",
    "x0", "y0", "side", "out_size", "source_height", "source_width",
    "image_file", "mask_file", "lesion_pixels",
)

_INT_FIELDS = ("x0", "y0", "side", "out_size", "source_height", "source_width", "lesion_pixels")


def lesion_window(mask: np.ndarray, size: int, margin: float = 0.15) -> PatchWindow:
    """Square window centered on the lesion bbox, grown to keep the whole lesion plus a margin."""
    ys, xs = np.nonzero(mask >= 128)
    if ys.size == 0:
        raise ValueError("lesion_window requires a non-empty mask")
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    cy = (ymin + ymax + 1) // 2
    cx = (xmin + xmax + 1) // 2
    bbox_side = max(ymax - ymin + 1, xmax - xmin + 1)
    side = max(size, math.ceil(bbox_side * (1 + 2 * margin)))
    return PatchWindow(x0=cx - side // 2, y0=cy - side // 2, side=side)


def crop_window(array: np.ndarray, window: PatchWindow) -> np.ndarray:
    """Zero-padded side x side crop, valid even when the window extends past the array."""
    height, width = array.shape[:2]
    canvas = np.zeros((window.side, window.side), dtype=array.dtype)
    src_x0, src_y0 = max(window.x0, 0), max(window.y0, 0)
    src_x1, src_y1 = min(window.x0 + window.side, width), min(window.y0 + window.side, height)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas
    dst_x0, dst_y0 = src_x0 - window.x0, src_y0 - window.y0
    dst_h, dst_w = src_y1 - src_y0, src_x1 - src_x0
    canvas[dst_y0:dst_y0 + dst_h, dst_x0:dst_x0 + dst_w] = array[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _tissue_mask(image: np.ndarray) -> np.ndarray:
    """Otsu-threshold a x1/8 downsample of `image` and keep only the largest component."""
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        tissue_small = binary
    else:
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # exclude the non-tissue background label
        largest_label = int(np.argmax(counts))
        tissue_small = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cv2.resize(tissue_small, (width, height), interpolation=cv2.INTER_NEAREST)


def background_windows(
    image: np.ndarray, mask: np.ndarray, size: int, count: int, rng: np.random.Generator, max_tries: int = 200,
) -> list[PatchWindow]:
    """Sample up to `count` size x size windows fully inside tissue and free of lesion pixels."""
    height, width = image.shape[:2]
    windows: list[PatchWindow] = []
    if size > height or size > width:
        return windows
    tissue = _tissue_mask(image)
    for _ in range(count):
        for _try in range(max_tries):
            x0 = int(rng.integers(0, width - size + 1))
            y0 = int(rng.integers(0, height - size + 1))
            center_y, center_x = y0 + size // 2, x0 + size // 2
            if tissue[center_y, center_x] == 0:
                continue
            if mask[y0:y0 + size, x0:x0 + size].max() > 0:
                continue
            windows.append(PatchWindow(x0, y0, size))
            break
    return windows


def extract(image: np.ndarray, mask: np.ndarray, window: PatchWindow, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop `window` from `image`/`mask`, resizing to `out_size` only when the side differs."""
    image_patch = crop_window(image, window)
    mask_patch = crop_window(mask, window)
    if window.side != out_size:
        image_patch = cv2.resize(image_patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
        mask_patch = cv2.resize(mask_patch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    mask_patch = np.where(mask_patch > 0, 255, 0).astype(np.uint8)
    return image_patch.astype(np.uint8), mask_patch


def reproject(prediction: np.ndarray, record: PatchRecord) -> np.ndarray:
    """Resize `prediction` back to the source window and paste it into the source image frame."""
    height, width = record.source_height, record.source_width
    canvas = np.zeros((height, width), dtype=np.float32)
    side = record.side
    resized = cv2.resize(prediction.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    x0, y0 = record.x0, record.y0
    src_x0, src_y0 = max(-x0, 0), max(-y0, 0)
    dst_x0, dst_y0 = max(x0, 0), max(y0, 0)
    dst_x1, dst_y1 = min(x0 + side, width), min(y0 + side, height)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return canvas
    patch_h, patch_w = dst_y1 - dst_y0, dst_x1 - dst_x0
    canvas[dst_y0:dst_y0 + patch_h, dst_x0:dst_x0 + patch_w] = resized[src_y0:src_y0 + patch_h, src_x0:src_x0 + patch_w]
    return canvas


def write_patch_index(records: Sequence[PatchRecord], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATCH_INDEX_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: getattr(record, field) for field in PATCH_INDEX_FIELDS}
            writer.writerow(row)


def read_patch_index(path: Path | str) -> tuple[PatchRecord, ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        records = []
        for record in reader:
            values = dict(record)
            for field in _INT_FIELDS:
                values[field] = int(values[field])
            records.append(PatchRecord(**{field: values[field] for field in PATCH_INDEX_FIELDS}))
        return tuple(records)


def _safe_patch_id(patient_id: str, laterality: str, view: str, kind: str, abnormality_id: str, index: int) -> str:
    suffix = abnormality_id if abnormality_id else str(index)
    return f"{patient_id}_{laterality}_{view}_{kind}_{suffix}"


def extract_image_patches(
    record: ImageRecord, *, partition: str, size: int, background_per_image: int,
    rng: np.random.Generator, images_dir: Path, masks_dir: Path, max_tries: int = 200,
) -> tuple[list[PatchRecord], int]:
    """Extract one lesion patch per lesion plus up to `background_per_image` background patches.

    Reads the source image and its lesion masks once, writes PNG patches under `images_dir` /
    `masks_dir`, and returns the produced `PatchRecord`s alongside the count of background
    windows that were requested but not found (a shortfall, not an error).
    """
    image = cv2.imread(str(record.full_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable image file: {record.full_image_path}")
    height, width = image.shape[:2]
    fused_mask = fuse_masks(record.mask_paths)

    results: list[PatchRecord] = []

    for abnormality_id, mask_path in zip(record.abnormality_ids, record.mask_paths):
        lesion_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if lesion_mask is None:
            raise ValueError(f"unreadable mask file: {mask_path}")
        window = lesion_window(lesion_mask, size=size)
        image_patch, mask_patch = extract(image, fused_mask, window, out_size=size)
        patch_id = _safe_patch_id(record.patient_id, record.laterality, record.view, "lesion", abnormality_id, 0)
        image_file = f"images/{patch_id}.png"
        mask_file = f"masks/{patch_id}.png"
        cv2.imwrite(str(images_dir / f"{patch_id}.png"), image_patch)
        cv2.imwrite(str(masks_dir / f"{patch_id}.png"), mask_patch)
        results.append(PatchRecord(
            patch_id=patch_id, image_key=record.image_key, patient_id=record.patient_id, partition=partition,
            kind="lesion", abnormality_id=abnormality_id, x0=window.x0, y0=window.y0, side=window.side,
            out_size=size, source_height=height, source_width=width, image_file=image_file, mask_file=mask_file,
            lesion_pixels=int((mask_patch == 255).sum()),
        ))

    background = background_windows(image, fused_mask, size=size, count=background_per_image, rng=rng, max_tries=max_tries)
    shortfall = background_per_image - len(background)
    for index, window in enumerate(background):
        image_patch, mask_patch = extract(image, fused_mask, window, out_size=size)
        patch_id = _safe_patch_id(record.patient_id, record.laterality, record.view, "background", "", index)
        image_file = f"images/{patch_id}.png"
        mask_file = f"masks/{patch_id}.png"
        cv2.imwrite(str(images_dir / f"{patch_id}.png"), image_patch)
        cv2.imwrite(str(masks_dir / f"{patch_id}.png"), mask_patch)
        results.append(PatchRecord(
            patch_id=patch_id, image_key=record.image_key, patient_id=record.patient_id, partition=partition,
            kind="background", abnormality_id="", x0=window.x0, y0=window.y0, side=window.side,
            out_size=size, source_height=height, source_width=width, image_file=image_file, mask_file=mask_file,
            lesion_pixels=int((mask_patch == 255).sum()),
        ))

    return results, shortfall
