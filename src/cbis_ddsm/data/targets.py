"""Per-mammogram-image targets: lesion grouping and pixel-wise mask fusion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .manifest import ManifestRow


@dataclass(frozen=True)
class ImageRecord:
    image_key: str
    patient_id: str
    laterality: str
    view: str
    split: str
    full_image_path: Path
    mask_paths: tuple[Path, ...]
    abnormality_ids: tuple[str, ...]
    pathologies: tuple[str, ...]
    density: str
    subtlety: tuple[str, ...]

    @property
    def has_malignant(self) -> bool:
        return any(pathology == "MALIGNANT" for pathology in self.pathologies)


def _identity(row: ManifestRow) -> tuple[str, str, str, str, str]:
    return (row.patient_id, row.laterality, row.view, row.abnormality_id, row.lesion_type)


def _image_key(row: ManifestRow) -> str:
    return f"{row.patient_id}|{row.laterality}|{row.view}|{row.split}"


def _abnormality_sort_key(abnormality_id: str) -> tuple[int, int, str]:
    # Numeric ids sort numerically and before any non-numeric id, which sorts lexicographically.
    if abnormality_id.isdigit():
        return (0, int(abnormality_id), abnormality_id)
    return (1, 0, abnormality_id)


def select_masses(rows: Iterable[ManifestRow]) -> tuple[ManifestRow, ...]:
    """Keep only mass lesion rows; pathology is metadata, never a filter here."""
    return tuple(row for row in rows if row.lesion_type == "mass")


def read_excluded_identities(path: Path | str) -> frozenset[tuple[str, str, str, str, str]]:
    """Parse the `lesion_identity` column of an exclusions CSV written by `write_exclusions_csv`."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return frozenset(tuple(record["lesion_identity"].split("|")) for record in reader)


def group_images(
    rows: Iterable[ManifestRow], excluded: frozenset[tuple[str, str, str, str, str]] = frozenset()
) -> tuple[tuple[ImageRecord, ...], tuple[str, ...]]:
    """Merge lesion rows sharing one mammogram image, dropping images with any excluded lesion."""
    groups: dict[str, list[ManifestRow]] = {}
    for row in rows:
        groups.setdefault(_image_key(row), []).append(row)
    records: list[ImageRecord] = []
    dropped: list[str] = []
    for key in sorted(groups):
        lesions = groups[key]
        if any(_identity(row) in excluded for row in lesions):
            dropped.append(key)
            continue
        ordered = sorted(lesions, key=lambda row: _abnormality_sort_key(row.abnormality_id))
        first = ordered[0]
        records.append(ImageRecord(
            image_key=key, patient_id=first.patient_id, laterality=first.laterality, view=first.view, split=first.split,
            full_image_path=first.full_image_path, mask_paths=tuple(row.mask_image_path for row in ordered),
            abnormality_ids=tuple(row.abnormality_id for row in ordered), pathologies=tuple(row.pathology for row in ordered),
            density=first.density, subtlety=tuple(row.subtlety for row in ordered),
        ))
    return tuple(records), tuple(dropped)


def fuse_masks(paths: Sequence[Path]) -> np.ndarray:
    """Binarize (pixel >= 128) and pixel-wise union every lesion mask of one image."""
    if not paths:
        raise ValueError("fuse_masks requires at least one mask path")
    binarized: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for path in paths:
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError(f"unreadable mask file: {path}")
        if shape is None:
            shape = pixels.shape
        elif pixels.shape != shape:
            raise ValueError(f"mask shape mismatch: {path} has shape {pixels.shape}, expected {shape}")
        binarized.append((pixels >= 128).astype(np.uint8))
    return np.maximum.reduce(binarized)
