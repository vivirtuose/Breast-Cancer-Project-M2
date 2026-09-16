from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from cbis_ddsm.data.audit import audit_manifest
from cbis_ddsm.data.manifest import ManifestRow


def _write_jpeg(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)


def _row(tmp_path: Path, *, abnormality_id: str = "1") -> ManifestRow:
    image = tmp_path / f"image-{abnormality_id}.jpg"
    crop = tmp_path / f"crop-{abnormality_id}.jpg"
    mask = tmp_path / f"mask-{abnormality_id}.jpg"
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    binary_mask = np.zeros((8, 8), dtype=np.uint8)
    binary_mask[2:4, 2:4] = 255
    _write_jpeg(image, pixels)
    _write_jpeg(crop, pixels)
    _write_jpeg(mask, binary_mask)
    return ManifestRow(
        patient_id="P_001",
        laterality="LEFT",
        view="CC",
        abnormality_id=abnormality_id,
        lesion_type="mass",
        pathology="BENIGN",
        density="3",
        assessment="4",
        split="train",
        source_csv="mass_case_description_train_set.csv",
        full_image_path=image,
        crop_image_path=crop,
        mask_image_path=mask,
    )


def test_audit_emits_one_highest_precedence_reason_per_row(tmp_path: Path) -> None:
    """A missing image must outrank duplicate, empty, and dimension findings on one lesion."""
    row = _row(tmp_path)
    row.full_image_path.unlink()
    _write_jpeg(row.mask_image_path, np.zeros((7, 7), dtype=np.uint8))

    result = audit_manifest([row, row])

    assert [(issue.row_number, issue.reason_code) for issue in result.exclusions] == [
        (0, "missing_path"),
        (1, "missing_path"),
    ]


def test_audit_reports_duplicate_after_paths_are_valid(tmp_path: Path) -> None:
    """Removing only the duplicate identity must leave no exclusion for valid rows."""
    first = _row(tmp_path, abnormality_id="1")
    duplicate = _row(tmp_path, abnormality_id="1")
    distinct = _row(tmp_path, abnormality_id="2")

    result = audit_manifest([first, duplicate, distinct])

    assert [(issue.row_number, issue.reason_code) for issue in result.exclusions] == [(0, "duplicate_lesion"), (1, "duplicate_lesion")]
    assert result.included_rows == (distinct,)


def test_audit_detects_dimension_empty_and_nonbinary_masks(tmp_path: Path) -> None:
    """Each corrupted mask property must produce its documented exclusion code."""
    dimension = _row(tmp_path, abnormality_id="dimension")
    empty = _row(tmp_path, abnormality_id="empty")
    nonbinary = _row(tmp_path, abnormality_id="nonbinary")
    _write_jpeg(dimension.mask_image_path, np.zeros((7, 8), dtype=np.uint8))
    _write_jpeg(empty.mask_image_path, np.zeros((8, 8), dtype=np.uint8))
    nonbinary_pixels = np.zeros((8, 8), dtype=np.uint8)
    nonbinary_pixels[2:4, 2:4] = 127
    _write_jpeg(nonbinary.mask_image_path, nonbinary_pixels)

    result = audit_manifest([dimension, empty, nonbinary])

    assert [(issue.row_number, issue.reason_code) for issue in result.exclusions] == [
        (0, "dimension_mismatch"),
        (1, "empty_mask"),
        (2, "non_binary_mask"),
    ]


def test_audit_counts_each_identity_a_linear_number_of_times(tmp_path: Path, monkeypatch) -> None:
    """Comparing every identity with every other identity would make larger audits quadratic."""
    seed = _row(tmp_path)
    unique_rows = tuple(replace(seed, abnormality_id=f"unique-{number}") for number in range(80))
    duplicate_rows = (replace(seed, abnormality_id="duplicate"), replace(seed, abnormality_id="duplicate"))
    rows = unique_rows + duplicate_rows
    original_property = ManifestRow.lesion_identity
    accesses = 0

    def counted_identity(row: ManifestRow) -> tuple[str, str, str, str, str]:
        nonlocal accesses
        accesses += 1
        return original_property.__get__(row, ManifestRow)

    monkeypatch.setattr(ManifestRow, "lesion_identity", property(counted_identity))

    result = audit_manifest(rows)

    assert [(issue.row_number, issue.reason_code) for issue in result.exclusions] == [
        (80, "duplicate_lesion"),
        (81, "duplicate_lesion"),
    ]
    assert len(result.included_rows) == 80
    assert accesses <= len(rows) * 3


def test_audit_treats_low_endpoint_noise_as_an_empty_mask(tmp_path: Path) -> None:
    """JPEG background values below the foreground threshold must not create a lesion."""
    row = _row(tmp_path)
    _write_jpeg(row.mask_image_path, np.full((8, 8), 5, dtype=np.uint8))

    result = audit_manifest([row])

    assert [(issue.row_number, issue.reason_code) for issue in result.exclusions] == [(0, "empty_mask")]
