"""Stable, one-reason-per-row audit primitives for a manifest."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .manifest import ManifestRow


EXCLUSION_FIELDS = ("row_number", "reason_code", "patient_id", "lesion_identity")


@dataclass(frozen=True)
class AuditIssue:
    row_number: int
    reason_code: str
    patient_id: str
    lesion_identity: tuple[str, str, str, str, str]

    def csv_record(self) -> dict[str, str | int]:
        return {"row_number": self.row_number, "reason_code": self.reason_code, "patient_id": self.patient_id, "lesion_identity": "|".join(self.lesion_identity)}


@dataclass(frozen=True)
class AuditResult:
    included_rows: tuple[ManifestRow, ...]
    exclusions: tuple[AuditIssue, ...]


def _missing_path(row: ManifestRow) -> bool:
    return any(not path.is_file() for path in (row.full_image_path, row.crop_image_path, row.mask_image_path))


def _dimensions(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if image is None else image.shape[:2]


def _mask_reason(row: ManifestRow) -> str | None:
    image_size = _dimensions(row.full_image_path)
    mask = cv2.imread(str(row.mask_image_path), cv2.IMREAD_UNCHANGED)
    if image_size is None or mask is None:
        return "missing_path"
    if mask.shape[:2] != image_size:
        return "dimension_mismatch"
    values = np.unique(mask)
    # Binarity first: mid-grey pixels without any foreground are corruption, not an empty lesion.
    if not np.all((values <= 10) | (values >= 245)):
        return "non_binary_mask"
    if not np.any(mask >= 245):
        return "empty_mask"
    return None


def audit_manifest(rows: Iterable[ManifestRow]) -> AuditResult:
    """Apply stable exclusion precedence: missing, duplicate, dimensions, binary, empty."""
    materialized = tuple(rows)
    identity_counts = Counter(row.lesion_identity for row in materialized)
    duplicate_identities = {identity for identity, count in identity_counts.items() if count > 1}
    included: list[ManifestRow] = []
    exclusions: list[AuditIssue] = []
    for number, row in enumerate(materialized):
        if _missing_path(row):
            reason = "missing_path"
        elif row.lesion_identity in duplicate_identities:
            reason = "duplicate_lesion"
        else:
            reason = _mask_reason(row)
        if reason:
            exclusions.append(AuditIssue(number, reason, row.patient_id, row.lesion_identity))
        else:
            included.append(row)
    return AuditResult(tuple(included), tuple(exclusions))


def write_exclusions_csv(issues: Iterable[AuditIssue], output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(issue.csv_record() for issue in issues)
