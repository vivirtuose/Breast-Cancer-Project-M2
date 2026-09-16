"""Deterministic CBIS-DDSM case metadata and image-path resolution."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import cv2
import numpy as np


CASE_FILES = (
    ("calc_case_description_test_set.csv", "test", "calc"),
    ("calc_case_description_train_set.csv", "train", "calc"),
    ("mass_case_description_test_set.csv", "test", "mass"),
    ("mass_case_description_train_set.csv", "train", "mass"),
)
MANIFEST_FIELDS = (
    "patient_id", "laterality", "view", "abnormality_id", "lesion_type", "pathology",
    "density", "assessment", "split", "source_csv", "full_image_path", "crop_image_path",
    "mask_image_path",
)
ACQUISITION_EXCLUSION_FIELDS = (
    "source_csv", "source_row_index", "patient_id", "lesion_identity", "message",
)


class ManifestResolutionError(ValueError):
    """A case CSV row cannot be mapped to exactly one valid image triplet."""


@dataclass(frozen=True)
class CaseCsv:
    path: Path
    split: str
    lesion_type: str


@dataclass(frozen=True)
class ManifestRow:
    patient_id: str
    laterality: str
    view: str
    abnormality_id: str
    lesion_type: str
    pathology: str
    density: str
    assessment: str
    split: str
    source_csv: str
    full_image_path: Path
    crop_image_path: Path
    mask_image_path: Path

    @property
    def lesion_identity(self) -> tuple[str, str, str, str, str]:
        return (self.patient_id, self.laterality, self.view, self.abnormality_id, self.lesion_type)

    def csv_record(self, raw_dir: Path) -> dict[str, str]:
        record = asdict(self)
        for key in ("full_image_path", "crop_image_path", "mask_image_path"):
            record[key] = Path(record[key]).resolve().relative_to(raw_dir).as_posix()
        return {key: str(record[key]) for key in MANIFEST_FIELDS}


@dataclass(frozen=True)
class AcquisitionIssue:
    source_csv: str
    source_row_index: int
    patient_id: str
    lesion_identity: tuple[str, str, str, str, str]
    message: str

    def csv_record(self) -> dict[str, str | int]:
        return {
            "source_csv": self.source_csv,
            "source_row_index": self.source_row_index,
            "patient_id": self.patient_id,
            "lesion_identity": "|".join(self.lesion_identity),
            "message": self.message,
        }


@dataclass(frozen=True)
class ManifestBuildResult:
    raw_dir: Path
    rows: tuple[ManifestRow, ...]
    acquisition_exclusions: tuple[AcquisitionIssue, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ManifestRow:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


@dataclass(frozen=True)
class DicomEntry:
    file_path: str
    image_path: Path
    series_description: str
    series_instance_uid: str


@dataclass(frozen=True)
class DicomIndex:
    """Immutable deterministic lookup tables for case-DICOM references."""

    by_exact_reference: Mapping[str, tuple[DicomEntry, ...]]
    by_stem: Mapping[str, tuple[DicomEntry, ...]]
    by_series_uid: Mapping[str, tuple[DicomEntry, ...]]


def _normalise_reference(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./").casefold()


def _without_suffix(value: str) -> str:
    return str(Path(value)).rsplit(".", 1)[0] if "." in Path(value).name else str(Path(value))


def _normalise_header(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").casefold().split())


def _value(row: dict[str, str], header: str) -> str:
    wanted = _normalise_header(header)
    for key, value in row.items():
        if _normalise_header(key) == wanted:
            return (value or "").strip()
    return ""


def discover_case_csvs(raw_dir: Path | str) -> tuple[CaseCsv, ...]:
    """Find the canonical case descriptions in a stable filename-derived order."""
    raw = Path(raw_dir)
    metadata_dir = raw / "csv" if (raw / "csv").is_dir() else raw
    return tuple(
        CaseCsv(metadata_dir / name, split, lesion_type)
        for name, split, lesion_type in CASE_FILES
        if (metadata_dir / name).is_file()
    )


def _freeze_lookup(groups: dict[str, list[DicomEntry]]) -> Mapping[str, tuple[DicomEntry, ...]]:
    return MappingProxyType({
        key: tuple(sorted(entries, key=lambda entry: str(entry.image_path)))
        for key, entries in sorted(groups.items())
    })


def build_dicom_index(raw_dir: Path | str) -> DicomIndex:
    raw = Path(raw_dir).resolve()
    metadata_dir = raw / "csv" if (raw / "csv").is_dir() else raw
    source = metadata_dir / "dicom_info.csv"
    if not source.is_file():
        raise ManifestResolutionError(f"missing DICOM index: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "image_path", "SeriesDescription", "SeriesInstanceUID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ManifestResolutionError(f"DICOM index has unexpected headers: {source}")
        entries = []
        for record in reader:
            image_value = (record["image_path"] or "").strip()
            if not image_value:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                path_parts = image_path.as_posix().split("/")
                jpeg_index = next((i for i, part in enumerate(path_parts) if part.casefold() == "jpeg"), None)
                candidates = [raw / image_path]
                if jpeg_index is not None:
                    candidates.append(raw / "jpeg" / Path(*path_parts[jpeg_index + 1:]))
                image_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            entries.append(DicomEntry(
                file_path=(record["file_path"] or "").strip(),
                image_path=image_path,
                series_description=(record["SeriesDescription"] or "").strip(),
                series_instance_uid=(record["SeriesInstanceUID"] or "").strip(),
            ))
    exact: dict[str, list[DicomEntry]] = {}
    stems: dict[str, list[DicomEntry]] = {}
    series_uids: dict[str, list[DicomEntry]] = {}
    for entry in entries:
        for reference in (entry.file_path, str(entry.image_path)):
            exact.setdefault(_normalise_reference(reference), []).append(entry)
            stems.setdefault(_normalise_reference(_without_suffix(reference)), []).append(entry)
        if entry.series_instance_uid:
            series_uids.setdefault(_normalise_reference(entry.series_instance_uid), []).append(entry)
    return DicomIndex(
        by_exact_reference=_freeze_lookup(exact),
        by_stem=_freeze_lookup(stems),
        by_series_uid=_freeze_lookup(series_uids),
    )


def _resolve_reference(reference: str, index: DicomIndex) -> tuple[DicomEntry, ...]:
    normalized = _normalise_reference(reference)
    stem = _normalise_reference(_without_suffix(reference))
    matches = index.by_exact_reference.get(normalized) or index.by_stem.get(stem) or ()
    if not matches:
        matches = tuple(
            entry
            for component in normalized.split("/")
            for entry in index.by_series_uid.get(component, ())
        )
    unique = {entry.image_path.resolve(): entry for entry in matches}
    if not unique:
        raise ManifestResolutionError(f"unresolved DICOM reference {reference!r}")
    return tuple(sorted(unique.values(), key=lambda entry: str(entry.image_path)))


def _is_mask_like(path: Path) -> bool:
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.size == 0:
        raise ManifestResolutionError(f"unreadable candidate image: {path}")
    values = np.unique(pixels)
    # JPEG compression can turn 0/255 source pixels into neighbouring endpoint values.
    binary = bool(np.all((values <= 10) | (values >= 245)))
    nonzero_fraction = float(np.count_nonzero(pixels >= 245)) / float(pixels.size)
    return binary and nonzero_fraction <= 0.5


def _resolve_row(case: CaseCsv, record: dict[str, str], index: DicomIndex) -> ManifestRow:
    references = [_value(record, "image file path"), _value(record, "cropped image file path"), _value(record, "ROI mask file path")]
    if any(not reference for reference in references):
        raise ManifestResolutionError(f"missing image reference in {case.path.name}")
    candidates = [_resolve_reference(reference, index) for reference in references]
    full = [entry for entry in candidates[0] if "full mammogram" in entry.series_description.casefold()]
    if not full and len(candidates[0]) == 1:
        # Some records have no SeriesDescription, but the full-image CSV reference remains unambiguous.
        full = list(candidates[0])
    if len(full) != 1:
        raise ManifestResolutionError(f"ambiguous full-image resolution in {case.path.name}: {len(full)} candidates")
    annotations = list({entry.image_path.resolve(): entry for group in candidates[1:] for entry in group if entry != full[0]}.values())
    if len(annotations) != 2:
        raise ManifestResolutionError(f"ambiguous annotation resolution in {case.path.name}: {len(annotations)} candidates")
    described_masks = [entry for entry in annotations if "roi mask" in entry.series_description.casefold()]
    described_crops = [entry for entry in annotations if "cropped" in entry.series_description.casefold()]
    if len(described_masks) == 1 and len(described_crops) == 1 and described_masks[0] != described_crops[0]:
        mask, crop = described_masks[0], described_crops[0]
    else:
        masks = [entry for entry in annotations if _is_mask_like(entry.image_path)]
        if len(masks) != 1:
            raise ManifestResolutionError(f"ambiguous crop/mask resolution in {case.path.name}: {len(masks)} mask-like candidates")
        mask = masks[0]
        crop = next(entry for entry in annotations if entry != mask)
    return ManifestRow(
        patient_id=_value(record, "patient id"), laterality=_value(record, "left or right breast"), view=_value(record, "image view"),
        abnormality_id=_value(record, "abnormality id"), lesion_type=case.lesion_type, pathology=_value(record, "pathology"),
        density=_value(record, "breast density"), assessment=_value(record, "assessment"), split=case.split, source_csv=case.path.name,
        full_image_path=full[0].image_path, crop_image_path=crop.image_path, mask_image_path=mask.image_path,
    )


def build_manifest(raw_dir: Path | str) -> ManifestBuildResult:
    """Load every available canonical case CSV and resolve one row per lesion."""
    raw = Path(raw_dir).resolve()
    metadata_dir = raw / "csv" if (raw / "csv").is_dir() else raw
    missing = [name for name, _, _ in CASE_FILES if not (metadata_dir / name).is_file()]
    if missing:
        raise ManifestResolutionError(f"missing canonical case CSVs: {', '.join(missing)}")
    cases = discover_case_csvs(raw)
    index = build_dicom_index(raw)
    rows: list[ManifestRow] = []
    acquisition_exclusions: list[AcquisitionIssue] = []
    for case in cases:
        with case.path.open(newline="", encoding="utf-8-sig") as handle:
            for source_row_index, record in enumerate(csv.DictReader(handle)):
                try:
                    rows.append(_resolve_row(case, record, index))
                except ManifestResolutionError as error:
                    identity = (
                        _value(record, "patient id"),
                        _value(record, "left or right breast"),
                        _value(record, "image view"),
                        _value(record, "abnormality id"),
                        case.lesion_type,
                    )
                    acquisition_exclusions.append(AcquisitionIssue(
                        source_csv=case.path.name,
                        source_row_index=source_row_index,
                        patient_id=identity[0],
                        lesion_identity=identity,
                        message=str(error),
                    ))
    return ManifestBuildResult(raw, tuple(rows), tuple(acquisition_exclusions))


def write_manifest_csv(rows: ManifestBuildResult | Iterable[ManifestRow], output_path: Path | str, raw_dir: Path | str | None = None) -> None:
    if isinstance(rows, ManifestBuildResult):
        raw = rows.raw_dir
        manifest_rows = rows.rows
    else:
        if raw_dir is None:
            raise ValueError("raw_dir is required when writing manifest rows directly")
        raw = Path(raw_dir).resolve()
        manifest_rows = rows
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(row.csv_record(raw) for row in manifest_rows)


def write_acquisition_exclusions_csv(issues: Iterable[AcquisitionIssue], output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACQUISITION_EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(issue.csv_record() for issue in issues)


def read_manifest_csv(path: Path | str, raw_dir: Path | str) -> tuple[ManifestRow, ...]:
    raw = Path(raw_dir).resolve()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != MANIFEST_FIELDS:
            raise ValueError("manifest CSV headers do not match the manifest contract")
        return tuple(ManifestRow(
            patient_id=record["patient_id"], laterality=record["laterality"], view=record["view"], abnormality_id=record["abnormality_id"],
            lesion_type=record["lesion_type"], pathology=record["pathology"], density=record["density"], assessment=record["assessment"],
            split=record["split"], source_csv=record["source_csv"], full_image_path=raw / Path(record["full_image_path"]),
            crop_image_path=raw / Path(record["crop_image_path"]), mask_image_path=raw / Path(record["mask_image_path"]),
        ) for record in reader)
