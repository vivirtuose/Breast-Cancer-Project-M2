from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import cbis_ddsm.data.manifest as manifest_module
from cbis_ddsm.data.manifest import DicomEntry, ManifestResolutionError, _resolve_reference, build_manifest, discover_case_csvs, read_manifest_csv, write_manifest_csv


def _write_jpeg(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)


def _write_case_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "patient_id",
        "left or right breast",
        "image view",
        "abnormality id",
        "breast density",
        "assessment",
        "pathology",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
        "subtlety",
        "mass shape",
        "mass margins",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_dicom_csv(raw: Path, entries: list[dict[str, str]]) -> None:
    with (raw / "dicom_info.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_path", "image_path", "SeriesDescription", "SeriesInstanceUID"],
        )
        writer.writeheader()
        writer.writerows(entries)


def _case_row(**overrides: str) -> dict[str, str]:
    row = {
        "patient_id": "P_00001",
        "left or right breast": "LEFT",
        "image view": "CC",
        "abnormality id": "1",
        "breast density": "3",
        "assessment": "4",
        "pathology": "BENIGN",
        "image file path": "full/uid-full.dcm",
        # Deliberately reversed, as in the known Kaggle-mirror defect.
        "cropped image file path": "roi/uid-mask.dcm",
        "ROI mask file path": "crop/uid-crop.dcm",
        "subtlety": "4",
        "mass shape": "IRREGULAR",
        "mass margins": "SPICULATED",
    }
    row.update(overrides)
    return row


def _synthetic_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_jpeg(raw / "full.jpg", np.arange(64, dtype=np.uint8).reshape(8, 8))
    _write_jpeg(raw / "crop.jpg", np.arange(64, dtype=np.uint8).reshape(8, 8))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    _write_jpeg(raw / "mask.jpg", mask)
    _write_dicom_csv(
        raw,
        [
            {"file_path": "full/uid-full.dcm", "image_path": "full.jpg", "SeriesDescription": "full mammogram images", "SeriesInstanceUID": "uid-full"},
            {"file_path": "crop/uid-crop.dcm", "image_path": "crop.jpg", "SeriesDescription": "cropped images", "SeriesInstanceUID": "uid-crop"},
            {"file_path": "roi/uid-mask.dcm", "image_path": "mask.jpg", "SeriesDescription": "ROI mask images", "SeriesInstanceUID": "uid-mask"},
        ],
    )
    for name in (
        "calc_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
    ):
        _write_case_csv(raw / name, [])
    return raw


def test_discovery_is_filename_authoritative_and_deterministic(tmp_path: Path) -> None:
    """A filename change must not silently change split or lesion type derivation."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in [
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "mass_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]:
        (raw / name).write_text("x\n", encoding="utf-8")

    discovered = discover_case_csvs(raw)

    assert [(item.path.name, item.split, item.lesion_type) for item in discovered] == [
        ("calc_case_description_test_set.csv", "test", "calc"),
        ("calc_case_description_train_set.csv", "train", "calc"),
        ("mass_case_description_test_set.csv", "test", "mass"),
        ("mass_case_description_train_set.csv", "train", "mass"),
    ]


def test_manifest_detects_swapped_crop_and_mask_by_pixels(tmp_path: Path) -> None:
    """Trusting the two annotation column names would swap these output paths."""
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])

    rows = build_manifest(raw)

    assert len(rows) == 1
    row = rows[0]
    assert row.split == "train"
    assert row.lesion_type == "mass"
    assert row.density == "3"
    assert row.full_image_path == raw / "full.jpg"
    assert row.crop_image_path == raw / "crop.jpg"
    assert row.mask_image_path == raw / "mask.jpg"


def test_manifest_rejects_unresolved_reference(tmp_path: Path) -> None:
    """Removing a DICOM index entry must turn the lesion into an explicit acquisition exclusion."""
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(
        raw / "mass_case_description_train_set.csv",
        [_case_row(**{"ROI mask file path": "roi/not-present.dcm"})],
    )

    result = build_manifest(raw)

    assert result.rows == ()
    assert len(result.acquisition_exclusions) == 1
    assert "unresolved" in result.acquisition_exclusions[0].message


def test_manifest_rejects_ambiguous_binary_annotation_candidates(tmp_path: Path) -> None:
    """Adding a second valid sparse binary candidate must not select one arbitrarily."""
    raw = _synthetic_raw(tmp_path)
    second_mask = np.zeros((8, 8), dtype=np.uint8)
    second_mask[1:3, 1:3] = 255
    _write_jpeg(raw / "mask-two.jpg", second_mask)
    with (raw / "dicom_info.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=["file_path", "image_path", "SeriesDescription", "SeriesInstanceUID"]).writerow(
            {"file_path": "roi/uid-mask-two.dcm", "image_path": "mask-two.jpg", "SeriesDescription": "ROI mask images", "SeriesInstanceUID": "uid-mask-two"}
        )
    _write_case_csv(
        raw / "mass_case_description_train_set.csv",
        [_case_row(**{"ROI mask file path": "roi/uid-mask-two.dcm"})],
    )

    result = build_manifest(raw)

    assert result.rows == ()
    assert len(result.acquisition_exclusions) == 1
    assert "ambiguous" in result.acquisition_exclusions[0].message


def test_manifest_is_repeatable_and_accepts_underscored_density_header(tmp_path: Path) -> None:
    """Changing CSV iteration order or density spelling must not alter a normalized manifest."""
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])
    content = (raw / "mass_case_description_train_set.csv").read_text(encoding="utf-8")
    (raw / "mass_case_description_train_set.csv").write_text(content.replace("breast density", "breast_density"), encoding="utf-8")

    first = build_manifest(raw)
    second = build_manifest(raw)

    assert first == second
    assert first[0].density == "3"


def test_manifest_supports_extracted_csv_and_jpeg_subdirectories(tmp_path: Path) -> None:
    """The real extraction nests metadata in csv and removes CBIS-DDSM from JPEG paths."""
    raw = _synthetic_raw(tmp_path)
    (raw / "csv").mkdir()
    (raw / "jpeg").mkdir()
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])
    for image in ("full.jpg", "crop.jpg", "mask.jpg"):
        (raw / image).replace(raw / "jpeg" / image)
    index = (raw / "dicom_info.csv").read_text(encoding="utf-8")
    (raw / "csv" / "dicom_info.csv").write_text(index.replace("full.jpg", "CBIS-DDSM/jpeg/full.jpg").replace("crop.jpg", "CBIS-DDSM/jpeg/crop.jpg").replace("mask.jpg", "CBIS-DDSM/jpeg/mask.jpg"), encoding="utf-8")
    (raw / "dicom_info.csv").unlink()
    # Once csv/ exists it is the only metadata location, so all four case CSVs move together.
    for name in (
        "calc_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
    ):
        (raw / name).replace(raw / "csv" / name)

    rows = build_manifest(raw)

    assert rows[0].full_image_path == raw / "jpeg" / "full.jpg"
    assert rows[0].mask_image_path == raw / "jpeg" / "mask.jpg"


def test_manifest_classifies_candidates_when_crop_and_mask_share_series_uid(tmp_path: Path) -> None:
    """A shared series UID must keep both candidate JPEGs available for pixel classification."""
    raw = _synthetic_raw(tmp_path)
    index = (raw / "dicom_info.csv").read_text(encoding="utf-8")
    (raw / "dicom_info.csv").write_text(index.replace("uid-crop", "uid-annotation").replace("uid-mask", "uid-annotation"), encoding="utf-8")
    _write_case_csv(
        raw / "mass_case_description_train_set.csv",
        [_case_row(**{
            "image file path": "legacy/uid-full/000000.dcm",
            "cropped image file path": "legacy/uid-annotation/000000.dcm",
            "ROI mask file path": "legacy/uid-annotation/000001.dcm",
        })],
    )

    rows = build_manifest(raw)

    assert rows[0].crop_image_path == raw / "crop.jpg"
    assert rows[0].mask_image_path == raw / "mask.jpg"


def test_manifest_uses_unambiguous_full_reference_when_series_description_is_blank(tmp_path: Path) -> None:
    """Incomplete DICOM description metadata must not discard a uniquely resolved full image."""
    raw = _synthetic_raw(tmp_path)
    index = (raw / "dicom_info.csv").read_text(encoding="utf-8")
    (raw / "dicom_info.csv").write_text(index.replace("full mammogram images", ""), encoding="utf-8")
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])

    rows = build_manifest(raw)

    assert rows[0].full_image_path == raw / "full.jpg"


def test_manifest_recognizes_endpoint_compressed_mask_values(tmp_path: Path) -> None:
    """A sparse 249-valued JPEG mask is still binary data after endpoint compression."""
    raw = _synthetic_raw(tmp_path)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 249
    _write_jpeg(raw / "mask.jpg", mask)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])

    rows = build_manifest(raw)

    assert rows[0].mask_image_path == raw / "mask.jpg"


def test_uid_resolution_uses_indexed_candidates_without_scanning_global_entries(tmp_path: Path) -> None:
    """A global DICOM scan would make exhaustive resolution scale with every unrelated image."""
    entry = DicomEntry("converted/shared.dcm", tmp_path / "shared.jpg", "ROI mask images", "shared")

    class IndexedCandidates:
        by_exact_reference: dict[str, tuple[DicomEntry, ...]] = {}
        by_stem: dict[str, tuple[DicomEntry, ...]] = {}
        by_series_uid = {"shared": (entry,)}

        def __iter__(self):
            raise AssertionError("UID resolution must not scan the global DICOM index")

    resolved = _resolve_reference("legacy/shared/000000.dcm", IndexedCandidates())

    assert resolved == (entry,)


def test_manifest_records_one_acquisition_exclusion_and_continues_after_bad_row(tmp_path: Path) -> None:
    """A second broken reference on one source row must not stop or duplicate its exclusion."""
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(
        raw / "mass_case_description_train_set.csv",
        [
            _case_row(),
            _case_row(**{
                "patient_id": "P_BAD",
                "image file path": "missing/full.dcm",
                "ROI mask file path": "missing/mask.dcm",
            }),
        ],
    )
    try:
        result = build_manifest(raw)
    except ManifestResolutionError as error:
        pytest.fail(f"one malformed row must become an acquisition exclusion: {error}")

    assert len(result.rows) == 1
    assert len(result.acquisition_exclusions) == 1
    issue = result.acquisition_exclusions[0]
    assert (issue.source_csv, issue.source_row_index, issue.patient_id) == (
        "mass_case_description_train_set.csv", 1, "P_BAD",
    )
    assert "unresolved" in issue.message


def test_manifest_requires_all_four_canonical_case_csvs(tmp_path: Path) -> None:
    """An incomplete extraction must fail before producing a partial manifest."""
    raw = _synthetic_raw(tmp_path)
    (raw / "calc_case_description_test_set.csv").unlink()

    with pytest.raises(ManifestResolutionError, match="calc_case_description_test_set.csv"):
        build_manifest(raw)


def test_manifest_prefers_unambiguous_series_descriptions_before_pixel_fallback(tmp_path: Path, monkeypatch) -> None:
    """When metadata names one crop and one ROI, decoding both pixels is unnecessary."""
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])

    def pixel_fallback_must_not_run(path: Path) -> bool:
        raise AssertionError(f"unexpected pixel fallback for {path}")

    monkeypatch.setattr(manifest_module, "_is_mask_like", pixel_fallback_must_not_run)

    result = build_manifest(raw)

    assert result.rows[0].crop_image_path == raw / "crop.jpg"
    assert result.rows[0].mask_image_path == raw / "mask.jpg"


def test_manifest_csv_uses_raw_relative_posix_paths_and_rebases_on_read(tmp_path: Path) -> None:
    """Moving an extraction must not change manifest bytes or leave paths bound to its old root."""
    first_base = tmp_path / "first"
    first_base.mkdir()
    first_raw = _synthetic_raw(first_base)
    _write_case_csv(first_raw / "mass_case_description_train_set.csv", [_case_row()])
    second_raw = tmp_path / "second" / "raw"
    shutil.copytree(first_raw, second_raw)
    first_csv = tmp_path / "first-manifest.csv"
    second_csv = tmp_path / "second-manifest.csv"

    write_manifest_csv(build_manifest(first_raw), first_csv)
    write_manifest_csv(build_manifest(second_raw), second_csv)

    assert first_csv.read_bytes() == second_csv.read_bytes()
    restored = read_manifest_csv(first_csv, raw_dir=first_raw)
    assert restored[0].mask_image_path == first_raw / "mask.jpg"


def test_manifest_keeps_subtlety_and_morphology_columns(tmp_path: Path) -> None:
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])
    row = build_manifest(raw)[0]
    assert (row.subtlety, row.shape_or_type, row.margins_or_distribution) == ("4", "IRREGULAR", "SPICULATED")
