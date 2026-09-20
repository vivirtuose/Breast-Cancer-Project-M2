from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from cbis_ddsm.data.manifest import MANIFEST_FIELDS


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_manifest.py"
AUDIT = ROOT / "scripts" / "audit_data.py"


def _run(script: Path, *arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_minimal_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    for name, image in (("full.jpg", pixels), ("crop.jpg", pixels), ("mask.jpg", mask)):
        assert cv2.imwrite(str(raw / name), image)
    with (raw / "dicom_info.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "image_path", "SeriesDescription", "SeriesInstanceUID"])
        writer.writeheader()
        writer.writerows([
            {"file_path": "full.dcm", "image_path": "full.jpg", "SeriesDescription": "full mammogram images", "SeriesInstanceUID": "full"},
            {"file_path": "crop.dcm", "image_path": "crop.jpg", "SeriesDescription": "cropped images", "SeriesInstanceUID": "crop"},
            {"file_path": "mask.dcm", "image_path": "mask.jpg", "SeriesDescription": "ROI mask images", "SeriesInstanceUID": "mask"},
        ])
    fields = ["patient_id", "left or right breast", "image view", "abnormality id", "breast density", "assessment", "pathology", "image file path", "cropped image file path", "ROI mask file path"]
    for name in ("calc_case_description_test_set.csv", "calc_case_description_train_set.csv", "mass_case_description_test_set.csv", "mass_case_description_train_set.csv"):
        with (raw / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            if name.startswith("mass_case_description_train"):
                writer.writerow({"patient_id": "P_001", "left or right breast": "LEFT", "image view": "CC", "abnormality id": "1", "breast density": "3", "assessment": "4", "pathology": "BENIGN", "image file path": "full.dcm", "cropped image file path": "crop.dcm", "ROI mask file path": "mask.dcm"})
    return raw


def test_build_and_audit_clis_write_distinct_relocatable_outputs(tmp_path: Path) -> None:
    """A successful build must emit both datasets, and audit must rebase the manifest paths."""
    raw = _write_minimal_raw(tmp_path)
    manifest = tmp_path / "manifest.csv"
    acquisition = tmp_path / "acquisition.csv"
    exclusions = tmp_path / "exclusions.csv"

    build = _run(BUILD, "--raw-dir", raw, "--output", manifest, "--acquisition-exclusions", acquisition)

    assert build.returncode == 0, build.stderr
    assert manifest.is_file() and acquisition.is_file()
    assert "full.jpg" in manifest.read_text(encoding="utf-8")
    audit = _run(AUDIT, "--raw-dir", raw, "--manifest", manifest, "--output", exclusions)
    assert audit.returncode == 0, audit.stderr
    assert exclusions.is_file()


def test_build_cli_refuses_output_inside_raw_before_writing(tmp_path: Path) -> None:
    raw = _write_minimal_raw(tmp_path)
    blocked = raw / "manifest.csv"

    result = _run(BUILD, "--raw-dir", raw, "--output", blocked, "--acquisition-exclusions", tmp_path / "acquisition.csv")

    assert result.returncode != 0
    assert "outside the raw directory" in result.stderr
    assert not blocked.exists()


def test_build_cli_refuses_identical_output_paths_before_writing(tmp_path: Path) -> None:
    raw = _write_minimal_raw(tmp_path)
    output = tmp_path / "same.csv"

    result = _run(BUILD, "--raw-dir", raw, "--output", output, "--acquisition-exclusions", output)

    assert result.returncode != 0
    assert "must be different files" in result.stderr
    assert not output.exists()


def _write_manifest_with_one_mass_and_one_broken_calc_row(tmp_path: Path) -> tuple[Path, Path]:
    """A valid mass row plus a calc row whose files do not exist on disk."""
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest.csv"
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    for name, image in (("mass_full.jpg", pixels), ("mass_crop.jpg", pixels), ("mass_mask.jpg", mask)):
        assert cv2.imwrite(str(raw / name), image)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow({
            "patient_id": "P_001", "laterality": "LEFT", "view": "CC", "abnormality_id": "1",
            "lesion_type": "mass", "pathology": "BENIGN", "density": "3", "assessment": "4",
            "split": "train", "source_csv": "mass_case_description_train_set.csv",
            "full_image_path": "mass_full.jpg", "crop_image_path": "mass_crop.jpg",
            "mask_image_path": "mass_mask.jpg", "subtlety": "3", "shape_or_type": "", "margins_or_distribution": "",
        })
        writer.writerow({
            "patient_id": "P_002", "laterality": "RIGHT", "view": "MLO", "abnormality_id": "1",
            "lesion_type": "calc", "pathology": "BENIGN", "density": "3", "assessment": "4",
            "split": "train", "source_csv": "calc_case_description_train_set.csv",
            "full_image_path": "calc_missing_full.jpg", "crop_image_path": "calc_missing_crop.jpg",
            "mask_image_path": "calc_missing_mask.jpg", "subtlety": "3", "shape_or_type": "", "margins_or_distribution": "",
        })
    return raw, manifest


def test_audit_cli_lesion_type_mass_skips_calc_rows_before_auditing(tmp_path: Path) -> None:
    """--lesion-type mass must drop the calc row before auditing, so its missing files never surface."""
    raw, manifest = _write_manifest_with_one_mass_and_one_broken_calc_row(tmp_path)
    exclusions = tmp_path / "exclusions.csv"

    result = _run(AUDIT, "--raw-dir", raw, "--manifest", manifest, "--output", exclusions, "--lesion-type", "mass")

    assert result.returncode == 0, result.stderr
    assert "wrote 0 exclusions; retained 1 rows" in result.stdout
    with exclusions.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_audit_cli_default_lesion_type_reports_the_broken_calc_row(tmp_path: Path) -> None:
    """Without --lesion-type, the calc row's missing files are audited like any other row."""
    raw, manifest = _write_manifest_with_one_mass_and_one_broken_calc_row(tmp_path)
    exclusions = tmp_path / "exclusions.csv"

    result = _run(AUDIT, "--raw-dir", raw, "--manifest", manifest, "--output", exclusions)

    assert result.returncode == 0, result.stderr
    assert "wrote 1 exclusions; retained 1 rows" in result.stdout


def test_audit_cli_refuses_to_overwrite_its_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sentinel", encoding="utf-8")

    result = _run(AUDIT, "--raw-dir", tmp_path / "raw", "--manifest", manifest, "--output", manifest)

    assert result.returncode != 0
    assert "must not overwrite the manifest" in result.stderr
    assert manifest.read_text(encoding="utf-8") == "sentinel"
