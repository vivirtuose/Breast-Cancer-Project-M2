from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cbis_ddsm.data.audit import EXCLUSION_FIELDS
from cbis_ddsm.data.manifest import MANIFEST_FIELDS
from cbis_ddsm.data.splits import assign_folds, build_split, partition_of, patient_labels, read_split_json, write_split_json
from cbis_ddsm.data.targets import ImageRecord


ROOT = Path(__file__).resolve().parents[1]
MAKE_SPLITS = ROOT / "scripts" / "make_splits.py"


def _records() -> tuple[ImageRecord, ...]:
    out = []
    for i in range(40):
        split = "train" if i < 30 else "test"
        pathology = "MALIGNANT" if i % 3 == 0 else "BENIGN"
        for view in ("CC", "MLO"):
            out.append(ImageRecord(
                f"P_{i:03d}|LEFT|{view}|{split}", f"P_{i:03d}", "LEFT", view, split, Path("x.jpg"),
                (Path("m.png"),), ("1",), (pathology,), "2", ("3",),
            ))
    return tuple(out)


def test_partitions_are_patient_disjoint_and_cover_everyone():
    split = build_split(_records(), manifest_sha256="abc")
    parts = {k: set(v) for k, v in split["partitions"].items()}
    assert not parts["train"] & parts["val"] and not (parts["train"] | parts["val"]) & parts["test"]
    assert len(parts["train"] | parts["val"]) == 30 and len(parts["test"]) == 10


def test_both_views_of_a_patient_share_a_partition():
    split = build_split(_records(), manifest_sha256="abc")
    lookup = partition_of(split)
    for record in _records():
        assert lookup[record.patient_id] in {"train", "val", "test"}


def test_folds_are_stratified_and_deterministic():
    labels = patient_labels(_records())
    first, second = assign_folds(labels, n_folds=5, seed=7), assign_folds(labels, n_folds=5, seed=7)
    assert first == second
    for fold in range(5):
        members = [p for p, f in first.items() if f == fold]
        assert len(members) == 6 and sum(labels[p] for p in members) == 2


def test_split_json_round_trip_is_byte_stable(tmp_path):
    split = build_split(_records(), seed=3, manifest_sha256="abc")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_split_json(split, a)
    write_split_json(build_split(_records(), seed=3, manifest_sha256="abc"), b)
    assert a.read_bytes() == b.read_bytes() and read_split_json(a) == split


def test_patient_labels_ignores_test_split_records():
    labels = patient_labels(_records())
    assert set(labels) == {f"P_{i:03d}" for i in range(30)}
    assert labels["P_000"] == 1 and labels["P_001"] == 0


def test_build_split_rejects_patient_in_both_official_splits():
    records = list(_records())
    # Duplicate a test-split record under a patient id that is otherwise a train patient.
    records.append(ImageRecord(
        "P_000|RIGHT|CC|test", "P_000", "RIGHT", "CC", "test", Path("x.jpg"),
        (Path("m.png"),), ("1",), ("BENIGN",), "2", ("3",),
    ))
    with pytest.raises(ValueError):
        build_split(tuple(records), manifest_sha256="abc")


def test_assign_folds_rejects_n_folds_exceeding_smallest_class():
    labels = {"P_1": 1, "P_2": 0, "P_3": 0, "P_4": 0}
    with pytest.raises(ValueError):
        assign_folds(labels, n_folds=2, seed=0)


def _run(script: Path, *arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_manifest_and_exclusions(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Synthetic manifest + exclusions with 4 train patients (2 malignant, 2 benign) and 1 test patient (R3)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest.csv"
    exclusions = tmp_path / "exclusions.csv"
    patients = [
        ("P_001", "train", "MALIGNANT"),
        ("P_002", "train", "MALIGNANT"),
        ("P_003", "train", "BENIGN"),
        ("P_004", "train", "BENIGN"),
        ("P_005", "test", "BENIGN"),
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for patient_id, split, pathology in patients:
            for view in ("CC", "MLO"):
                writer.writerow({
                    "patient_id": patient_id, "laterality": "LEFT", "view": view, "abnormality_id": "1",
                    "lesion_type": "mass", "pathology": pathology, "density": "3", "assessment": "4",
                    "split": split, "source_csv": f"mass_case_description_{split}_set.csv",
                    "full_image_path": f"{patient_id}_{view}_full.jpg", "crop_image_path": f"{patient_id}_{view}_crop.jpg",
                    "mask_image_path": f"{patient_id}_{view}_mask.jpg", "subtlety": "3",
                    "shape_or_type": "", "margins_or_distribution": "",
                })
    with exclusions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
    return raw, manifest, exclusions


def test_make_splits_cli_writes_json_with_two_folds(tmp_path):
    raw, manifest, exclusions = _write_manifest_and_exclusions(tmp_path)
    output = tmp_path / "split_v1.json"

    result = _run(
        MAKE_SPLITS, "--raw-dir", raw, "--manifest", manifest, "--exclusions", exclusions,
        "--output", output, "--n-folds", "2", "--val-fold", "0", "--seed", "0",
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    split = json.loads(output.read_text(encoding="utf-8"))
    assert split["n_folds"] == 2
    assert set(split["partitions"]["test"]) == {"P_005"}
    assert set(split["partitions"]["train"]) | set(split["partitions"]["val"]) == {"P_001", "P_002", "P_003", "P_004"}


def test_make_splits_cli_refuses_output_inside_raw_dir(tmp_path):
    raw, manifest, exclusions = _write_manifest_and_exclusions(tmp_path)
    blocked = raw / "split_v1.json"

    result = _run(
        MAKE_SPLITS, "--raw-dir", raw, "--manifest", manifest, "--exclusions", exclusions,
        "--output", blocked, "--n-folds", "2",
    )

    assert result.returncode != 0
    assert "outside the raw directory" in result.stderr
    assert not blocked.exists()
