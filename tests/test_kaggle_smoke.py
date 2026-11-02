from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SMOKE = ROOT / "scripts" / "kaggle_smoke.py"

sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_smoke  # noqa: E402  (module under test, imported for the fast unit tests below)


CASE_FIELDS = (
    "patient_id", "left or right breast", "image view", "abnormality id", "breast density",
    "assessment", "pathology", "image file path", "cropped image file path", "ROI mask file path",
)
DICOM_FIELDS = ("file_path", "image_path", "SeriesDescription", "SeriesInstanceUID")


def _scene(width: int = 256, height: int = 192) -> tuple[np.ndarray, np.ndarray]:
    """A 256x192 synthetic mammogram: tissue on the left, one bright 50x40 lesion inside it."""
    image = np.zeros((height, width), dtype=np.uint8)
    tissue_width = int(width * 0.75)
    image[:, :tissue_width] = 120
    mask = np.zeros_like(image)
    top, left = height // 3, width // 6
    mask[top:top + 40, left:left + 50] = 255
    image[mask > 0] = 230
    return image, mask


def _write_synthetic_raw_dir(raw: Path) -> None:
    """4 train patients (2 malignant, 2 benign) + 2 test patients; real csv/ + jpeg/ layout (R3)."""
    csv_dir, jpeg_dir = raw / "csv", raw / "jpeg"
    csv_dir.mkdir(parents=True)
    jpeg_dir.mkdir(parents=True)

    patients = [
        ("P_001", "train", "MALIGNANT"),
        ("P_002", "train", "MALIGNANT"),
        ("P_003", "train", "BENIGN"),
        ("P_004", "train", "BENIGN"),
        ("P_005", "test", "BENIGN"),
        ("P_006", "test", "BENIGN"),
    ]

    dicom_rows: list[dict[str, str]] = []
    mass_train_rows: list[dict[str, str]] = []
    mass_test_rows: list[dict[str, str]] = []
    for patient_id, split, pathology in patients:
        patient_dir = jpeg_dir / patient_id
        patient_dir.mkdir()
        image, mask = _scene()
        assert cv2.imwrite(str(patient_dir / "full.jpg"), image)
        assert cv2.imwrite(str(patient_dir / "crop.jpg"), image)
        assert cv2.imwrite(str(patient_dir / "mask.jpg"), mask)

        full_ref, crop_ref, mask_ref = f"{patient_id}_full.dcm", f"{patient_id}_crop.dcm", f"{patient_id}_mask.dcm"
        dicom_rows.extend([
            {"file_path": full_ref, "image_path": f"jpeg/{patient_id}/full.jpg", "SeriesDescription": "full mammogram images", "SeriesInstanceUID": f"{patient_id}-full"},
            {"file_path": crop_ref, "image_path": f"jpeg/{patient_id}/crop.jpg", "SeriesDescription": "cropped images", "SeriesInstanceUID": f"{patient_id}-crop"},
            {"file_path": mask_ref, "image_path": f"jpeg/{patient_id}/mask.jpg", "SeriesDescription": "ROI mask images", "SeriesInstanceUID": f"{patient_id}-mask"},
        ])
        row = {
            "patient_id": patient_id, "left or right breast": "LEFT", "image view": "CC", "abnormality id": "1",
            "breast density": "3", "assessment": "4", "pathology": pathology,
            "image file path": full_ref, "cropped image file path": crop_ref, "ROI mask file path": mask_ref,
        }
        (mass_train_rows if split == "train" else mass_test_rows).append(row)

    with (csv_dir / "dicom_info.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DICOM_FIELDS)
        writer.writeheader()
        writer.writerows(dicom_rows)

    for name, rows in (
        ("mass_case_description_train_set.csv", mass_train_rows),
        ("mass_case_description_test_set.csv", mass_test_rows),
        ("calc_case_description_train_set.csv", []),
        ("calc_case_description_test_set.csv", []),
    ):
        with (csv_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


@pytest.mark.slow
@pytest.mark.integration
def test_kaggle_smoke_end_to_end_on_synthetic_raw_dir(tmp_path: Path) -> None:
    """The orchestrator chains every stage on a tiny synthetic dataset and reports success."""
    raw = tmp_path / "raw"
    _write_synthetic_raw_dir(raw)
    work = tmp_path / "work"

    result = subprocess.run(
        [
            sys.executable, str(KAGGLE_SMOKE),
            "--raw-dir", str(raw), "--work-dir", str(work),
            "--size", "64", "--epochs", "1", "--overfit-steps", "5",
            "--encoder-weights", "none", "--device", "cpu",
            "--limit-patients", "4", "--n-folds", "2",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    report_path = work / "smoke_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stage_names = [stage["name"] for stage in report["stages"]]
    assert stage_names == [
        "capture_environment", "build_manifest", "audit", "make_splits",
        "extract_patches", "overfit", "train", "predict",
    ]
    for stage in report["stages"]:
        if stage["name"] != "overfit":
            assert stage["status"] == "ok", (stage["name"], stage["detail"])
    assert {"started_at_utc", "finished_at_utc", "work_dir", "arguments"} <= report.keys()

    zip_path = work / "smoke_run.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert "smoke_report.json" in names
    assert "env.yaml" in names
    assert "splits/split_v1.json" in names
    assert "processed/patches_v1/index.csv" in names
    assert any(name.endswith("checkpoint_best.pt") for name in names)
    # The run directory's own overlay figures are part of "the run directory" and are kept; only
    # the *patch dataset* PNGs (processed/patches_v1/images|masks) must be excluded from the zip.
    assert not any(name.startswith("processed/patches_v1/") and name.endswith(".png") for name in names)


def test_main_refuses_work_dir_inside_raw_dir(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    blocked = raw / "work"

    result = subprocess.run(
        [sys.executable, str(KAGGLE_SMOKE), "--raw-dir", str(raw), "--work-dir", str(blocked)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )

    assert result.returncode != 0
    assert "outside the raw directory" in result.stderr
    assert not blocked.exists()


def test_main_refuses_missing_raw_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    work = tmp_path / "work"

    result = subprocess.run(
        [sys.executable, str(KAGGLE_SMOKE), "--raw-dir", str(missing), "--work-dir", str(work)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )

    assert result.returncode != 0
    assert "missing raw directory" in result.stderr


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_smoke_skips_remaining_stages_after_a_fatal_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing required stage (audit here) must abort the rest of the pipeline and exit 1."""
    raw = tmp_path / "raw"
    raw.mkdir()
    work = tmp_path / "work"

    def fake_run(command, *, cwd, text, capture_output):
        script = Path(command[1]).name
        if script == "audit_data.py":
            return _FakeCompleted(1, stdout="", stderr="boom")
        return _FakeCompleted(0, stdout=f"wrote stage output for {script}", stderr="")

    monkeypatch.setattr(kaggle_smoke.subprocess, "run", fake_run)
    args = kaggle_smoke.build_arg_parser().parse_args(["--raw-dir", str(raw), "--work-dir", str(work)])

    exit_code, report = kaggle_smoke.run_smoke(args)

    assert exit_code == 1
    statuses = {stage["name"]: stage["status"] for stage in report["stages"]}
    assert statuses["build_manifest"] == "ok"
    assert statuses["audit"] == "failed"
    for name in ("make_splits", "extract_patches", "overfit", "train", "predict"):
        assert statuses[name] == "skipped"


def test_run_smoke_skip_if_present_avoids_rerunning_build_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-audit-if-present must not invoke build_manifest.py when the manifest already exists."""
    raw = tmp_path / "raw"
    raw.mkdir()
    work = tmp_path / "work"
    manifest_path = work / "interim" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("existing manifest", encoding="utf-8")

    invoked_scripts: list[str] = []

    def fake_run(command, *, cwd, text, capture_output):
        script = Path(command[1]).name
        invoked_scripts.append(script)
        if script == "train.py":
            return _FakeCompleted(0, stdout="wrote run to nonexistent_run: best_epoch=0 best_val_dice=0.5", stderr="")
        return _FakeCompleted(0, stdout=f"wrote stage output for {script}", stderr="")

    monkeypatch.setattr(kaggle_smoke.subprocess, "run", fake_run)
    args = kaggle_smoke.build_arg_parser().parse_args(
        ["--raw-dir", str(raw), "--work-dir", str(work), "--skip-audit-if-present"]
    )

    kaggle_smoke.run_smoke(args)

    assert "build_manifest.py" not in invoked_scripts


def test_collect_zip_entries_skips_files_over_the_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_smoke, "MAX_ZIP_FILE_BYTES", 1)
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / "small.bin").write_bytes(b"")
    (run_dir / "big.bin").write_bytes(b"0123456789")
    env_path = tmp_path / "env.yaml"
    env_path.write_text("{}", encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    index_path = tmp_path / "index.csv"
    index_path.write_text("a,b\n", encoding="utf-8")

    entries, skipped = kaggle_smoke._collect_zip_entries(run_dir, env_path, split_path, index_path)

    arcnames = {arcname for _, arcname in entries}
    assert f"{run_dir.name}/small.bin" in arcnames
    assert f"{run_dir.name}/big.bin" not in arcnames
    assert f"{run_dir.name}/big.bin" in skipped


def _captured_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[list[str]]:
    """Run the orchestrator with every stage faked, returning the argv of each invoked stage."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, text, capture_output):
        commands.append([str(part) for part in command])
        script = Path(command[1]).name
        if script == "train.py":
            return _FakeCompleted(0, stdout="wrote run to some_run: best_epoch=0 best_val_dice=0.5", stderr="")
        return _FakeCompleted(0, stdout=f"wrote stage output for {script}", stderr="")

    monkeypatch.setattr(kaggle_smoke.subprocess, "run", fake_run)
    args = kaggle_smoke.build_arg_parser().parse_args(
        ["--raw-dir", str(raw), "--work-dir", str(tmp_path / "work"), *argv]
    )
    kaggle_smoke.run_smoke(args)
    return commands


def _command_for(commands: list[list[str]], script_name: str) -> list[str]:
    return next(command for command in commands if Path(command[1]).name == script_name)


def test_run_smoke_drives_the_requested_train_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--train-config must reach both the overfit gate and the training stage."""
    config = tmp_path / "full.yaml"
    config.write_text("patch_root: x\n", encoding="utf-8")

    commands = _captured_commands(tmp_path, monkeypatch, ["--train-config", str(config)])

    for script in ("overfit_batch.py", "train.py"):
        command = _command_for(commands, script)
        assert command[command.index("--config") + 1] == str(config)


def test_run_smoke_defaults_to_the_smoke_train_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _captured_commands(tmp_path, monkeypatch, [])
    command = _command_for(commands, "train.py")
    assert command[command.index("--config") + 1] == str(kaggle_smoke.DEFAULT_TRAIN_CONFIG)


def test_run_smoke_leaves_the_config_epochs_alone_unless_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-training config must keep its own epoch count when --epochs is not given."""
    commands = _captured_commands(tmp_path, monkeypatch, [])
    assert not any(part.startswith("epochs=") for part in _command_for(commands, "train.py"))

    commands = _captured_commands(tmp_path, monkeypatch, ["--epochs", "7"])
    assert "epochs=7" in _command_for(commands, "train.py")


def test_run_smoke_leaves_the_config_encoder_weights_alone_unless_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _captured_commands(tmp_path, monkeypatch, [])
    assert not any(part.startswith("encoder_weights=") for part in _command_for(commands, "train.py"))

    commands = _captured_commands(tmp_path, monkeypatch, ["--encoder-weights", "none"])
    assert "encoder_weights=null" in _command_for(commands, "train.py")


def test_all_patients_removes_the_extraction_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _captured_commands(tmp_path, monkeypatch, ["--all-patients"])
    assert "--limit-patients" not in _command_for(commands, "extract_patches.py")

    commands = _captured_commands(tmp_path, monkeypatch, [])
    assert "--limit-patients" in _command_for(commands, "extract_patches.py")


def test_resume_run_passes_resume_from_to_the_training_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "experiments" / "run_x"
    commands = _captured_commands(tmp_path, monkeypatch, ["--resume-run", str(run_dir)])
    assert f"resume_from={run_dir}" in _command_for(commands, "train.py")
    assert not any(part.startswith("run_name=") for part in _command_for(commands, "train.py"))

    commands = _captured_commands(tmp_path, monkeypatch, [])
    assert not any(part.startswith("resume_from=") for part in _command_for(commands, "train.py"))


def test_limit_patients_and_all_patients_are_mutually_exclusive(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    result = subprocess.run(
        [
            sys.executable, str(KAGGLE_SMOKE), "--raw-dir", str(raw),
            "--work-dir", str(tmp_path / "work"), "--all-patients", "--limit-patients", "5",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
