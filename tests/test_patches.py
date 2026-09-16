from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from cbis_ddsm.data.manifest import MANIFEST_FIELDS
from cbis_ddsm.data.patches import PatchRecord, PatchWindow, background_windows, crop_window, extract, lesion_window, reproject


ROOT = Path(__file__).resolve().parents[1]
EXTRACT_PATCHES = ROOT / "scripts" / "extract_patches.py"


def _scene():
    image = np.zeros((600, 400), dtype=np.uint8)
    image[:, :300] = 120  # "breast" on the left
    mask = np.zeros_like(image)
    mask[200:260, 100:180] = 255  # 60 x 80 lesion
    image[mask > 0] = 230
    return image, mask


def test_lesion_window_centers_lesion_and_keeps_native_resolution():
    _, mask = _scene()
    w = lesion_window(mask, size=128)
    assert w.side == 128 and (w.x0 + w.side // 2, w.y0 + w.side // 2) == (140, 230)


def test_large_lesion_grows_the_window_instead_of_cutting_it():
    mask = np.zeros((600, 400), dtype=np.uint8)
    mask[100:400, 50:250] = 255
    w = lesion_window(mask, size=128, margin=0.1)
    assert w.side == 360


def test_crop_window_pads_outside_the_image():
    image, _ = _scene()
    out = crop_window(image, PatchWindow(-10, -10, 50))
    assert out.shape == (50, 50) and out[:10, :].max() == 0 and out[10:, 10:].min() == 120


def test_extract_keeps_image_and_mask_aligned():
    image, mask = _scene()
    img, msk = extract(image, mask, lesion_window(mask, size=128), out_size=128)
    assert img.shape == msk.shape == (128, 128)
    assert np.all(img[msk > 0] == 230)


def test_reproject_inverts_extract_geometry():
    image, mask = _scene()
    w = lesion_window(mask, size=128)
    _, msk = extract(image, mask, w, out_size=64)  # downscaled patch
    rec = PatchRecord("p", "k", "P", "train", "lesion", "1", w.x0, w.y0, w.side, 64, 600, 400, "i.png", "m.png", int((msk > 0).sum()))
    back = reproject((msk > 0).astype(np.float32), rec) >= 0.5
    inter = np.logical_and(back, mask > 0).sum()
    union = np.logical_or(back, mask > 0).sum()
    assert inter / union > 0.9


def test_background_windows_avoid_lesions_and_stay_in_tissue():
    image, mask = _scene()
    windows = background_windows(image, mask, size=64, count=5, rng=np.random.default_rng(0))
    assert len(windows) == 5
    for w in windows:
        assert crop_window(mask, w).max() == 0
        assert image[w.y0 + 32, w.x0 + 32] > 0
    again = background_windows(image, mask, size=64, count=5, rng=np.random.default_rng(0))
    assert windows == again


def test_lesion_window_rejects_empty_mask():
    import pytest

    with pytest.raises(ValueError):
        lesion_window(np.zeros((20, 20), dtype=np.uint8), size=16)


def test_patch_index_round_trip(tmp_path):
    from cbis_ddsm.data.patches import PATCH_INDEX_FIELDS, read_patch_index, write_patch_index

    records = (
        PatchRecord("p1", "k1", "P_1", "train", "lesion", "1", 10, 20, 64, 64, 600, 400, "images/p1.png", "masks/p1.png", 42),
        PatchRecord("p2", "k1", "P_1", "train", "background", "", -5, 0, 64, 64, 600, 400, "images/p2.png", "masks/p2.png", 0),
    )
    path = tmp_path / "index.csv"

    write_patch_index(records, path)
    restored = read_patch_index(path)

    assert restored == records
    assert list(csv.reader(path.open(encoding="utf-8")))[0] == list(PATCH_INDEX_FIELDS)


def test_read_patch_index_rejects_mismatched_header(tmp_path):
    import pytest

    from cbis_ddsm.data.patches import PATCH_INDEX_FIELDS, read_patch_index

    path = tmp_path / "index.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*PATCH_INDEX_FIELDS[1:], "extra_column"])
        writer.writeheader()

    with pytest.raises(ValueError):
        read_patch_index(path)


# --- CLI: synthetic raw dir with 4 train patients (2 malignant, 2 benign), one image and one
# lesion each; n_folds=2 splits them 2/2 into train/val (context.md ruling R3 supersedes the
# brief's "2 train patients / 4 rows" wording).

def _write_jpeg(path: Path, pixels: np.ndarray) -> None:
    assert cv2.imwrite(str(path), pixels)


def _scene_for(patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((300, 400), dtype=np.uint8)
    image[:, :300] = 120  # "breast" tissue on the left
    mask = np.zeros_like(image)
    mask[120:160, 100:140] = 255  # 40 x 40 lesion, away from the image border
    image[mask > 0] = 230
    return image, mask


def _write_synthetic_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    exclusions_path = tmp_path / "exclusions.csv"
    split_path = tmp_path / "split_v1.json"

    patients = [
        ("P_001", "MALIGNANT"),
        ("P_002", "MALIGNANT"),
        ("P_003", "BENIGN"),
        ("P_004", "BENIGN"),
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for patient_id, pathology in patients:
            image, mask = _scene_for(patient_id)
            _write_jpeg(raw / f"{patient_id}_full.jpg", image)
            _write_jpeg(raw / f"{patient_id}_mask.jpg", mask)
            writer.writerow({
                "patient_id": patient_id, "laterality": "LEFT", "view": "CC", "abnormality_id": "1",
                "lesion_type": "mass", "pathology": pathology, "density": "3", "assessment": "4",
                "split": "train", "source_csv": "mass_case_description_train_set.csv",
                "full_image_path": f"{patient_id}_full.jpg", "crop_image_path": f"{patient_id}_full.jpg",
                "mask_image_path": f"{patient_id}_mask.jpg", "subtlety": "3",
                "shape_or_type": "", "margins_or_distribution": "",
            })

    from cbis_ddsm.data.audit import EXCLUSION_FIELDS
    with exclusions_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS).writeheader()

    from cbis_ddsm.data.manifest import read_manifest_csv
    from cbis_ddsm.data.splits import build_split, write_split_json
    from cbis_ddsm.data.targets import group_images, select_masses

    rows = read_manifest_csv(manifest_path, raw)
    records, _dropped = group_images(select_masses(rows))
    split = build_split(records, n_folds=2, val_fold=0, seed=0, manifest_sha256="abc")
    write_split_json(split, split_path)

    return raw, manifest_path, exclusions_path, split_path


def _run(*arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXTRACT_PATCHES), *(str(argument) for argument in arguments)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def test_extract_patches_cli_writes_index_and_pngs(tmp_path):
    raw, manifest_path, exclusions_path, split_path = _write_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "patches_v1"

    result = _run(
        "--raw-dir", raw, "--manifest", manifest_path, "--exclusions", exclusions_path, "--split", split_path,
        "--output-dir", output_dir, "--size", "64", "--background-per-image", "1", "--seed", "0",
        "--partitions", "train", "val",
    )

    assert result.returncode == 0, result.stderr
    index_path = output_dir / "index.csv"
    assert index_path.is_file()

    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # 4 images x (1 lesion + 1 background), per R3

    lesion_rows = [row for row in rows if row["kind"] == "lesion"]
    assert len(lesion_rows) == 4
    for row in lesion_rows:
        assert int(row["lesion_pixels"]) > 0

    for row in rows:
        image_path = output_dir / row["image_file"]
        mask_path = output_dir / row["mask_file"]
        assert image_path.is_file() and mask_path.is_file()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (64, 64)
        assert mask.shape == (64, 64)


def test_extract_patches_cli_refuses_output_inside_raw_dir(tmp_path):
    raw, manifest_path, exclusions_path, split_path = _write_synthetic_dataset(tmp_path)
    blocked = raw / "patches_v1"

    result = _run(
        "--raw-dir", raw, "--manifest", manifest_path, "--exclusions", exclusions_path, "--split", split_path,
        "--output-dir", blocked, "--size", "64",
    )

    assert result.returncode != 0
    assert "outside the raw directory" in result.stderr
    assert not blocked.exists()


def test_extract_patches_cli_limit_patients_keeps_first_n_sorted(tmp_path):
    raw, manifest_path, exclusions_path, split_path = _write_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "patches_v1"

    result = _run(
        "--raw-dir", raw, "--manifest", manifest_path, "--exclusions", exclusions_path, "--split", split_path,
        "--output-dir", output_dir, "--size", "64", "--background-per-image", "1", "--seed", "0",
        "--partitions", "train", "val", "--limit-patients", "1",
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # 1 patient per requested partition (train, val) x 2 rows each = 4
    assert len(rows) == 4
    assert len({row["patient_id"] for row in rows}) == 2
