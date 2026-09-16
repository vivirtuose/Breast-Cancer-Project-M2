# Kaggle Smoke Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every missing stage between the existing lesion manifest (S1) and a first end-to-end *smoke run* on Kaggle: mass targets with fused masks, patient-grouped splits, native-resolution patches, a synchronized-augmentation Dataset, a U-Net baseline, a trainer with an overfit gate, patch inference with overlays, and a single Kaggle entry point.

**Architecture:** A linear, file-based pipeline. Each stage is a pure library module in `src/cbis_ddsm/<layer>/` plus a thin CLI in `scripts/`. Stages communicate only through versioned files: `data/interim/manifest.csv` → `data/interim/exclusions.csv` → `data/splits/split_v1.json` → `data/processed/patches_v1/index.csv` + PNGs → `experiments/<run_id>/` (config, metrics, checkpoint, predictions, figures). All paths come from CLI arguments or YAML configs; nothing is machine-specific, so the same commands run on the laptop and on Kaggle.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, segmentation-models-pytorch, scikit-learn, albumentations 1.4.24, OpenCV headless, NumPy 1.26.4, pandas, PyYAML, pytest. Environment managed by `uv` (`uv.lock`).

**Spec:** `PLAN.md` (stages 2–3), `docs/PROTOCOLE.md`, `docs/ARCHITECTURE.md`, `schemas/coding_plan.svg` (S2–S6), `schemas/labels_and_splits.svg` (lessons 1–4).

## Global Constraints

- Target: **masses only** (`lesion_type == "mass"`), all three pathologies (`MALIGNANT`, `BENIGN`, `BENIGN_WITHOUT_CALLBACK`) are positive lesions; pathology is metadata, never a model input.
- Python `>=3.12,<3.13`; every direct dependency pinned `name==version` in `pyproject.toml` and resolved in `uv.lock`.
- Never write inside `data/raw`. Never modify raw images.
- No absolute or machine-specific paths in code or configs. Manifest paths stay relative to `--raw-dir`.
- Patient is the split unit: `train ∩ val = ∅`, `(train ∪ val) ∩ test = ∅`. The official test split is never used for training or selection.
- Masks are binarized with `pixel >= 128` after loading (JPEG compression moves 255 to ~249 and 0 to small values).
- An image with at least one excluded lesion row is excluded **entirely** (its fused mask would be incomplete).
- The 78 `dimension_mismatch` rows stay excluded (decision of 2026-09-16; stretching was measured at 40 px median error vs 7 px for normal rows).
- Patches are cut at **native resolution**; the whole mammogram is never resized for training. Default patch size: 512.
- Tests are portable: synthetic data under `tmp_path`, CPU only, no network, no dependency on `data/raw`. Tests that need real data or a GPU are marked `integration` / `gpu` and skipped when unavailable.
- Determinism: every random operation takes an explicit `seed`.
- Test command for the whole suite: `.venv/Scripts/python.exe -m pytest -q` (Windows) — must stay green after every task.
- Commits: one commit per task on branch `s2-s6-kaggle-smoke`. Git needs `-c "safe.directory=<repo path>"` on this machine (the `.git` folder belongs to another Windows user). End every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Code comments and identifiers in English; match the style of `src/cbis_ddsm/data/manifest.py` (frozen dataclasses, `from __future__ import annotations`, small pure functions, CLIs that validate paths before writing and report errors through `parser.error`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | + torchvision, segmentation-models-pytorch, scikit-learn |
| `scripts/capture_environment.py` | + `--record-only` mode for Kaggle (records drift, never blocks) |
| `src/cbis_ddsm/data/manifest.py` | + `subtlety`, `shape_or_type`, `margins_or_distribution` columns |
| `src/cbis_ddsm/data/targets.py` | masses → `ImageRecord` (one per image), exclusion propagation, mask fusion |
| `src/cbis_ddsm/data/splits.py` | patient labels, stratified patient folds, split JSON I/O |
| `scripts/make_splits.py` | CLI: manifest + exclusions → `split_v1.json` |
| `src/cbis_ddsm/data/patches.py` | patch geometry, extraction, re-projection, patch index I/O |
| `scripts/extract_patches.py` | CLI: images + split → PNG patches + `index.csv` |
| `src/cbis_ddsm/data/transforms.py` | synchronized albumentations pipelines |
| `src/cbis_ddsm/data/dataset.py` | `PatchDataset`, `make_fake_patch_dir` |
| `src/cbis_ddsm/models/factory.py` | `build_model(name, …)` over smp architectures |
| `src/cbis_ddsm/training/losses.py` | `BCEDiceLoss` |
| `src/cbis_ddsm/evaluation/metrics.py` | Dice, IoU with explicit empty-mask rules |
| `src/cbis_ddsm/training/config.py` | `TrainConfig` dataclass + YAML loader |
| `src/cbis_ddsm/training/trainer.py` | seeded training loop, AMP, accumulation, checkpoints, run directory |
| `scripts/train.py`, `scripts/overfit_batch.py` | CLIs |
| `src/cbis_ddsm/inference/predict.py` | patch predictions + per-patch metrics |
| `src/cbis_ddsm/evaluation/overlays.py` | image / ground truth / prediction figure |
| `scripts/predict.py` | CLI |
| `configs/unet_smoke.yaml` | smoke configuration |
| `scripts/kaggle_smoke.py`, `docs/KAGGLE.md` | one command for the whole smoke run + instructions |

---

### Task 1: Dependencies and Kaggle-tolerant environment capture

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Modify: `tests/test_dependency_lock.py` (`CANONICAL_DIRECT_DISTRIBUTIONS`)
- Modify: `scripts/capture_environment.py` (`REQUIRED_PACKAGES`, new `--record-only`)
- Test: `tests/test_dependency_lock.py`, `tests/test_capture_environment.py`

**Interfaces:**
- Produces: importable `torchvision`, `segmentation_models_pytorch`, `sklearn` in `.venv`; `capture_environment.main(["--record-only", "--output", path])` returns 0 even when the lock does not match.

- [ ] **Step 1: Add dependencies with uv.** torchvision must come from the same explicit cu121 index as torch.
  In `pyproject.toml` add to `dependencies`: `"torchvision==0.20.1+cu121"`, `"segmentation-models-pytorch==<latest release compatible with torch 2.5.1>"`, `"scikit-learn==<latest release supporting numpy 1.26 and Python 3.12>"`, and in `[tool.uv.sources]` add `torchvision = { index = "pytorch-cu121" }`. Then run `uv lock` and `uv sync --frozen`. If a resolution conflict appears (e.g. smp pulling a CPU torchvision from PyPI), fix it through `[tool.uv.sources]`, never by unpinning torch. Record the exact versions chosen.
- [ ] **Step 2: Update the lock contract tests.** Add `"torchvision"`, `"segmentation-models-pytorch"`, `"scikit-learn"` to `CANONICAL_DIRECT_DISTRIBUTIONS`; extend the torch-index test so `torchvision` is also required to use the `pytorch-cu121` index. Add the three names to `REQUIRED_PACKAGES` in `scripts/capture_environment.py`.
- [ ] **Step 3: Write the failing test for `--record-only`.**

```python
def test_record_only_mode_exits_zero_and_marks_snapshot(tmp_path, monkeypatch):
    blocked = {"cuda_gate": {"passed": False}, "capture": {"status": "blocked"}}
    monkeypatch.setattr(capture_environment, "capture_snapshot", lambda **_: dict(blocked))
    output = tmp_path / "env.yaml"
    assert capture_environment.main(["--record-only", "--output", str(output)]) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["capture"]["mode"] == "record-only"
```
  (Add it inside the existing test module style; import `json` if missing.) Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_environment.py -q` → FAIL (unknown argument).
- [ ] **Step 4: Implement.** `--record-only`: set `snapshot["capture"]["mode"] = "record-only"` (otherwise `"gate"`), write the snapshot, return 0. Help text: "record the runtime (e.g. Kaggle) without enforcing the lock or the CUDA gate". Do not regenerate `configs/env.yaml` in this task unless the new packages make its package list stale — if so, rerun `uv run python scripts/capture_environment.py` and commit the refreshed file.
- [ ] **Step 5: Verify.** `.venv/Scripts/python.exe -c "import torch, torchvision, segmentation_models_pytorch as smp, sklearn; print(torch.__version__, torchvision.__version__, smp.__version__, sklearn.__version__, torch.cuda.is_available())"` prints versions and `True`; full suite green.
- [ ] **Step 6: Commit** `build: ajouter torchvision, smp et scikit-learn; mode --record-only pour Kaggle`.

---

### Task 2: Richer manifest + mass image records and mask fusion

**Files:**
- Modify: `src/cbis_ddsm/data/manifest.py`, `tests/test_manifest.py`
- Create: `src/cbis_ddsm/data/targets.py`, `tests/test_targets.py`

**Interfaces:**
- Consumes: `ManifestRow`, `read_manifest_csv(path, raw_dir)`; exclusions CSV written by `write_exclusions_csv` (columns `row_number,reason_code,patient_id,lesion_identity` where `lesion_identity = "patient|side|view|abnormality|lesion_type"`).
- Produces:
  - `ManifestRow` gains `subtlety: str`, `shape_or_type: str`, `margins_or_distribution: str` (appended at the **end** of `MANIFEST_FIELDS`). Source columns: `subtlety`; `mass shape` or `calc type`; `mass margins` or `calc distribution`.
  - `@dataclass(frozen=True) class ImageRecord: image_key: str; patient_id: str; laterality: str; view: str; split: str; full_image_path: Path; mask_paths: tuple[Path, ...]; abnormality_ids: tuple[str, ...]; pathologies: tuple[str, ...]; density: str; subtlety: tuple[str, ...]` with property `has_malignant -> bool`.
  - `select_masses(rows: Iterable[ManifestRow]) -> tuple[ManifestRow, ...]`
  - `read_excluded_identities(path: Path | str) -> frozenset[tuple[str, str, str, str, str]]`
  - `group_images(rows, excluded: frozenset[...] = frozenset()) -> tuple[tuple[ImageRecord, ...], tuple[str, ...]]` → (kept records sorted by `image_key`, sorted `image_key`s dropped because at least one lesion is excluded). `image_key = f"{patient_id}|{laterality}|{view}|{split}"`.
  - `fuse_masks(paths: Sequence[Path]) -> np.ndarray` → `uint8` array of 0/1, shape of the first mask; raises `ValueError` if shapes differ or `paths` is empty.

- [ ] **Step 1: Manifest columns — failing test.** In `tests/test_manifest.py`, extend `_write_case_csv` fields with `"subtlety", "mass shape", "mass margins"`, give `_case_row` defaults `"subtlety": "4", "mass shape": "IRREGULAR", "mass margins": "SPICULATED"`, and add:

```python
def test_manifest_keeps_subtlety_and_morphology_columns(tmp_path: Path) -> None:
    raw = _synthetic_raw(tmp_path)
    _write_case_csv(raw / "mass_case_description_train_set.csv", [_case_row()])
    row = build_manifest(raw)[0]
    assert (row.subtlety, row.shape_or_type, row.margins_or_distribution) == ("4", "IRREGULAR", "SPICULATED")
```
  Run → FAIL (`AttributeError`).
- [ ] **Step 2: Implement** in `_resolve_row`: `subtlety=_value(record, "subtlety")`, `shape_or_type=_value(record, "mass shape") or _value(record, "calc type")`, `margins_or_distribution=_value(record, "mass margins") or _value(record, "calc distribution")`; add the fields to `ManifestRow`, to the end of `MANIFEST_FIELDS`, and to `read_manifest_csv`. Existing tests that write a CSV without these columns must still pass (missing columns → `""`). Run the manifest tests → PASS.
- [ ] **Step 3: Targets — failing tests** in `tests/test_targets.py`:

```python
from pathlib import Path
import cv2, numpy as np, pytest
from cbis_ddsm.data.manifest import ManifestRow
from cbis_ddsm.data.targets import fuse_masks, group_images, read_excluded_identities, select_masses

def _mask(path: Path, box: tuple[int, int, int, int], shape=(20, 16), value=255) -> Path:
    pixels = np.zeros(shape, dtype=np.uint8)
    y0, y1, x0, x1 = box
    pixels[y0:y1, x0:x1] = value
    assert cv2.imwrite(str(path), pixels)
    return path

def _row(tmp_path, patient="P_1", view="CC", abn="1", pathology="BENIGN", lesion_type="mass", split="train", mask=None):
    return ManifestRow(patient, "LEFT", view, abn, lesion_type, pathology, "2", "4", split, "x.csv",
                       tmp_path / f"{patient}-{view}.jpg", tmp_path / "crop.jpg", mask or tmp_path / f"m{abn}.png",
                       "3", "OVAL", "CIRCUMSCRIBED")

def test_fuse_masks_is_pixelwise_union_after_threshold(tmp_path):
    a = _mask(tmp_path / "a.png", (2, 6, 2, 6))
    b = _mask(tmp_path / "b.png", (10, 14, 8, 12), value=249)
    fused = fuse_masks([a, b])
    assert fused.dtype == np.uint8 and fused.shape == (20, 16)
    assert fused.sum() == 32 and set(np.unique(fused)) == {0, 1}

def test_fuse_masks_rejects_shape_mismatch_and_empty_input(tmp_path):
    with pytest.raises(ValueError):
        fuse_masks([_mask(tmp_path / "a.png", (0, 2, 0, 2)), _mask(tmp_path / "b.png", (0, 2, 0, 2), shape=(10, 10))])
    with pytest.raises(ValueError):
        fuse_masks([])

def test_group_images_merges_lesions_of_one_image_only(tmp_path):
    rows = [_row(tmp_path, abn="2", pathology="MALIGNANT"), _row(tmp_path, abn="1"), _row(tmp_path, view="MLO", abn="1"),
            _row(tmp_path, abn="1", lesion_type="calc")]
    records, dropped = group_images(select_masses(rows))
    assert dropped == ()
    assert [r.image_key for r in records] == ["P_1|LEFT|CC|train", "P_1|LEFT|MLO|train"]
    assert records[0].abnormality_ids == ("1", "2") and records[0].has_malignant
    assert not records[1].has_malignant

def test_group_images_drops_whole_image_when_one_lesion_is_excluded(tmp_path):
    rows = [_row(tmp_path, abn="1"), _row(tmp_path, abn="2"), _row(tmp_path, view="MLO")]
    excluded = frozenset({("P_1", "LEFT", "CC", "2", "mass")})
    records, dropped = group_images(rows, excluded)
    assert dropped == ("P_1|LEFT|CC|train",)
    assert [r.view for r in records] == ["MLO"]

def test_read_excluded_identities_parses_audit_csv(tmp_path):
    path = tmp_path / "exclusions.csv"
    path.write_text("row_number,reason_code,patient_id,lesion_identity\n3,dimension_mismatch,P_1,P_1|LEFT|CC|2|mass\n", encoding="utf-8")
    assert read_excluded_identities(path) == frozenset({("P_1", "LEFT", "CC", "2", "mass")})
```
  Run → FAIL (module missing).
- [ ] **Step 4: Implement `targets.py`.** Sort lesions inside an image by `(abnormality_id as int when numeric, abnormality_id)`. `density` of an image = density of its first lesion. `fuse_masks` loads with `cv2.IMREAD_GRAYSCALE`, raises `ValueError` on unreadable file, returns `np.maximum.reduce([(m >= 128).astype(np.uint8) ...])`. Run → PASS; full suite green.
- [ ] **Step 5: Commit** `feat(data): colonnes de morphologie au manifeste; images de masses et fusion des masques`.

---

### Task 3: Patient-grouped stratified splits

**Files:**
- Create: `src/cbis_ddsm/data/splits.py`, `scripts/make_splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `ImageRecord`, `group_images`, `select_masses`, `read_excluded_identities`, `read_manifest_csv`.
- Produces:
  - `patient_labels(records: Iterable[ImageRecord]) -> dict[str, int]` — 1 if any image of the patient has a malignant lesion; only `split == "train"` records.
  - `assign_folds(labels: Mapping[str, int], n_folds: int = 5, seed: int = 0) -> dict[str, int]` — `sklearn.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)` over patients sorted by id (one sample per patient ⇒ grouping is exact).
  - `build_split(records, *, n_folds=5, val_fold=0, seed=0, manifest_sha256: str) -> dict` with keys `version` (=1), `seed`, `n_folds`, `val_fold`, `manifest_sha256`, `folds` (patient → fold), `partitions` = `{"train": [...], "val": [...], "test": [...]}` (sorted patient ids; test = patients of official test records).
  - `write_split_json(split: dict, path)` (sorted keys, indent 2, trailing newline) and `read_split_json(path) -> dict`.
  - `partition_of(split: dict) -> dict[str, str]` patient → `"train" | "val" | "test"`.
  - `sha256_file(path) -> str`.
  - CLI `scripts/make_splits.py --raw-dir data/raw --manifest data/interim/manifest.csv --exclusions data/interim/exclusions.csv --output data/splits/split_v1.json --n-folds 5 --val-fold 0 --seed 0`; refuses an output inside `--raw-dir`; prints patient counts per partition and the malignant share of train and val.

- [ ] **Step 1: Failing tests** (`tests/test_splits.py`) — build 40 synthetic `ImageRecord`s directly (no files needed): 30 train patients (every third malignant, each with CC and MLO images), 10 test patients.

```python
from pathlib import Path
from cbis_ddsm.data.targets import ImageRecord
from cbis_ddsm.data.splits import assign_folds, build_split, partition_of, patient_labels, read_split_json, write_split_json

def _records():
    out = []
    for i in range(40):
        split = "train" if i < 30 else "test"
        pathology = "MALIGNANT" if i % 3 == 0 else "BENIGN"
        for view in ("CC", "MLO"):
            out.append(ImageRecord(f"P_{i:03d}|LEFT|{view}|{split}", f"P_{i:03d}", "LEFT", view, split, Path("x.jpg"),
                                   (Path("m.png"),), ("1",), (pathology,), "2", ("3",)))
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
    write_split_json(split, a); write_split_json(build_split(_records(), seed=3, manifest_sha256="abc"), b)
    assert a.read_bytes() == b.read_bytes() and read_split_json(a) == split
```
  Add one CLI test in the style of `tests/test_data_clis.py` (subprocess, synthetic raw dir built with the helper from that file extended to 2 patients) asserting the JSON is written and that an output inside `--raw-dir` is refused with `"outside the raw directory"` in stderr. Run → FAIL.
- [ ] **Step 2: Implement** `splits.py` and the CLI (CLI reads manifest → `select_masses` → `read_excluded_identities` (if the exclusions file is absent, fail with a clear message telling to run `scripts/audit_data.py`) → `group_images` → `build_split` with `manifest_sha256=sha256_file(manifest)`). Run → PASS; full suite green.
- [ ] **Step 3: Commit** `feat(data): partitions par patiente stratifiées (split_v1.json)`.

---

### Task 4: Native-resolution patch extraction and re-projection

**Files:**
- Create: `src/cbis_ddsm/data/patches.py`, `scripts/extract_patches.py`, `tests/test_patches.py`

**Interfaces:**
- Consumes: `ImageRecord`, `fuse_masks`, `group_images`, `select_masses`, `read_excluded_identities`, `read_split_json`, `partition_of`.
- Produces:
  - `@dataclass(frozen=True) class PatchWindow: x0: int; y0: int; side: int` — square window in source-image pixels; may extend beyond the image (padding with 0).
  - `@dataclass(frozen=True) class PatchRecord: patch_id: str; image_key: str; patient_id: str; partition: str; kind: str; abnormality_id: str; x0: int; y0: int; side: int; out_size: int; source_height: int; source_width: int; image_file: str; mask_file: str; lesion_pixels: int` (`kind` ∈ `{"lesion", "background"}`, files relative to the patch root). `PATCH_INDEX_FIELDS` = the field names in that order.
  - `lesion_window(mask: np.ndarray, size: int, margin: float = 0.15) -> PatchWindow` — foreground = `mask >= 128`; bbox `[ymin, ymax]`, `[xmin, xmax]` inclusive; center `cy = (ymin + ymax + 1) // 2`, `cx = (xmin + xmax + 1) // 2`; `side = max(size, ceil(max(ymax - ymin + 1, xmax - xmin + 1) * (1 + 2 * margin)))`; `x0 = cx - side // 2`, `y0 = cy - side // 2`. Raises `ValueError` on an empty mask.
  - `background_windows(image: np.ndarray, mask: np.ndarray, size: int, count: int, rng: np.random.Generator, max_tries: int = 200) -> list[PatchWindow]` — windows fully inside the image, whose center pixel lies in breast tissue (Otsu threshold on a ×1/8 downsampled image, largest connected component) and that contain **no** lesion pixel.
  - `crop_window(array: np.ndarray, window: PatchWindow) -> np.ndarray` — zero-padded crop of `side×side`.
  - `extract(image, mask, window, out_size) -> tuple[np.ndarray, np.ndarray]` — crop then resize to `out_size` (`INTER_AREA` for image, `INTER_NEAREST` for mask) only when `side != out_size`; mask returned as 0/255 uint8.
  - `reproject(prediction: np.ndarray, record: PatchRecord) -> np.ndarray` — float array `(source_height, source_width)` with the prediction resized back to `side` (`INTER_LINEAR`) and pasted at `(x0, y0)`, clipped to the image; pixels outside the window are 0.
  - `write_patch_index(records, path)`, `read_patch_index(path) -> tuple[PatchRecord, ...]`.
  - CLI `scripts/extract_patches.py --raw-dir --manifest --exclusions --split --output-dir data/processed/patches_v1 --size 512 --background-per-image 1 --seed 0 [--limit-patients N] [--partitions train val]`: writes `images/<patch_id>.png`, `masks/<patch_id>.png`, `index.csv`; one lesion patch **per lesion** (window from that lesion's own mask, but **target = fused mask of the image**) + `background-per-image` background patches per image; `--limit-patients` keeps the first N patients (sorted) of each requested partition; refuses an output dir inside `--raw-dir`; prints counts per partition and kind.

- [ ] **Step 1: Failing tests** (`tests/test_patches.py`), all synthetic:

```python
import numpy as np
from cbis_ddsm.data.patches import PatchRecord, PatchWindow, background_windows, crop_window, extract, lesion_window, reproject

def _scene():
    image = np.zeros((600, 400), dtype=np.uint8); image[:, :300] = 120   # "breast" on the left
    mask = np.zeros_like(image); mask[200:260, 100:180] = 255             # 60 x 80 lesion
    image[mask > 0] = 230
    return image, mask

def test_lesion_window_centers_lesion_and_keeps_native_resolution():
    _, mask = _scene()
    w = lesion_window(mask, size=128)
    assert w.side == 128 and (w.x0 + w.side // 2, w.y0 + w.side // 2) == (140, 230)

def test_large_lesion_grows_the_window_instead_of_cutting_it():
    mask = np.zeros((600, 400), dtype=np.uint8); mask[100:400, 50:250] = 255
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
    _, msk = extract(image, mask, w, out_size=64)          # downscaled patch
    rec = PatchRecord("p", "k", "P", "train", "lesion", "1", w.x0, w.y0, w.side, 64, 600, 400, "i.png", "m.png", int((msk > 0).sum()))
    back = reproject((msk > 0).astype(np.float32), rec) >= 0.5
    inter = np.logical_and(back, mask > 0).sum(); union = np.logical_or(back, mask > 0).sum()
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
```
  Plus a CLI test: synthetic raw dir (2 train patients × 1 image with one lesion, 400×300 JPEGs), manifest via `build_manifest`, empty exclusions CSV, split via `build_split`; run the CLI with `--size 64 --background-per-image 1`; assert `index.csv` rows = 4, every listed PNG exists with shape (64, 64), and lesion patches have `lesion_pixels > 0`. Run → FAIL.
- [ ] **Step 2: Implement.** `patch_id = f"{patient_id}_{laterality}_{view}_{kind}_{abnormality_id or index}"` (filesystem-safe). Use `rng = np.random.default_rng(seed)` once per run, iterating images in sorted `image_key` order so outputs are deterministic. Run → PASS; full suite green.
- [ ] **Step 3: Commit** `feat(data): extraction d'imagettes à résolution native et reprojection`.

---

### Task 5: Dataset contract and synchronized transforms

**Files:**
- Create: `src/cbis_ddsm/data/transforms.py`, `src/cbis_ddsm/data/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `read_patch_index`, `PatchRecord`, `PATCH_INDEX_FIELDS`, `write_patch_index`.
- Produces:
  - `train_transforms() -> albumentations.Compose` — `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `RandomRotate90(p=0.5)`, `ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5)`, `RandomBrightnessContrast(0.15, 0.15, p=0.5)`. `eval_transforms() -> None`.
  - `class PatchDataset(torch.utils.data.Dataset)`: `__init__(self, root: Path | str, partition: str | None = None, transform=None, seed: int = 0, limit: int | None = None)`; `__getitem__(i) -> tuple[torch.Tensor, torch.Tensor]` = (`image float32 [1,H,W]` in `[0,1]`, `mask float32 [1,H,W]` in `{0,1}`); attribute `records: tuple[PatchRecord, ...]`. Augmentation randomness is reproducible: before applying `transform`, seed `random` and `numpy` with `hash((self.seed, self.epoch, i)) & 0xFFFFFFFF`; `set_epoch(epoch: int)` changes the stream.
  - `make_fake_patch_dir(root: Path, n: int = 20, size: int = 64, seed: int = 0, partitions: Sequence[str] = ("train", "val")) -> Path` — writes random ellipse lesions (image intensity 200 inside, noise background ~60) as PNGs + `index.csv` compatible with `PatchDataset`; alternates partitions.

- [ ] **Step 1: Failing tests** (`tests/test_dataset.py`):

```python
import numpy as np, torch
from cbis_ddsm.data.dataset import PatchDataset, make_fake_patch_dir
from cbis_ddsm.data.transforms import train_transforms

def test_dataset_contract_shapes_types_ranges(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=6, size=64)
    ds = PatchDataset(root, partition="train")
    image, mask = ds[0]
    assert image.shape == mask.shape == (1, 64, 64)
    assert image.dtype == mask.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0} and mask.sum() > 0

def test_partition_filter_and_limit(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=10, size=32)
    assert len(PatchDataset(root, partition="val")) == 5
    assert len(PatchDataset(root, partition="train", limit=2)) == 2

def test_augmentation_keeps_mask_on_the_bright_lesion(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    ds = PatchDataset(root, partition="train", transform=train_transforms(), seed=1)
    for epoch in range(3):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            image, mask = ds[i]
            inside = image[mask > 0.5]
            outside = image[mask < 0.5]
            assert inside.numel() > 0 and inside.mean() > outside.mean() + 0.2

def test_augmentation_is_reproducible_for_a_fixed_seed_and_epoch(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=4, size=64)
    a = PatchDataset(root, "train", train_transforms(), seed=5); b = PatchDataset(root, "train", train_transforms(), seed=5)
    a.set_epoch(2); b.set_epoch(2)
    assert torch.equal(a[1][0], b[1][0]) and torch.equal(a[1][1], b[1][1])
```
  Run → FAIL.
- [ ] **Step 2: Implement.** Load PNGs with `cv2.IMREAD_GRAYSCALE`; albumentations receives `image=uint8 HxW`, `mask=uint8 0/1 HxW`. If albumentations 1.4.24 warns about `ShiftScaleRotate` argument names, use the names it accepts without warnings in that version. Run → PASS; full suite green.
- [ ] **Step 3: Commit** `feat(data): Dataset d'imagettes et augmentations synchronisées`.

---

### Task 6: Model factory, loss and metrics

**Files:**
- Create: `src/cbis_ddsm/models/__init__.py`, `src/cbis_ddsm/models/factory.py`, `src/cbis_ddsm/training/__init__.py`, `src/cbis_ddsm/training/losses.py`, `src/cbis_ddsm/evaluation/__init__.py`, `src/cbis_ddsm/evaluation/metrics.py`
- Test: `tests/test_models.py`, `tests/test_losses_metrics.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_ARCHITECTURES = ("unet", "unetplusplus", "deeplabv3plus")`
  - `build_model(architecture: str = "unet", encoder: str = "resnet34", encoder_weights: str | None = "imagenet", in_channels: int = 1) -> torch.nn.Module` — smp model with `classes=1`, `activation=None` (returns logits); raises `ValueError` for unknown architectures.
  - `class BCEDiceLoss(torch.nn.Module)`: `__init__(self, bce_weight: float = 0.5, smooth: float = 1.0)`; `forward(logits, target) -> torch.Tensor` = `bce_weight * BCEWithLogits + (1 - bce_weight) * (1 - soft_dice)` with soft Dice computed per sample over `[C,H,W]` then averaged.
  - `dice_score(pred: np.ndarray, target: np.ndarray) -> float` and `iou_score(pred, target) -> float` on boolean-convertible arrays; **both empty ⇒ 1.0**, exactly one empty ⇒ 0.0.
  - `threshold_probabilities(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray[bool]`.

- [ ] **Step 1: Failing tests.**

```python
# tests/test_losses_metrics.py
import numpy as np, pytest, torch
from cbis_ddsm.evaluation.metrics import dice_score, iou_score
from cbis_ddsm.training.losses import BCEDiceLoss

def test_metrics_on_analytic_squares():
    a = np.zeros((10, 10), bool); a[0:4, 0:4] = True     # 16 px
    b = np.zeros((10, 10), bool); b[2:6, 2:6] = True     # 16 px, overlap 4
    assert dice_score(a, b) == pytest.approx(2 * 4 / 32)
    assert iou_score(a, b) == pytest.approx(4 / 28)

def test_empty_mask_rules():
    empty = np.zeros((5, 5), bool); full = np.ones((5, 5), bool)
    assert dice_score(empty, empty) == 1.0 and iou_score(empty, empty) == 1.0
    assert dice_score(full, empty) == 0.0 and dice_score(empty, full) == 0.0

def test_loss_is_lower_for_the_correct_prediction():
    target = torch.zeros(2, 1, 8, 8); target[:, :, 2:6, 2:6] = 1
    good = (target * 2 - 1) * 8.0; bad = -good
    loss = BCEDiceLoss()
    assert loss(good, target) < 0.05 < loss(bad, target)

def test_loss_backpropagates():
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    BCEDiceLoss()(logits, torch.ones(1, 1, 4, 4)).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
```

```python
# tests/test_models.py
import pytest, torch
from cbis_ddsm.models.factory import build_model

@pytest.mark.parametrize("arch", ["unet", "unetplusplus", "deeplabv3plus"])
def test_models_return_one_logit_map_of_input_size(arch):
    model = build_model(arch, encoder="resnet18", encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)

def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_model("transformer", encoder_weights=None)
```
  Tests must never download weights (`encoder_weights=None`). Run → FAIL.
- [ ] **Step 2: Implement.** Run → PASS; full suite green.
- [ ] **Step 3: Commit** `feat(models): fabrique smp, perte BCE+Dice et métriques Dice/IoU`.

---

### Task 7: Training configuration, trainer, overfit gate

**Files:**
- Create: `src/cbis_ddsm/training/config.py`, `src/cbis_ddsm/training/trainer.py`, `scripts/train.py`, `scripts/overfit_batch.py`, `configs/unet_smoke.yaml`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `PatchDataset`, `train_transforms`, `build_model`, `BCEDiceLoss`, `dice_score`.
- Produces:
  - `@dataclass(frozen=True) class TrainConfig`: `patch_root: str`, `output_root: str = "experiments"`, `run_name: str = "unet_smoke"`, `architecture: str = "unet"`, `encoder: str = "resnet34"`, `encoder_weights: str | None = "imagenet"`, `epochs: int = 2`, `batch_size: int = 8`, `accumulation_steps: int = 1`, `learning_rate: float = 3e-4`, `weight_decay: float = 1e-4`, `bce_weight: float = 0.5`, `amp: bool = True`, `num_workers: int = 2`, `seed: int = 0`, `device: str = "auto"`, `augment: bool = True`, `limit_train: int | None = None`, `limit_val: int | None = None`, `threshold: float = 0.5`.
  - `load_config(path, overrides: Mapping[str, object] | None = None) -> TrainConfig` (unknown keys ⇒ `ValueError`); `config_to_dict(cfg) -> dict`.
  - `seed_everything(seed: int) -> None` (random, numpy, torch, cuda; `torch.backends.cudnn.benchmark = False`).
  - `resolve_device(name: str) -> torch.device` (`auto` ⇒ cuda if available else cpu).
  - `@dataclass class RunResult: run_dir: Path; best_epoch: int; best_val_dice: float; history: list[dict]`.
  - `train(cfg: TrainConfig) -> RunResult` — creates `experiments/<run_name>_<UTC yyyymmdd-HHMMSS>/` with `config.yaml`, `metrics.jsonl` (one JSON line per epoch: `epoch, train_loss, val_loss, val_dice, lr, seconds`), `checkpoint_best.pt` (`{"model": state_dict, "config": dict, "epoch": int, "val_dice": float}`), `run.json` (`git_commit` via `git -c safe.directory=* rev-parse HEAD` or `null`, torch/cuda versions, device). AMP only on CUDA. Gradient accumulation divides the loss by `accumulation_steps`. Val Dice = mean per-patch `dice_score(prob >= threshold, mask)`.
  - `load_checkpoint(path, device="cpu") -> tuple[torch.nn.Module, dict]` — rebuilds the model from the saved config with `encoder_weights=None`.
  - `overfit_batch(cfg: TrainConfig, n: int = 8, steps: int = 300) -> float` — no augmentation, same `n` train patches every step, returns final Dice on those patches.
  - CLIs: `scripts/train.py --config configs/unet_smoke.yaml [--set key=value ...]`; `scripts/overfit_batch.py --config ... --n 8 --steps 300 --target-dice 0.95` (exit 1 with a clear message if the target is not reached).
  - `configs/unet_smoke.yaml`: `patch_root: data/processed/patches_v1`, `epochs: 2`, `batch_size: 4`, `accumulation_steps: 2`, `encoder: resnet34`, `encoder_weights: imagenet`, `amp: true`, `num_workers: 2`, `seed: 0`.

- [ ] **Step 1: Failing tests** (`tests/test_trainer.py`) on `make_fake_patch_dir` with `encoder="resnet18"`, `encoder_weights=None`, `device="cpu"`, `amp=False`, `num_workers=0`, size 32 or 64:

```python
import json, pytest, torch
from cbis_ddsm.data.dataset import make_fake_patch_dir
from cbis_ddsm.training.config import TrainConfig, load_config
from cbis_ddsm.training.trainer import load_checkpoint, overfit_batch, train

def _cfg(tmp_path, **kw):
    root = make_fake_patch_dir(tmp_path / "fake", n=12, size=64)
    base = dict(patch_root=str(root), output_root=str(tmp_path / "exp"), encoder="resnet18", encoder_weights=None,
                device="cpu", amp=False, num_workers=0, batch_size=4, epochs=1, augment=False)
    base.update(kw)
    return TrainConfig(**base)

def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"; path.write_text("patch_root: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)

def test_train_writes_a_complete_run_directory(tmp_path):
    result = train(_cfg(tmp_path))
    for name in ("config.yaml", "metrics.jsonl", "checkpoint_best.pt", "run.json"):
        assert (result.run_dir / name).is_file()
    lines = (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["epoch"] == 0 and 0.0 <= result.best_val_dice <= 1.0

def test_same_seed_gives_identical_first_epoch_loss(tmp_path):
    a = train(_cfg(tmp_path / "a", seed=3)).history[0]["train_loss"]
    b = train(_cfg(tmp_path / "b", seed=3)).history[0]["train_loss"]
    assert a == pytest.approx(b, rel=1e-6)

def test_checkpoint_reload_reproduces_predictions(tmp_path):
    result = train(_cfg(tmp_path))
    model, _ = load_checkpoint(result.run_dir / "checkpoint_best.pt")
    again, _ = load_checkpoint(result.run_dir / "checkpoint_best.pt")
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), again.eval()(x))

@pytest.mark.slow
def test_overfit_gate_on_synthetic_patches(tmp_path):
    assert overfit_batch(_cfg(tmp_path, learning_rate=1e-3), n=8, steps=150) > 0.9
```
  Run → FAIL.
- [ ] **Step 2: Implement.** Use `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` only when the device is CUDA and `amp` is true; `DataLoader(..., generator=torch.Generator().manual_seed(seed), worker_init_fn=...)`; call `dataset.set_epoch(epoch)` each epoch. Optimizer AdamW. Run the fast tests → PASS, then run `-m slow` once and report its runtime; full suite green.
- [ ] **Step 3: Commit** `feat(training): configuration, boucle d'entraînement reproductible et gate de surapprentissage`.

---

### Task 8: Patch inference, metrics table and overlay figures

**Files:**
- Create: `src/cbis_ddsm/inference/__init__.py`, `src/cbis_ddsm/inference/predict.py`, `src/cbis_ddsm/evaluation/overlays.py`, `scripts/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `PatchDataset`, `dice_score`, `iou_score`, `threshold_probabilities`, `PatchRecord`.
- Produces:
  - `predict_dataset(model, dataset, device, batch_size=8) -> Iterator[tuple[PatchRecord, np.ndarray, np.ndarray, np.ndarray]]` yielding `(record, image HxW float, mask HxW bool, probability HxW float)` in dataset order.
  - `evaluate_patches(run_dir: Path, patch_root: Path, partition: str = "val", threshold: float = 0.5, limit: int | None = None, n_figures: int = 8, device: str = "auto") -> Path` — writes `run_dir/predictions/<partition>_metrics.csv` (columns `patch_id, patient_id, kind, lesion_pixels, dice, iou, predicted_pixels`), `run_dir/predictions/<partition>_summary.json` (`n_patches`, `mean_dice_lesion`, `mean_iou_lesion`, `background_false_positive_rate` = share of background patches with ≥1 predicted pixel), and `n_figures` overlays in `run_dir/figures/`; returns the summary path.
  - `make_overlay(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, title: str) -> np.ndarray` — BGR uint8 panel of three tiles side by side: image, image + ground-truth contour (green), image + predicted contour (red), with the title drawn on top (OpenCV only, no matplotlib).
  - CLI `scripts/predict.py --run-dir experiments/<run> --patch-root data/processed/patches_v1 --partition val [--limit N] [--threshold 0.5]`.

- [ ] **Step 1: Failing tests.**

```python
import json, numpy as np
from cbis_ddsm.data.dataset import make_fake_patch_dir
from cbis_ddsm.evaluation.overlays import make_overlay
from cbis_ddsm.inference.predict import evaluate_patches
from cbis_ddsm.training.config import TrainConfig
from cbis_ddsm.training.trainer import train

def test_make_overlay_places_three_tiles_side_by_side():
    image = np.zeros((32, 32), np.float32); mask = np.zeros((32, 32), bool); mask[8:16, 8:16] = True
    panel = make_overlay(image, mask, mask, "demo")
    assert panel.dtype == np.uint8 and panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[1] == 3 * 32 and panel.shape[0] > 32   # three 32-px tiles + a title band

def test_evaluate_patches_writes_metrics_summary_and_figures(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = TrainConfig(patch_root=str(root), output_root=str(tmp_path / "exp"), encoder="resnet18", encoder_weights=None,
                      device="cpu", amp=False, num_workers=0, batch_size=4, epochs=1, augment=False)
    run = train(cfg).run_dir
    summary_path = evaluate_patches(run, root, partition="val", n_figures=2, device="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_patches"] == 4
    assert (run / "predictions" / "val_metrics.csv").is_file()
    assert len(list((run / "figures").glob("*.png"))) == 2
```
  Tiles keep the input size (no rescaling); the title band sits above them. Run → FAIL.
- [ ] **Step 2: Implement.** Run → PASS; full suite green.
- [ ] **Step 3: Commit** `feat(inference): prédiction sur imagettes, métriques par imagette et superpositions`.

---

### Task 9: Kaggle smoke entry point, local dry run on real data, documentation

**Files:**
- Create: `scripts/kaggle_smoke.py`, `docs/KAGGLE.md`, `tests/test_kaggle_smoke.py`
- Modify: `README.md` (Démarrage), `PLAN.md` (check completed items), `.gitignore` only if needed

**Interfaces:**
- Consumes: every CLI above (called as Python functions or subprocesses with the current interpreter).
- Produces:
  - `scripts/kaggle_smoke.py --raw-dir <path> --work-dir <path> [--limit-patients 30] [--size 512] [--epochs 2] [--overfit-steps 300] [--device auto] [--encoder-weights imagenet|none] [--skip-audit-if-present]` running, in order and logging each stage with its duration:
    1. `capture_environment.py --record-only --output <work>/env.yaml`
    2. `build_manifest.py` → `<work>/interim/manifest.csv`, `<work>/interim/acquisition_exclusions.csv`
    3. `audit_data.py` **restricted to mass rows** (add `--lesion-type mass` to `scripts/audit_data.py` in this task, with a test) → `<work>/interim/exclusions.csv`
    4. `make_splits.py` → `<work>/splits/split_v1.json`
    5. `extract_patches.py --limit-patients N --partitions train val` → `<work>/processed/patches_v1`
    6. `overfit_batch.py` with the smoke config (`--n 8`), recorded but **not fatal** for the rest of the smoke run; its Dice is printed in the final report
    7. `train.py` with `patch_root`, `output_root=<work>/experiments`, `epochs`, `encoder_weights` overrides
    8. `predict.py --partition val`
    9. writes `<work>/smoke_report.json` (stage → status, seconds, key numbers) and zips `<work>/experiments/<run>` into `<work>/smoke_run.zip` (checkpoint excluded when larger than 200 MB)
  - Exit code 0 only if stages 1–5, 7 and 8 succeeded.
  - `docs/KAGGLE.md` (French): create a notebook with GPU T4 and Internet ON; add the dataset `awsaf49/cbis-ddsm-breast-cancer-image-dataset`; get the code (`git clone` of the GitHub repo, branch name as parameter, or upload of the repo as a private Kaggle dataset); install only what Kaggle lacks: `pip install segmentation-models-pytorch==<pinned>` (do **not** `pip install -e .`, the project pins Python 3.12 and cu121 wheels); run `python scripts/kaggle_smoke.py --raw-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset --work-dir /kaggle/working/smoke --limit-patients 30`; download `smoke_run.zip` and `smoke_report.json`; list what "success" means (see Definition of Done below).

- [ ] **Step 1: Failing tests.** (a) `audit_data.py --lesion-type mass` only audits mass rows (synthetic manifest with one calc row whose files are missing ⇒ no exclusion reported). (b) `kaggle_smoke.py` end to end on a **synthetic raw directory** (4 train patients + 2 test patients, 1 image each, 256×192 JPEGs, real CSV layout under `csv/` and `jpeg/`) with `--size 64 --epochs 1 --overfit-steps 5 --encoder-weights none --device cpu --limit-patients 4`, marked `@pytest.mark.slow` and `@pytest.mark.integration`: exit code 0, `smoke_report.json` lists every stage, `smoke_run.zip` exists. Run → FAIL.
- [ ] **Step 2: Implement** the orchestrator (resolve all paths from arguments; honor `--encoder-weights none` by passing `encoder_weights=null`; make every stage re-runnable into an existing work dir). Run the fast suite and `-m "slow or integration"` → PASS.
- [ ] **Step 3: Local dry run on real data** (evidence, not a unit test): 
  `.venv/Scripts/python.exe scripts/kaggle_smoke.py --raw-dir data/raw --work-dir C:/Users/metzg/AppData/Local/Temp/cbis_smoke --limit-patients 6 --size 256 --epochs 1 --overfit-steps 100 --device cuda --encoder-weights none` (MX350 2 GB: batch 4, AMP on). The full-mass audit takes ~8 min on this laptop. Record in the task report: stage durations, patch counts, overfit Dice, val Dice, and open one overlay PNG to confirm image/mask alignment. If memory fails on the GPU, rerun with `--device cpu`.
- [ ] **Step 4: Docs.** Write `docs/KAGGLE.md`; update `README.md` Démarrage with the new commands; tick the completed items of `PLAN.md` stages 1–3 that are now true (do not tick "première référence" — that needs the real Kaggle training).
- [ ] **Step 5: Commit** `feat: point d'entrée du run de fumée Kaggle, audit limité aux masses et documentation`.

---

## Definition of Done (smoke run)

On Kaggle, one command produces: `env.yaml` (record-only), manifest + exclusions, `split_v1.json` with disjoint patients, lesion and background patches for the limited patients, an overfit Dice on 8 real patches (target > 0.95, reported), a trained checkpoint after ≥ 1 epoch with a decreasing train loss, `val_metrics.csv` + `val_summary.json`, and overlay figures in which the green ground-truth contour sits on the visible mass.

## Self-review notes

- Spec coverage: PLAN stage 2 (masses, patient audit, grouped validation, synchronized transforms, empty-mask rule) → Tasks 2, 3, 5, 6; stage 3 (U-Net, Kaggle with configurable paths and env capture, overfit gate, BCE+Dice with seed, overlays + validation report) → Tasks 1, 6, 7, 8, 9. Out of scope, on purpose: HD95/ASSD, per-lesion sensitivity, full-image inference, cross-validation runs, nnU-Net (stages 4–5).
- Names used across tasks: `ImageRecord`, `group_images`, `select_masses`, `read_excluded_identities`, `fuse_masks`, `build_split`, `read_split_json`, `partition_of`, `PatchRecord`, `read_patch_index`, `write_patch_index`, `PatchDataset`, `make_fake_patch_dir`, `train_transforms`, `build_model`, `BCEDiceLoss`, `dice_score`, `iou_score`, `threshold_probabilities`, `TrainConfig`, `load_config`, `train`, `load_checkpoint`, `overfit_batch`, `evaluate_patches`, `make_overlay` — each is defined in exactly one task's Produces block.
