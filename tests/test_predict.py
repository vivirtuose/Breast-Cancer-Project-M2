from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cbis_ddsm.data.dataset import make_fake_patch_dir, PatchDataset
from cbis_ddsm.data.patches import PatchRecord, write_patch_index
from cbis_ddsm.evaluation.overlays import make_overlay
from cbis_ddsm.inference.predict import evaluate_patches, predict_dataset
from cbis_ddsm.training.config import TrainConfig
from cbis_ddsm.training.trainer import load_checkpoint, train


ROOT = Path(__file__).resolve().parents[1]
PREDICT_CLI = ROOT / "scripts" / "predict.py"


def _run(script: Path, *arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _cfg(tmp_path, patch_root, **kw):
    base = dict(
        patch_root=str(patch_root), output_root=str(tmp_path / "exp"), encoder="resnet18",
        encoder_weights=None, device="cpu", amp=False, num_workers=0, batch_size=4, epochs=1, augment=False,
    )
    base.update(kw)
    return TrainConfig(**base)


def _add_background_patches(root: Path, n: int, size: int, seed: int = 1) -> None:
    """Append `n` background patches (empty mask, no lesion) to a fake patch dir, split train/val."""
    from cbis_ddsm.data.patches import read_patch_index

    records = list(read_patch_index(root / "index.csv"))
    rng = np.random.default_rng(seed)
    images_dir, masks_dir = root / "images", root / "masks"
    for index in range(n):
        patch_id = f"bg_{index:04d}"
        image = rng.integers(50, 71, size=(size, size), endpoint=False).astype(np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        image_file, mask_file = f"images/{patch_id}.png", f"masks/{patch_id}.png"
        cv2.imwrite(str(images_dir / f"{patch_id}.png"), image)
        cv2.imwrite(str(masks_dir / f"{patch_id}.png"), mask)
        partition = "train" if index % 2 == 0 else "val"
        records.append(PatchRecord(
            patch_id=patch_id, image_key=patch_id, patient_id=f"BG_{index:04d}", partition=partition,
            kind="background", abnormality_id="", x0=0, y0=0, side=size, out_size=size,
            source_height=size, source_width=size, image_file=image_file, mask_file=mask_file, lesion_pixels=0,
        ))
    write_patch_index(records, root / "index.csv")


def test_make_overlay_places_three_tiles_side_by_side():
    image = np.zeros((32, 32), np.float32)
    mask = np.zeros((32, 32), bool)
    mask[8:16, 8:16] = True
    panel = make_overlay(image, mask, mask, "demo")
    assert panel.dtype == np.uint8 and panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[1] == 3 * 32 and panel.shape[0] > 32  # three 32-px tiles + a title band


def test_make_overlay_accepts_uint8_image_and_all_zero_masks():
    image = (np.random.default_rng(0).integers(0, 255, size=(16, 16))).astype(np.uint8)
    empty = np.zeros((16, 16), bool)
    panel = make_overlay(image, empty, empty, "empty")
    assert panel.dtype == np.uint8 and panel.shape == (16 + panel.shape[0] - 16, 3 * 16, 3)


def test_evaluate_patches_writes_metrics_summary_and_figures(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir
    summary_path = evaluate_patches(run, root, partition="val", n_figures=2, device="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_patches"] == 4
    assert (run / "predictions" / "val_metrics.csv").is_file()
    assert len(list((run / "figures").glob("*.png"))) == 2


def test_evaluate_patches_summary_has_exact_keys_and_no_background_is_null(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir
    summary_path = evaluate_patches(run, root, partition="val", n_figures=1, device="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary.keys()) == {
        "n_patches", "mean_dice_lesion", "mean_iou_lesion", "background_false_positive_rate",
    }
    assert summary["background_false_positive_rate"] is None
    assert 0.0 <= summary["mean_dice_lesion"] <= 1.0
    assert 0.0 <= summary["mean_iou_lesion"] <= 1.0


def test_evaluate_patches_metrics_csv_has_exact_columns_in_order(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir
    evaluate_patches(run, root, partition="val", n_figures=0, device="cpu")
    with (run / "predictions" / "val_metrics.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["patch_id", "patient_id", "kind", "lesion_pixels", "dice", "iou", "predicted_pixels"]


def test_evaluate_patches_computes_background_false_positive_rate(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    _add_background_patches(root, n=6, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir
    summary_path = evaluate_patches(run, root, partition="val", n_figures=0, device="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["background_false_positive_rate"] is not None
    assert 0.0 <= summary["background_false_positive_rate"] <= 1.0

    with (run / "predictions" / "val_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    background_rows = [row for row in rows if row["kind"] == "background"]
    assert len(background_rows) == 3  # half of 6, alternating partitions
    for row in background_rows:
        assert row["lesion_pixels"] == "0"


def test_predict_dataset_yields_records_in_dataset_order(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir
    model, _ = load_checkpoint(run / "checkpoint_best.pt", device="cpu")
    dataset = PatchDataset(root, partition="val")
    outputs = list(predict_dataset(model, dataset, device="cpu", batch_size=4))
    assert [record.patch_id for record, _, _, _ in outputs] == [r.patch_id for r in dataset.records]
    for record, image, mask, probability in outputs:
        assert image.shape == (64, 64) and image.dtype == np.float32
        assert mask.shape == (64, 64) and mask.dtype == bool
        assert probability.shape == (64, 64) and probability.dtype == np.float32


def test_predict_cli_writes_summary_and_figures(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    cfg = _cfg(tmp_path, root)
    run = train(cfg).run_dir

    result = _run(
        PREDICT_CLI, "--run-dir", run, "--patch-root", root, "--partition", "val", "--n-figures", "1",
    )
    assert result.returncode == 0, result.stderr
    assert (run / "predictions" / "val_summary.json").is_file()
    assert len(list((run / "figures").glob("*.png"))) == 1


def test_predict_cli_reports_missing_checkpoint(tmp_path):
    root = make_fake_patch_dir(tmp_path / "fake", n=8, size=64)
    run_dir = tmp_path / "no_run"
    run_dir.mkdir()
    result = _run(PREDICT_CLI, "--run-dir", run_dir, "--patch-root", root)
    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()
