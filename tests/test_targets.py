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
