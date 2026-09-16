from __future__ import annotations

import numpy as np
import torch

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
