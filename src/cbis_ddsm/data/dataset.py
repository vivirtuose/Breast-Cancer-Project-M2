"""PyTorch dataset over extracted patches, and a synthetic patch directory for tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .patches import PatchRecord, read_patch_index, write_patch_index


class PatchDataset(Dataset):
    """Loads (image, mask) patch pairs written by `extract_image_patches`/`write_patch_index`.

    `image_file`/`mask_file` in each `PatchRecord` are POSIX paths relative to `root`. Masks are
    stored as 0/255 PNGs and binarized to {0, 1} on load. When `transform` is given, it must be
    an albumentations pipeline applied identically to image and mask; the augmentation stream for
    item `i` is reseeded deterministically from `(seed, epoch, i)` so a run is reproducible for a
    fixed seed and epoch and changes when `set_epoch` moves to a new epoch.
    """

    def __init__(
        self,
        root: Path | str,
        partition: str | None = None,
        transform=None,
        seed: int = 0,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        records = read_patch_index(self.root / "index.csv")
        if partition is not None:
            records = tuple(record for record in records if record.partition == partition)
        if limit is not None:
            records = records[:limit]
        self.records: tuple[PatchRecord, ...] = tuple(records)
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Move the reproducible augmentation stream to a new epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(str(self.root / record.image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unreadable patch image: {record.image_file}")
        mask_raw = cv2.imread(str(self.root / record.mask_file), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise ValueError(f"unreadable patch mask: {record.mask_file}")
        mask = (mask_raw >= 128).astype(np.uint8)

        if self.transform is not None:
            item_seed = hash((self.seed, self.epoch, index)) & 0xFFFFFFFF
            random.seed(item_seed)
            np.random.seed(item_seed)
            # albumentations 1.4.24 draws transform params from generators owned by each
            # BasicTransform/Compose instance (`self.py_random`, `self.random_generator`), not
            # from the global `random`/`numpy.random` state; re-seeding those instance
            # generators is what actually makes the augmentation stream reproducible.
            set_transform_seed = getattr(self.transform, "set_random_seed", None)
            if set_transform_seed is not None:
                set_transform_seed(item_seed)
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def make_fake_patch_dir(
    root: Path,
    n: int = 20,
    size: int = 64,
    seed: int = 0,
    partitions: Sequence[str] = ("train", "val"),
    kinds: Sequence[str] = ("lesion",),
) -> Path:
    """Write `n` synthetic patches (PNGs + `index.csv`) usable by `PatchDataset`.

    Each patch is a noisy background (~60 grey levels); a "lesion" patch also carries one bright
    ellipse (grey level 200) and the matching mask, while a "background" patch keeps an empty
    mask (`lesion_pixels=0`), as `extract_image_patches` writes for a lesion-free window. Rows
    cycle independently over `partitions` and `kinds` in order (row 0 -> partitions[0]/kinds[0],
    row 1 -> partitions[1]/kinds[1], ...), so coprime lengths mix both axes.
    """
    root = Path(root)
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    min_axis = max(3, size // 8)
    max_axis = max(min_axis, size // 4)

    records: list[PatchRecord] = []
    for index in range(n):
        patch_id = f"fake_{index:04d}"
        image = rng.integers(50, 71, size=(size, size), endpoint=False).astype(np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        axis_x = int(rng.integers(min_axis, max_axis + 1))
        axis_y = int(rng.integers(min_axis, max_axis + 1))
        margin = min(max(axis_x, axis_y) + 2, max(1, size // 2 - 1))
        center_x = int(rng.integers(margin, size - margin)) if size - margin > margin else size // 2
        center_y = int(rng.integers(margin, size - margin)) if size - margin > margin else size // 2

        kind = kinds[index % len(kinds)]
        if kind == "lesion":
            cv2.ellipse(image, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 200, -1)
            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

        image_file, mask_file = f"images/{patch_id}.png", f"masks/{patch_id}.png"
        cv2.imwrite(str(images_dir / f"{patch_id}.png"), image)
        cv2.imwrite(str(masks_dir / f"{patch_id}.png"), mask)

        records.append(PatchRecord(
            patch_id=patch_id, image_key=patch_id, patient_id=f"FAKE_{index:04d}",
            partition=partitions[index % len(partitions)], kind=kind, abnormality_id="1" if kind == "lesion" else "",
            x0=0, y0=0, side=size, out_size=size, source_height=size, source_width=size,
            image_file=image_file, mask_file=mask_file, lesion_pixels=int((mask == 255).sum()),
        ))

    write_patch_index(records, root / "index.csv")
    return root
