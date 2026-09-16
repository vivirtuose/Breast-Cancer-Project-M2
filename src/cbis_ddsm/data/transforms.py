"""Synchronized image/mask augmentations for patch training."""

from __future__ import annotations

import os

# Silence albumentations' import-time "new version available" network check: it emits a
# UserWarning that would make the test suite non-deterministic and non-warning-free. Must be
# set before the first `import albumentations` in the process.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2


def train_transforms() -> A.Compose:
    """Geometric + photometric augmentations applied identically to image and mask.

    `ShiftScaleRotate` is deprecated in albumentations 1.4.24 (it emits a DeprecationWarning
    on every call, as do its `value`/`mask_value` arguments), so the shift/scale/rotate step
    is expressed as the equivalent `Affine` transform instead, using the `fill`/`fill_mask`
    argument names that 1.4.24 accepts without warning.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
    ])


def eval_transforms() -> None:
    """No augmentation at evaluation/inference time."""
    return None
