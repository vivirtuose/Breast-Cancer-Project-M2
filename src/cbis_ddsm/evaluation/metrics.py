"""Evaluation metrics for binary segmentation."""
from __future__ import annotations

import numpy as np


def threshold_probabilities(
    prob: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Threshold probabilities to binary predictions.

    Args:
        prob: Probability array of any dtype.
        threshold: Probability threshold.

    Returns:
        Boolean array.
    """
    return prob >= threshold


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice score (F1) between binary arrays.

    Args:
        pred: Predicted binary segmentation.
        target: Target binary segmentation.

    Returns:
        Dice score in [0, 1].
        - Both empty: 1.0 (perfect)
        - Exactly one empty: 0.0 (no overlap)
        - Otherwise: 2 * |intersection| / (|pred| + |target|)

    Raises:
        ValueError: If shapes differ.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    # Both empty
    if not pred.any() and not target.any():
        return 1.0

    # Exactly one empty
    if not pred.any() or not target.any():
        return 0.0

    # Compute Dice
    intersection = (pred & target).sum()
    dice = 2 * intersection / (pred.sum() + target.sum())
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard) score.

    Args:
        pred: Predicted binary segmentation.
        target: Target binary segmentation.

    Returns:
        IoU score in [0, 1].
        - Both empty: 1.0 (perfect)
        - Exactly one empty: 0.0 (no overlap)
        - Otherwise: |intersection| / |union|

    Raises:
        ValueError: If shapes differ.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    # Both empty
    if not pred.any() and not target.any():
        return 1.0

    # Exactly one empty
    if not pred.any() or not target.any():
        return 0.0

    # Compute IoU
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = intersection / union
    return float(iou)
