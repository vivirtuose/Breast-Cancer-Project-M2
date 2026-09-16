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


def _as_boolean_pair(
    pred: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate boolean pair with empty-mask rules.

    Args:
        pred: Predicted binary segmentation.
        target: Target binary segmentation.

    Returns:
        Tuple of (pred, target) as boolean arrays.

    Raises:
        ValueError: If shapes differ.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

    return pred, target


def _apply_empty_mask_rule(pred_empty: bool, target_empty: bool) -> float | None:
    """Apply empty-mask rules to metric computation.

    Returns:
        1.0 if both empty, 0.0 if exactly one empty, None if both non-empty.
    """
    if not pred_empty and not target_empty:
        return None  # Both non-empty, proceed with computation
    if pred_empty and target_empty:
        return 1.0  # Both empty
    return 0.0  # Exactly one empty


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
    pred, target = _as_boolean_pair(pred, target)

    empty_result = _apply_empty_mask_rule(not pred.any(), not target.any())
    if empty_result is not None:
        return empty_result

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
    pred, target = _as_boolean_pair(pred, target)

    empty_result = _apply_empty_mask_rule(not pred.any(), not target.any())
    if empty_result is not None:
        return empty_result

    # Compute IoU
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = intersection / union
    return float(iou)
