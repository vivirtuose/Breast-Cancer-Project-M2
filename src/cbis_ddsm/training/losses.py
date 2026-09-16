"""Training losses."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined BCE with logits + Dice loss for binary segmentation.

    The loss combines binary cross-entropy and Dice loss:
        loss = bce_weight * BCE + (1 - bce_weight) * (1 - Dice)
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        """Initialize the loss.

        Args:
            bce_weight: Weight for BCE term [0, 1].
            smooth: Smoothing constant for Dice computation.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Model output, shape (N, C, H, W).
            target: Binary target, shape (N, C, H, W).

        Returns:
            Scalar loss.
        """
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, target)

        # Soft Dice loss
        prob = torch.sigmoid(logits)
        dice = self._soft_dice(prob, target)

        # Combine
        loss = self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)
        return loss

    def _soft_dice(
        self, prob: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft Dice over spatial dims, averaged over batch.

        Args:
            prob: Probabilities from sigmoid, shape (N, C, H, W).
            target: Binary target, shape (N, C, H, W).

        Returns:
            Scalar Dice score.
        """
        # Sum over spatial dims (C, H, W) for each sample
        intersection = (prob * target).sum(dim=(1, 2, 3))
        prob_sum = prob.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        dice_per_sample = (
            2 * intersection + self.smooth
        ) / (prob_sum + target_sum + self.smooth)

        # Average over batch
        return dice_per_sample.mean()
