import numpy as np
import pytest
import torch
from cbis_ddsm.evaluation.metrics import dice_score, iou_score
from cbis_ddsm.training.losses import BCEDiceLoss


def test_metrics_on_analytic_squares():
    a = np.zeros((10, 10), bool)
    a[0:4, 0:4] = True  # 16 px
    b = np.zeros((10, 10), bool)
    b[2:6, 2:6] = True  # 16 px, overlap 4
    assert dice_score(a, b) == pytest.approx(2 * 4 / 32)
    assert iou_score(a, b) == pytest.approx(4 / 28)


def test_empty_mask_rules():
    empty = np.zeros((5, 5), bool)
    full = np.ones((5, 5), bool)
    assert dice_score(empty, empty) == 1.0 and iou_score(empty, empty) == 1.0
    assert dice_score(full, empty) == 0.0 and dice_score(empty, full) == 0.0


def test_loss_is_lower_for_the_correct_prediction():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    good = (target * 2 - 1) * 8.0
    bad = -good
    loss = BCEDiceLoss()
    assert loss(good, target) < 0.05 < loss(bad, target)


def test_loss_backpropagates():
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    BCEDiceLoss()(logits, torch.ones(1, 1, 4, 4)).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
