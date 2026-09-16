"""Model factory for segmentation architectures."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

SUPPORTED_ARCHITECTURES = ("unet", "unetplusplus", "deeplabv3plus")


def build_model(
    architecture: str = "unet",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 1,
) -> torch.nn.Module:
    """Build a segmentation model from segmentation-models-pytorch.

    Args:
        architecture: One of SUPPORTED_ARCHITECTURES.
        encoder: Encoder name (e.g. "resnet34", "resnet18").
        encoder_weights: Pretrained weights or None for random init.
        in_channels: Number of input channels.

    Returns:
        A torch model that outputs logits (no activation).

    Raises:
        ValueError: If architecture is not supported.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    elif architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    elif architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
