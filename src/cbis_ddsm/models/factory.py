"""Model factory for segmentation architectures."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}
SUPPORTED_ARCHITECTURES = tuple(_ARCHITECTURES)


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
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )

    model_class = _ARCHITECTURES[architecture]
    return model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
