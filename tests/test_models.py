import pytest
import torch
from cbis_ddsm.models.factory import build_model


@pytest.mark.parametrize("arch", ["unet", "unetplusplus", "deeplabv3plus"])
def test_models_return_one_logit_map_of_input_size(arch):
    model = build_model(arch, encoder="resnet18", encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_model("transformer", encoder_weights=None)
