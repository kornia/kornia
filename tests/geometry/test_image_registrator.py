import pytest
import torch
from kornia.geometry.transform import ImageRegistrator


def test_shape_mismatch_enabled():
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 32, 32)

    registrator = ImageRegistrator("similarity", allow_shape_mismatch=True)
    output = registrator.register(img1, img2)

    assert output is not None
    assert isinstance(output, torch.Tensor)


def test_shape_mismatch_disabled():
    img1 = torch.rand(1, 1, 64, 64)
    img2 = torch.rand(1, 1, 32, 32)

    registrator = ImageRegistrator("similarity", allow_shape_mismatch=False)

    with pytest.raises(ValueError):
        registrator.register(img1, img2)