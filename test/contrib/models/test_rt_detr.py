import pytest
import torch

from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD


@pytest.mark.parametrize('shape', ((1, 3, 224, 224), (2, 3, 256, 256)))
def test_backbone_fmaps(shape, device, dtype):
    backbone = ResNetD([2, 3, 3, 4])
    imgs = torch.randn(shape, device=device, dtype=dtype)
    fmaps = backbone(imgs)

    assert len(fmaps) == 4
    downscale = 4
    for fmap in fmaps:
        assert fmap.shape[0] == shape[0]
        assert fmap.shape[2] == shape[2] // downscale
        assert fmap.shape[3] == shape[3] // downscale
        downscale *= 2
