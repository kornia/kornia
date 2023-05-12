import pytest
import torch

from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_decoder import RTDETRDecoder


@pytest.mark.parametrize('shape', ((1, 3, 224, 224), (2, 3, 256, 256)))
def test_backbone_fmaps(shape, device, dtype):
    backbone = ResNetD([2, 3, 3, 4]).to(device, dtype)
    imgs = torch.randn(shape, device=device, dtype=dtype)
    fmaps = backbone(imgs)

    assert len(fmaps) == 4
    downscale = 4
    for fmap in fmaps:
        assert fmap.shape[0] == shape[0]
        assert fmap.shape[2] == shape[2] // downscale
        assert fmap.shape[3] == shape[3] // downscale
        downscale *= 2


@pytest.mark.parametrize("batch_size", (1, 2))
def test_neck_fmaps(batch_size, device, dtype):
    in_channels = [64, 128, 256]
    sizes = [32, 16, 8]
    neck = HybridEncoder(in_channels, 64).to(device, dtype)
    fmaps = []
    for ch_in, sz in zip(in_channels, sizes):
        fmaps.append(torch.randn(batch_size, ch_in, sz, sz, device=device, dtype=dtype))

    outs = neck(fmaps)
    assert len(outs) == len(fmaps)
    for out, fmap in zip(outs, fmaps):
        assert out.shape[0] == fmap.shape[0]
        assert out.shape[2:] == fmap.shape[2:]


def test_rtdetr_decoder(device, dtype):
    in_channels = [32, 64, 128]
    sizes = [32, 16, 8]
    decoder = RTDETRDecoder(4, 32, 10, in_channels, 4, 8, 6).to(device, dtype)
    fmaps = [torch.randn(2, ch_in, sz, sz, device=device, dtype=dtype) for ch_in, sz in zip(in_channels, sizes)]

    bboxes, logits = decoder(fmaps)
