from functools import partial

import pytest
import torch

from kornia.contrib.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig


@pytest.mark.parametrize('shape', ((1, 3, 224, 224), (2, 3, 256, 256)))
@pytest.mark.parametrize('backbone_factory', (partial(ResNetD.from_config, 18), partial(PPHGNetV2.from_config, "L")))
def test_backbone(backbone_factory, shape, device, dtype):
    backbone = backbone_factory().to(device, dtype)
    imgs = torch.randn(shape, device=device, dtype=dtype)
    fmaps = backbone(imgs)

    assert len(fmaps) == 3
    downscale = 8
    for fmap in fmaps:
        assert fmap.shape[0] == shape[0]
        assert fmap.shape[2] == shape[2] // downscale
        assert fmap.shape[3] == shape[3] // downscale
        downscale *= 2


@pytest.mark.parametrize("batch_size", (1, 2))
def test_neck(batch_size, device, dtype):
    in_channels = [64, 128, 256]
    sizes = [32, 16, 8]
    hidden_dim = 64
    neck = HybridEncoder(in_channels, hidden_dim, 128).to(device, dtype)
    fmaps = []
    for ch_in, sz in zip(in_channels, sizes):
        fmaps.append(torch.randn(batch_size, ch_in, sz, sz, device=device, dtype=dtype))

    outs = neck(fmaps)
    assert len(outs) == len(fmaps)
    for out, fmap in zip(outs, fmaps):
        assert out.shape[0] == fmap.shape[0]
        assert out.shape[1] == hidden_dim
        assert out.shape[2:] == fmap.shape[2:]


def test_rtdetr_head(device, dtype):
    in_channels = [32, 64, 128]
    sizes = [32, 16, 8]
    decoder = RTDETRHead(4, 32, 10, in_channels, 4, 8, 6).to(device, dtype)
    fmaps = [torch.randn(2, ch_in, sz, sz, device=device, dtype=dtype) for ch_in, sz in zip(in_channels, sizes)]

    bboxes, logits = decoder(fmaps)


@pytest.mark.parametrize("variant", ("r50", "l"))
def test_rtdetr(variant, device, dtype):
    model = RTDETR.from_config(RTDETRConfig(variant)).to(device, dtype)
    images = torch.randn(2, 3, 256, 256, device=device, dtype=dtype)
    model(images)
