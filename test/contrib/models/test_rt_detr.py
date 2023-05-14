from functools import partial

import pytest
import torch

from kornia.contrib.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.contrib.models.structures import DetectionResults
from kornia.testing import BaseTester


@pytest.mark.parametrize('shape', ((1, 3, 224, 224), (2, 3, 256, 256)))
@pytest.mark.parametrize('backbone_factory', (partial(ResNetD.from_config, 18), partial(PPHGNetV2.from_config, "L")))
def test_backbone(backbone_factory, shape, device, dtype):
    backbone = backbone_factory().to(device, dtype)
    assert hasattr(backbone, "out_channels")
    assert len(backbone.out_channels) == 3

    imgs = torch.randn(shape, device=device, dtype=dtype)
    fmaps = backbone(imgs)

    assert len(fmaps) == 3
    downscale = 8
    N, _, H, W = shape
    for fmap, ch in zip(fmaps, backbone.out_channels):
        assert fmap.shape == (N, ch, H // downscale, W // downscale)
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


class TestRTDETR(BaseTester):
    @pytest.mark.parametrize("variant", ("resnet50", "hgnetv2_l"))
    def test_smoke(self, variant, device, dtype):
        model = RTDETR.from_config(RTDETRConfig(variant, 80)).to(device, dtype)
        images = torch.randn(2, 3, 256, 256, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, DetectionResults)

    @pytest.mark.parametrize("shape", ((1, 3, 128, 128), (2, 3, 256, 256)))
    def test_cardinality(self, shape, device, dtype):
        model = RTDETR.from_config(RTDETRConfig("resnet50", 80)).to(device, dtype)
        images = torch.randn(shape, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, DetectionResults)
        assert out.labels.shape[0] == shape[0]
        assert out.scores.shape[0] == shape[0]
        assert out.bboxes.shape[0] == shape[0]

    @pytest.mark.skip("Unnecessary")
    def test_exception(self):
        ...

    @pytest.mark.skip("Unnecessary")
    def test_gradcheck(self):
        pass

    @pytest.mark.skip("Unnecessary")
    def test_module(self):
        pass

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = RTDETR.from_config(RTDETRConfig('resnet50', 80)).to(device, dtype)
        model_optimized = torch_optimizer(model)

        img = torch.rand(1, 3, 128, 128, device=device, dtype=dtype)
        expected = model(img)
        actual = model_optimized(img)

        self.assert_close(expected.labels, actual.labels)
        self.assert_close(expected.scores, actual.scores)
        self.assert_close(expected.bboxes, actual.bboxes)

    def test_correctness(self, device, dtype):
        pytest.skip("Not implemented")
