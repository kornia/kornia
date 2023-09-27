from functools import partial

import pytest
import torch

from kornia.contrib.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder, RepVggBlock
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.contrib.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.testing import BaseTester, assert_close


@pytest.mark.parametrize(
    'backbone_factory',
    (partial(ResNetD.from_config, 18), partial(ResNetD.from_config, 50), partial(PPHGNetV2.from_config, "L")),
)
def test_backbone(backbone_factory, device, dtype):
    backbone = backbone_factory().to(device, dtype)
    assert hasattr(backbone, "out_channels")
    assert len(backbone.out_channels) == 3

    N, C, H, W = 2, 3, 224, 256
    imgs = torch.randn(N, C, H, W, device=device, dtype=dtype)
    fmaps = backbone(imgs)

    assert len(fmaps) == 3
    downscale = 8
    for fmap, ch in zip(fmaps, backbone.out_channels):
        assert fmap.shape == (N, ch, H // downscale, W // downscale)
        downscale *= 2


def test_neck(device, dtype):
    N = 2
    in_channels = [64, 128, 256]
    sizes = [(32, 24), (16, 12), (8, 6)]
    hidden_dim = 64
    neck = HybridEncoder(in_channels, hidden_dim, 128).to(device, dtype)
    fmaps = [torch.randn(N, ch_in, h, w, device=device, dtype=dtype) for ch_in, (h, w) in zip(in_channels, sizes)]

    outs = neck(fmaps)
    assert len(outs) == len(fmaps)
    for out, (h, w) in zip(outs, sizes):
        assert out.shape == (N, hidden_dim, h, w)


def test_head(device, dtype):
    N = 2
    in_channels = [32, 64, 128]
    sizes = [(32, 24), (16, 12), (8, 6)]
    num_classes = 5
    num_queries = 10
    decoder = RTDETRHead(num_classes, 32, num_queries, in_channels, 2).to(device, dtype).eval()
    fmaps = [torch.randn(N, ch_in, h, w, device=device, dtype=dtype) for ch_in, (h, w) in zip(in_channels, sizes)]

    logits, boxes = decoder(fmaps)
    assert logits.shape == (N, num_queries, num_classes)
    assert boxes.shape == (N, num_queries, 4)


def test_regvgg_optimize_for_deployment(device, dtype):
    module = RepVggBlock(64, 64).to(device, dtype).eval()
    x = torch.randn(2, 64, 9, 9, device=device, dtype=dtype)

    expected = module(x)
    module.optimize_for_deployment()
    actual = module(x)
    assert_close(actual, expected)


class TestRTDETR(BaseTester):
    @pytest.mark.parametrize("variant", ("resnet18d", "resnet34d", "resnet50d", "resnet101d", "hgnetv2_l", "hgnetv2_x"))
    def test_smoke(self, variant, device, dtype):
        model = RTDETR.from_config(RTDETRConfig(variant, 10)).to(device, dtype).eval()
        images = torch.randn(2, 3, 224, 256, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, tuple)
        assert len(out) == 2

    @pytest.mark.parametrize("shape", ((1, 3, 96, 128), (2, 3, 224, 256)))
    def test_cardinality(self, shape, device, dtype):
        num_classes = 10
        num_queries = 10
        config = RTDETRConfig("resnet50d", num_classes, head_num_queries=num_queries)
        model = RTDETR.from_config(config).to(device, dtype).eval()

        images = torch.randn(shape, device=device, dtype=dtype)
        logits, boxes = model(images)

        assert logits.shape == (shape[0], num_queries, num_classes)
        assert boxes.shape == (shape[0], num_queries, 4)

    @pytest.mark.skip("Unnecessary")
    def test_exception(self):
        ...

    @pytest.mark.skip("Unnecessary")
    def test_gradcheck(self):
        ...

    @pytest.mark.skip("Unnecessary")
    def test_module(self):
        ...

    @pytest.mark.skip("Needs more investigation")
    @pytest.mark.parametrize("variant", ("resnet50d", "hgnetv2_l"))
    def test_dynamo(self, variant, device, dtype, torch_optimizer):
        # NOTE: This test passes on Mac M1 CPU, PyTorch 2.0.0,
        # but fails on GitHub Actions Ubuntu-latest CPU, PyTorch 2.0.0.
        # Perhaps random weights cause outputs to be much more different?
        # Using pre-trained weights might see a smaller difference.
        model = RTDETR.from_config(RTDETRConfig(variant, 10, head_num_queries=10)).to(device, dtype).eval()
        model_optimized = torch_optimizer(model)

        img = torch.rand(1, 3, 224, 256, device=device, dtype=dtype)
        expected = model(img)
        actual = model_optimized(img)

        self.assert_close(actual, expected)
