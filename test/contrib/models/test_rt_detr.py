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


@pytest.mark.parametrize('backbone_factory', (partial(ResNetD.from_config, 18), partial(PPHGNetV2.from_config, "L")))
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


def test_rtdetr_head(device, dtype):
    N = 2
    in_channels = [32, 64, 128]
    sizes = [(32, 24), (16, 12), (8, 6)]
    num_classes = 4
    num_queries = 10
    decoder = RTDETRHead(num_classes, 32, num_queries, in_channels, 4, 8, 6).to(device, dtype)
    fmaps = [torch.randn(N, ch_in, h, w, device=device, dtype=dtype) for ch_in, (h, w) in zip(in_channels, sizes)]

    bboxes, logits = decoder(fmaps)
    assert bboxes.shape == (N, num_queries, 4)
    assert logits.shape == (N, num_queries, num_classes)


class TestRTDETR(BaseTester):
    @pytest.mark.parametrize("variant", ("resnet50", "hgnetv2_l"))
    def test_smoke(self, variant, device, dtype):
        model = RTDETR.from_config(RTDETRConfig(variant, 80)).to(device, dtype)
        images = torch.randn(2, 3, 224, 256, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, DetectionResults)

    @pytest.mark.parametrize("shape", ((1, 3, 96, 128), (2, 3, 224, 256)))
    def test_cardinality(self, shape, device, dtype):
        num_queries = 10
        model = RTDETR.from_config(RTDETRConfig("resnet50", 10, head_num_queries=num_queries)).to(device, dtype)
        images = torch.randn(shape, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, DetectionResults)
        assert out.labels.shape == (shape[0], num_queries)
        assert out.scores.shape == (shape[0], num_queries)
        assert out.bounding_boxes.shape == (shape[0], num_queries, 4)

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
        model = RTDETR.from_config(RTDETRConfig('resnet50', 10, head_num_queries=10)).to(device, dtype)
        model_optimized = torch_optimizer(model)

        img = torch.rand(1, 3, 224, 256, device=device, dtype=dtype)
        expected = model(img)
        actual = model_optimized(img)

        self.assert_close(expected.labels, actual.labels)
        self.assert_close(expected.scores, actual.scores, low_tolerance=True)
        self.assert_close(expected.bounding_boxes, actual.bounding_boxes, low_tolerance=True)
