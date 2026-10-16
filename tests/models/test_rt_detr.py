# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import partial

import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
from kornia.models.rt_detr.architecture.hybrid_encoder import HybridEncoder, RepVggBlock
from kornia.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.models.rt_detr.model import RTDETR, RTDETRConfig
from kornia.models.rt_detr.post_processor import DETRPostProcessor

from testing.base import BaseTester, assert_close


@pytest.mark.parametrize(
    "backbone_factory",
    [partial(ResNetD.from_config, 18), partial(ResNetD.from_config, 50), partial(PPHGNetV2.from_config, "L")],
)
@pytest.mark.slow
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
    assert_close(actual, expected, atol=1e-3, rtol=1e-2)


class TestDETRPostProcessor(BaseTester):
    @pytest.mark.parametrize("sizes_dtype", [torch.float32, torch.int64])
    def test_postprocessor_scales_each_image_by_its_size(self, device, dtype, sizes_dtype):
        post_processor = DETRPostProcessor(
            confidence_threshold=0.0, num_classes=2, num_top_queries=1, confidence_filtering=False
        )
        logits = torch.tensor([[[5.0, -5.0]]] * 2, device=device, dtype=dtype)
        # one normalised (cx, cy, w, h) box per image
        boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]] * 2, device=device, dtype=dtype)
        # (H, W) of each image
        sizes = torch.tensor([[100, 200], [400, 800]], device=device, dtype=sizes_dtype)

        out = post_processor(logits, boxes, sizes)

        # xywh in the pixels of each image's own size: x = (cx - w / 2) * W, y = (cy - h / 2) * H, w * W, h * H.
        # Image 0: (0.375 * 200, 0.375 * 100, 0.25 * 200, 0.25 * 100); image 1: the same with H = 400, W = 800.
        expected = torch.tensor(
            [[[75.0, 37.5, 50.0, 25.0]], [[300.0, 150.0, 200.0, 100.0]]], device=device, dtype=dtype
        )
        assert out.shape == (2, 1, 6)
        assert out.dtype == dtype
        self.assert_close(out[..., 2:], expected)
        # the top query of sigmoid([5, -5]) is class 0
        assert out[..., 0].tolist() == [[0.0], [0.0]]


class TestRTDETR(BaseTester):
    @pytest.mark.slow  # This will be slow for the bigger variants
    @pytest.mark.parametrize("variant", ["resnet18d", "resnet34d", "resnet50d", "resnet101d", "hgnetv2_l", "hgnetv2_x"])
    def test_smoke(self, variant, device, dtype):
        model = RTDETR.from_config(RTDETRConfig(variant, 10)).to(device, dtype).eval()
        images = torch.randn(2, 3, 224, 256, device=device, dtype=dtype)
        out = model(images)

        assert isinstance(out, tuple)
        assert len(out) == 2

    @pytest.mark.parametrize("shape", [(1, 3, 96, 128), (2, 3, 64, 96)])
    def test_cardinality(self, shape, device, dtype):
        num_classes = 10
        num_queries = 10
        config = RTDETRConfig("resnet18d", num_classes, head_num_queries=num_queries)
        model = RTDETR.from_config(config).to(device, dtype).eval()

        images = torch.randn(shape, device=device, dtype=dtype)
        logits, boxes = model(images)

        assert logits.shape == (shape[0], num_queries, num_classes)
        assert boxes.shape == (shape[0], num_queries, 4)

    def test_resnet_d_odd_sizes(self, device, dtype):
        backbone = ResNetD.from_config(18).to(device, dtype).eval()
        # The stem takes a height of 636 to 159, an odd size. A stride-2 stage then gives ceil(159 / 2) = 80 on both
        # paths only when the shortcut pool rounds up, as the reference ``nn.AvgPool2d(2, 2, 0, ceil_mode=True)`` of
        # lyuwenyu/RT-DETR ``presnet.py`` does; 80 -> 40 -> 20 for the next two stages. The width, 64 -> 16 after the
        # stem, stays even (a narrow width keeps the run short in CPU half precision).
        fmaps = backbone(torch.rand(1, 3, 636, 64, device=device, dtype=dtype))
        assert [tuple(fmap.shape[-2:]) for fmap in fmaps] == [(80, 8), (40, 4), (20, 2)]

        # The whole detector: the stem takes 92 x 96 to 23 x 24, an odd height, and the backbone returns 12 x 12,
        # 6 x 6 and 3 x 3, sizes the neck's nearest x2 upsampling joins again.
        model = RTDETR.from_config(RTDETRConfig("resnet18d", 10, head_num_queries=10)).to(device, dtype).eval()
        logits, boxes = model(torch.rand(1, 3, 92, 96, device=device, dtype=dtype))
        assert logits.shape == (1, 10, 10)
        assert boxes.shape == (1, 10, 4)

    @pytest.mark.device_agnostic
    def test_decoder_eval_idx(self):
        in_channels = [32, 64, 128]
        head = RTDETRHead(5, 32, 10, in_channels, 3, eval_idx=0).eval()
        ran = []
        for i, layer in enumerate(head.decoder.layers):
            layer.register_forward_hook(lambda module, args, output, i=i: ran.append(i))
        generator = torch.Generator().manual_seed(0)
        fmaps = [
            torch.rand(1, ch, h, w, generator=generator) for ch, (h, w) in zip(in_channels, [(16, 12), (8, 6), (4, 3)])
        ]
        logits, boxes = head(fmaps)
        assert ran == [0]
        assert logits.shape == (1, 10, 5)
        assert boxes.shape == (1, 10, 4)

        def eval_idx(config: RTDETRConfig) -> int:
            return RTDETR.from_config(config).decoder.decoder.eval_idx

        # Both reference configs of the r50vd_m checkpoint (lyuwenyu/RT-DETR rtdetr_pytorch and rtdetr_paddle
        # ``rtdetr_r50vd_m_6x_coco.yml``) set ``eval_idx: 2`` on a 6-layer decoder; the other variants decode with
        # their last layer: 3 layers for r18, 6 for r50.
        assert eval_idx(RTDETRConfig("resnet50d_m", 80)) == 2
        assert eval_idx(RTDETRConfig("resnet18d", 80)) == 2
        assert eval_idx(RTDETRConfig("resnet50d", 80)) == 5
        assert eval_idx(RTDETRConfig("resnet50d_m", 80, eval_idx=4)) == 4
        assert eval_idx(RTDETRConfig("resnet50d_m", 80, eval_idx=0)) == 0
        assert eval_idx(RTDETRConfig("resnet18d", 80, eval_idx=-2)) == 1
        # a 2-layer r50vd_m decoder keeps its last layer, min(2, 2 - 1)
        assert eval_idx(RTDETRConfig("resnet50d_m", 80, head_num_decoder_layers=2)) == 1

        # a 3-layer decoder accepts -3..2
        for out_of_range in (3, -4):
            with pytest.raises(BaseError, match="eval_idx"):
                RTDETRHead(5, 32, 10, in_channels, 3, eval_idx=out_of_range)

    @pytest.mark.device_agnostic
    def test_from_pretrained_mode(self, monkeypatch):
        monkeypatch.setattr(
            "kornia.models.rt_detr.model.load_state_dict_from_url", lambda *args, **kwargs: {"ema": {"module": {}}}
        )
        monkeypatch.setattr(RTDETR, "load_state_dict", lambda self, state_dict, *args, **kwargs: None)
        # ``training`` of the model and of every submodule: ``{False}`` is evaluation mode throughout.
        assert {m.training for m in RTDETR.from_pretrained("rtdetr_r18vd").modules()} == {False}

        monkeypatch.setattr(RTDETR, "load_checkpoint", lambda self, checkpoint, device=None: None)
        from_checkpoint = RTDETR.from_config(RTDETRConfig("resnet18d", 10, checkpoint="x.pt"))
        assert {m.training for m in from_checkpoint.modules()} == {False}

        assert {m.training for m in RTDETR.from_config(RTDETRConfig("resnet18d", 10)).modules()} == {True}
        assert {m.training for m in RTDETR.from_name("rtdetr_r18vd").modules()} == {True}

    @pytest.mark.slow
    def test_from_pretrained_is_batch_independent(self):
        model = RTDETR.from_pretrained("rtdetr_r18vd")
        generator = torch.Generator().manual_seed(0)
        x1 = torch.rand(1, 3, 256, 320, generator=generator)
        x2 = torch.rand(1, 3, 256, 320, generator=generator)

        with torch.no_grad():
            alone = model(x1)[0][0]
            batched = model(torch.cat([x1, x2]))[0][0]

        # In evaluation mode image 0 does not see image 1; only the kernels' summation order may change with the
        # batch size, so the float32 tolerance is widened to 1e-4.
        self.assert_close(alone, batched, rtol=1e-4, atol=1e-4)

    @pytest.mark.skip("Unnecessary")
    def test_exception(self): ...

    @pytest.mark.skip("Unnecessary")
    def test_gradcheck(self): ...

    @pytest.mark.skip("Unnecessary")
    def test_module(self): ...

    @pytest.mark.skip("Needs more investigation")
    @pytest.mark.parametrize("variant", ["resnet50d", "hgnetv2_l"])
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
