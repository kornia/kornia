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

import pytest
import torch
from torch import nn

from kornia.contrib import super_resolution as super_resolution_module
from kornia.contrib.super_resolution import RRDBNetBuilder
from kornia.models import RRDBNet
from kornia.models.rrdbnet import RRDB, ResidualDenseBlock, _default_init_weights

from testing.base import BaseTester

# The upstream attribute names, which the published Real-ESRGAN checkpoints are keyed on. Read off
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py: `conv_first`,
# `body` (an `nn.Sequential` of `RRDB`, each holding `rdb1`..`rdb3` of `conv1`..`conv5`),
# `conv_body`, `conv_up1`, `conv_up2`, `conv_hr`, `conv_last`. A rename here silently breaks
# `load_state_dict(..., strict=True)` on every released checkpoint, so the names are pinned.
EXPECTED_KEYS_ONE_BLOCK = [
    "body.0.rdb1.conv1.bias",
    "body.0.rdb1.conv1.weight",
    "body.0.rdb1.conv2.bias",
    "body.0.rdb1.conv2.weight",
    "body.0.rdb1.conv3.bias",
    "body.0.rdb1.conv3.weight",
    "body.0.rdb1.conv4.bias",
    "body.0.rdb1.conv4.weight",
    "body.0.rdb1.conv5.bias",
    "body.0.rdb1.conv5.weight",
    "body.0.rdb2.conv1.bias",
    "body.0.rdb2.conv1.weight",
    "body.0.rdb2.conv2.bias",
    "body.0.rdb2.conv2.weight",
    "body.0.rdb2.conv3.bias",
    "body.0.rdb2.conv3.weight",
    "body.0.rdb2.conv4.bias",
    "body.0.rdb2.conv4.weight",
    "body.0.rdb2.conv5.bias",
    "body.0.rdb2.conv5.weight",
    "body.0.rdb3.conv1.bias",
    "body.0.rdb3.conv1.weight",
    "body.0.rdb3.conv2.bias",
    "body.0.rdb3.conv2.weight",
    "body.0.rdb3.conv3.bias",
    "body.0.rdb3.conv3.weight",
    "body.0.rdb3.conv4.bias",
    "body.0.rdb3.conv4.weight",
    "body.0.rdb3.conv5.bias",
    "body.0.rdb3.conv5.weight",
    "conv_body.bias",
    "conv_body.weight",
    "conv_first.bias",
    "conv_first.weight",
    "conv_hr.bias",
    "conv_hr.weight",
    "conv_last.bias",
    "conv_last.weight",
    "conv_up1.bias",
    "conv_up1.weight",
    "conv_up2.bias",
    "conv_up2.weight",
]


def tiny_rrdbnet(scale: int = 4) -> RRDBNet:
    """Build the smallest RRDBNet that still exercises every branch of the forward."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=8, num_block=1, num_grow_ch=4)


class TestRRDBNet(BaseTester):
    def test_smoke(self, device, dtype):
        model = tiny_rrdbnet(scale=4).to(device, dtype)
        out = model(torch.rand(1, 3, 8, 8, device=device, dtype=dtype))
        assert out.shape == (1, 3, 32, 32)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        ("scale", "expected_shape"),
        # scale 2 pixel-unshuffles by 2 and scale 1 by 4 before the two fixed x2 upsamplings, so the
        # net factor is `scale` in every case.
        [(4, (2, 3, 32, 32)), (2, (2, 3, 16, 16)), (1, (2, 3, 8, 8))],
    )
    def test_cardinality(self, device, dtype, scale, expected_shape):
        model = tiny_rrdbnet(scale=scale).to(device, dtype)
        out = model(torch.rand(2, 3, 8, 8, device=device, dtype=dtype))
        assert out.shape == expected_shape

    def test_exception(self, device, dtype):
        # scale=2 pixel-unshuffles by 2, so an odd spatial size cannot be unshuffled.
        model = tiny_rrdbnet(scale=2).to(device, dtype)
        with pytest.raises(RuntimeError):
            model(torch.rand(1, 3, 7, 7, device=device, dtype=dtype))

    def test_module(self, device, dtype):
        model = tiny_rrdbnet(scale=4).to(device, dtype)
        assert isinstance(model.body[0], RRDB)
        assert isinstance(model.body[0].rdb1, ResidualDenseBlock)
        assert model.scale == 4

    def test_state_dict_keys_match_upstream(self, device, dtype):
        model = tiny_rrdbnet(scale=4).to(device, dtype)
        assert sorted(model.state_dict().keys()) == EXPECTED_KEYS_ONE_BLOCK

    @pytest.mark.parametrize("scale", [1, 2, 4])
    def test_state_dict_roundtrip_is_strict(self, device, dtype, scale):
        """A checkpoint saved from one instance must load into another with ``strict=True``."""
        source = tiny_rrdbnet(scale=scale).to(device, dtype)
        target = tiny_rrdbnet(scale=scale).to(device, dtype)
        target.load_state_dict(source.state_dict(), strict=True)
        source.eval()
        target.eval()
        x = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)
        with torch.no_grad():
            self.assert_close(source(x), target(x))

    def test_num_in_ch_is_widened_for_small_scales(self, device, dtype):
        """``scale`` 2 and 1 unshuffle by 2 and 4, so ``conv_first`` takes 4x and 16x the channels."""
        assert tiny_rrdbnet(scale=4).conv_first.in_channels == 3
        assert tiny_rrdbnet(scale=2).conv_first.in_channels == 3 * 4
        assert tiny_rrdbnet(scale=1).conv_first.in_channels == 3 * 16

    def test_residual_dense_block_scaling(self, device, dtype):
        """Pin the 0.2 residual scaling that upstream applies in both block types."""
        block = ResidualDenseBlock(num_feat=8, num_grow_ch=4).to(device, dtype).eval()
        x = torch.rand(1, 8, 6, 6, device=device, dtype=dtype)
        with torch.no_grad():
            out = block(x)
            x5 = block.conv5(
                torch.cat(
                    (
                        x,
                        (x1 := block.lrelu(block.conv1(x))),
                        (x2 := block.lrelu(block.conv2(torch.cat((x, x1), 1)))),
                        (x3 := block.lrelu(block.conv3(torch.cat((x, x1, x2), 1)))),
                        block.lrelu(block.conv4(torch.cat((x, x1, x2, x3), 1))),
                    ),
                    1,
                )
            )
        self.assert_close(out, x5 * 0.2 + x)

    def test_default_init_weights(self, device, dtype):
        """The vendored initializer zeroes biases and sets BatchNorm weights to one."""
        module = nn.Sequential(nn.Conv2d(2, 2, 3), nn.Linear(2, 2), nn.BatchNorm2d(2)).to(device, dtype)
        _default_init_weights(module, scale=0.1, bias_fill=0.0)
        assert torch.count_nonzero(module[0].bias) == 0
        assert torch.count_nonzero(module[1].bias) == 0
        self.assert_close(module[2].weight, torch.ones_like(module[2].weight))
        # a single module and a list of modules are both accepted
        _default_init_weights([module[0]], scale=0.1)
        assert torch.count_nonzero(module[0].bias) == 0

    def test_gradcheck(self, device):
        pytest.skip("RRDBNet is a deep convolutional generator; gradcheck is prohibitively slow.")

    def test_dynamo(self, device, dtype, torch_optimizer):
        model = tiny_rrdbnet(scale=4).to(device, dtype).eval()
        x = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)

        op = model
        op_optimized = torch_optimizer(model)

        with torch.no_grad():
            self.assert_close(op(x), op_optimized(x), rtol=1e-4, atol=1e-4)


class TestRRDBNetBuilder:
    """Cover ``RRDBNetBuilder.build``'s architecture selection, now served by the vendored generator."""

    @pytest.mark.parametrize(
        ("model_name", "scale", "num_block"),
        [
            ("RealESRGAN_x4plus", 4, 23),
            ("RealESRNet_x4plus", 4, 23),
            ("RealESRGAN_x4plus_anime_6B", 4, 6),
            ("RealESRGAN_x2plus", 2, 23),
        ],
    )
    def test_build_selects_the_vendored_rrdbnet(self, monkeypatch, model_name, scale, num_block):
        # `SuperResolution` cannot be instantiated today (`ModelBase.from_config` is abstract, a
        # pre-existing kornia defect that also skips this builder in the export survey), so the
        # wrapper is stubbed out to reach the model the builder constructs.
        captured = {}

        def record(model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return model

        monkeypatch.setattr(super_resolution_module, "SuperResolution", record)

        returned = RRDBNetBuilder.build(model_name, pretrained=False)

        model = captured["model"]
        assert model is returned
        assert isinstance(model, RRDBNet)
        assert model.scale == scale
        assert len(model.body) == num_block
        assert model.conv_first.out_channels == 64
        assert not model.training
        assert captured["kwargs"]["name"] == model_name

    def test_build_rejects_an_unknown_model_name(self):
        with pytest.raises(ValueError, match="not found"):
            RRDBNetBuilder.build("not_a_model", pretrained=False)
