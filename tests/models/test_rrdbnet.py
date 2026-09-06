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
from kornia.models import rrdbnet as rrdbnet_module
from kornia.models.processors import OutputRangePostProcessor
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


def _cosine_ramp_weights(model: nn.Module) -> None:
    """Overwrite every parameter with an RNG-free pattern of roughly kaiming magnitude.

    Parameter ``i`` in the sorted ``state_dict`` becomes ``0.1 * cos(0.37 * arange(numel) + i)``, computed in
    float64 and cast to the parameter's dtype, so the same weights come out on every device, dtype and torch
    version.
    """
    with torch.no_grad():
        for i, (_, param) in enumerate(sorted(model.state_dict().items())):
            ramp = torch.cos(torch.arange(param.numel(), dtype=torch.float64) * 0.37 + i)
            param.copy_((0.1 * ramp).reshape(param.shape).to(param.dtype))


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

    @pytest.mark.parametrize("scale", [0, 3, 8])
    def test_exception_rejects_unsupported_scale(self, scale):
        # Upstream lets any scale other than 1/2 fall through to the x4 path, so `scale=3` silently
        # returns a 4x output. The vendored copy rejects it at construction instead.
        with pytest.raises(ValueError, match="scale must be 1, 2 or 4"):
            tiny_rrdbnet(scale=scale)

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
        # BatchNorm2d already ships ones/zeros, so perturb both first: without this the BatchNorm
        # assertions below pass whether or not the `_BatchNorm` branch ever runs.
        module[2].weight.data.fill_(3.0)
        module[2].bias.data.fill_(3.0)
        for m in (module[0], module[1]):
            m.bias.data.fill_(3.0)

        _default_init_weights(module, scale=0.1, bias_fill=0.0)
        assert torch.count_nonzero(module[0].bias) == 0
        assert torch.count_nonzero(module[1].bias) == 0
        self.assert_close(module[2].weight, torch.ones_like(module[2].weight))
        assert torch.count_nonzero(module[2].bias) == 0
        # a single module and a list of modules are both accepted
        module[0].bias.data.fill_(3.0)
        _default_init_weights([module[0]], scale=0.1)
        assert torch.count_nonzero(module[0].bias) == 0

    def test_default_init_weights_applies_the_scale(self, device, dtype):
        """``scale`` multiplies the drawn weights -- the 0.1 residual scaling ESRGAN needs.

        ``kaiming_normal_`` draws from the global RNG, so seeding both calls identically makes the
        two initializations differ only by the multiplier.
        """
        unscaled = nn.Conv2d(4, 4, 3).to(device, dtype)
        scaled = nn.Conv2d(4, 4, 3).to(device, dtype)

        torch.manual_seed(0)
        _default_init_weights(unscaled, scale=1.0)
        torch.manual_seed(0)
        _default_init_weights(scaled, scale=0.1)

        assert torch.count_nonzero(unscaled.weight) > 0
        self.assert_close(scaled.weight, unscaled.weight * 0.1)

    def test_residual_dense_block_init_is_scaled_by_0_1(self, monkeypatch):
        """Upstream initializes the five dense convolutions with ``default_init_weights(..., 0.1)``.

        The multiplier is what the checkpoints were trained from; a block that skips it still builds
        and runs, so the call is recorded rather than inferred from weight statistics.
        """
        calls = []
        monkeypatch.setattr(
            rrdbnet_module, "_default_init_weights", lambda modules, scale=1.0, **kw: calls.append((modules, scale))
        )
        block = ResidualDenseBlock(num_feat=8, num_grow_ch=4)
        assert calls == [([block.conv1, block.conv2, block.conv3, block.conv4, block.conv5], 0.1)]

    def test_rrdb_residual_scaling(self, device, dtype):
        """Pin the outer 0.2 residual scaling of :class:`RRDB`, separately from the inner one."""
        block = RRDB(num_feat=8, num_grow_ch=4).to(device, dtype).eval()
        x = torch.rand(1, 8, 6, 6, device=device, dtype=dtype)
        with torch.no_grad():
            out = block(x)
            expected = block.rdb3(block.rdb2(block.rdb1(x))) * 0.2 + x
        self.assert_close(out, expected)

    @pytest.mark.parametrize(
        ("scale", "expected_sum", "expected_first", "expected_last"),
        [
            (
                4,
                20.91297187055941,
                [-0.004543595971873713, 0.027261616129030625, 0.049594551942778516],
                [-0.016568600126998914, 0.036161773272664045, 0.06274978373798744],
            ),
            (
                2,
                5.639646965369472,
                [-0.0017096129540983347, 0.02974716223297922, 0.047076207498800086],
                [-0.01746539690853942, 0.036172963633466226, 0.0636480015792775],
            ),
            (
                1,
                1.3505577490199843,
                [-0.0021678145765582375, 0.029241923490125363, 0.04747024904152217],
                [-0.017201746167144025, 0.03585330869516471, 0.0633437579763486],
            ),
        ],
    )
    def test_numerical_matches_upstream(
        self, device, dtype, scale, expected_sum, expected_first, expected_last, cudnn_tf32_follows_option
    ):
        """Pin the forward pass against upstream BasicSR on RNG-free weights.

        The name snapshot above pins the parameter *layout*; this pins the *arithmetic* (the two 0.2
        residual scalings, the 0.2 LeakyReLU slope, nearest-neighbour upsampling and the unshuffle
        order), which a tensor of the right shape says nothing about. Every parameter is overwritten
        with a cosine ramp keyed on its position in the sorted ``state_dict``, so the reference
        depends on no random draw and holds across torch versions. Generated in float64 with
        ``basicsr/archs/rrdbnet_arch.py`` at BasicSR ``master`` (2026-09), its ``ARCH_REGISTRY``
        decorator and ``arch_util`` import replaced by the three helpers it uses::

            model = RRDBNet(3, 3, scale=scale, num_feat=8, num_block=1, num_grow_ch=4).double().eval()
            _cosine_ramp_weights(model)                          # the module-level helper above
            out = model(torch.linspace(0.0, 1.0, 48, dtype=torch.float64).reshape(1, 3, 4, 4))
            out.sum(), out[0, :, 0, 0], out[0, :, -1, -1]

        Compared to the same snippet on the vendored class, the outputs were ``torch.equal``.
        """
        model = tiny_rrdbnet(scale=scale).to(device, dtype).eval()
        _cosine_ramp_weights(model)
        x = torch.linspace(0.0, 1.0, 48, dtype=torch.float64).reshape(1, 3, 4, 4).to(device, dtype)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 3, 4 * scale, 4 * scale)
        self.assert_close(out.sum(), torch.tensor(expected_sum, device=device, dtype=dtype))
        self.assert_close(out[0, :, 0, 0], torch.tensor(expected_first, device=device, dtype=dtype))
        self.assert_close(out[0, :, -1, -1], torch.tensor(expected_last, device=device, dtype=dtype))

    def test_gradcheck(self, device):
        pytest.skip("RRDBNet is a deep convolutional generator; gradcheck is prohibitively slow.")

    def test_dynamo(self, device, dtype, torch_optimizer, cudnn_tf32_follows_option):
        # `cudnn_tf32_follows_option` keeps the CUDA float32 leg in real float32 (see its docstring).
        # The tolerances are `assert_close`'s per-dtype defaults: inductor and eager legitimately
        # differ by an ulp or so on the half dtypes, which a hard-coded 1e-4 rejects.
        model = tiny_rrdbnet(scale=4).to(device, dtype).eval()
        x = torch.rand(1, 3, 8, 8, device=device, dtype=dtype)

        op = model
        op_optimized = torch_optimizer(model)

        with torch.no_grad():
            self.assert_close(op(x), op_optimized(x))


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
        # `SuperResolution` cannot be instantiated today (`ModelBase.from_config` is abstract --
        # kornia#4291, a pre-existing defect that also skips this builder in the export survey), so
        # the wrapper is stubbed out to reach the model the builder constructs. Once #4291 is fixed,
        # drop the stub and assert on the returned `SuperResolution` directly.
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
        assert model.conv_first.out_channels == 64  # num_feat=64
        assert model.body[0].rdb1.conv1.out_channels == 32  # num_grow_ch=32
        assert not model.training
        assert captured["kwargs"]["name"] == model_name

        # the rest of `build`'s contract: no pre-processing, outputs clamped back into [0, 1]
        assert isinstance(captured["kwargs"]["pre_processor"], nn.Identity)
        post_processor = captured["kwargs"]["post_processor"]
        assert isinstance(post_processor, OutputRangePostProcessor)
        assert (post_processor.min_val, post_processor.max_val) == (0.0, 1.0)

    def test_build_rejects_an_unknown_model_name(self):
        with pytest.raises(ValueError, match="not found"):
            RRDBNetBuilder.build("not_a_model", pretrained=False)
