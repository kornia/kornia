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

import warnings

import pytest
import torch

from kornia.enhance.normalize import Normalize
from kornia.feature.dedode import DeDoDe
from kornia.feature.dedode.decoder import ConvRefiner
from kornia.feature.dedode.transformer.dinov2 import DinoVisionTransformer
from kornia.feature.dedode.transformer.layers import MemEffAttention, NestedTensorBlock
from kornia.feature.dedode.utils import dedode_denormalize_pixel_coordinates

from testing.base import BaseTester


class TestConvRefiner(BaseTester):
    def test_depthwise_channels_must_be_divisible(self):
        refiner = ConvRefiner(in_dim=4, hidden_dim=4, out_dim=4)
        with pytest.raises(ValueError, match="out_dim must be divisible by in_dim"):
            refiner.create_block(in_dim=3, out_dim=4, dw=True)

    def test_amp_matches_input_device(self, device):
        refiner = ConvRefiner(in_dim=4, hidden_dim=4, out_dim=4).to(device)
        autocast_enabled = []
        # ``torch.is_autocast_enabled`` only accepts a device type from torch 2.4 onwards; the
        # no-argument spelling reports the CUDA state on every torch kornia supports, which is
        # the state the ``torch.autocast("cuda", ...)`` region inside ``ConvRefiner`` controls.
        refiner.block1.register_forward_pre_hook(
            lambda _module, _inputs: autocast_enabled.append(torch.is_autocast_enabled())
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refiner(torch.rand(1, 4, 8, 8, device=device))

        assert autocast_enabled == [device.type == "cuda"]
        assert not any("device_type of 'cuda'" in str(w.message) for w in caught)


class TestMemEffAttention(BaseTester):
    def test_attn_bias_raises(self, device, dtype):
        attention = MemEffAttention(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        with pytest.raises(NotImplementedError, match="attn_bias is not supported"):
            attention(x, attn_bias=torch.zeros(1, 2, 5, 5, device=device, dtype=dtype))


class TestNestedTensorBlock(BaseTester):
    def test_tensor_input(self, device, dtype):
        block = NestedTensorBlock(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        out = block(x)

        assert out.shape == (1, 5, 8)
        # the block is a residual one, so the output has to differ from the input by the
        # attention and feed-forward branches actually having run.
        assert not torch.allclose(out, x)

    def test_list_input_raises(self, device, dtype):
        block = NestedTensorBlock(dim=8, num_heads=2).to(device, dtype).eval()
        x = torch.rand(1, 5, 8, device=device, dtype=dtype)

        with pytest.raises(TypeError, match="nested-tensor list path was removed"):
            block([x])


class TestDinoVisionTransformer(BaseTester):
    def test_forward_features_tensor(self, device, dtype):
        model = DinoVisionTransformer(
            img_size=8,
            patch_size=4,
            embed_dim=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
            block_chunks=0,
        ).to(device, dtype)
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        masks = torch.zeros(2, 4, device=device, dtype=torch.bool)

        features = model.forward_features(x, masks)

        assert set(features) == {"x_norm_clstoken", "x_norm_patchtokens", "x_prenorm", "masks"}
        assert features["x_norm_clstoken"].shape == (2, 8)
        assert features["x_norm_patchtokens"].shape == (2, 4, 8)
        assert features["x_prenorm"].shape == (2, 5, 8)
        assert features["masks"] is masks


class _BlurNet(torch.nn.Module):
    """Stand-in for a DeDoDe network: a 15x15 box blur, so the output near the border depends on the padded values."""

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.seen = []

    def forward(self, images):
        self.seen.append(images)
        blurred = torch.nn.functional.avg_pool2d(images, 15, stride=1, padding=7)
        return blurred[:, : self.out_channels]


class TestDeDoDePadding(BaseTester):
    @staticmethod
    def _model(device, dtype):
        # Networks are replaced by blurs: the padding rule is under test, not the pretrained weights.
        model = DeDoDe.__new__(DeDoDe)  # skips building the VGG networks
        torch.nn.Module.__init__(model)
        model.detector = _BlurNet(1)
        model.descriptor = _BlurNet(3)
        model.normalizer = Normalize(torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        return model.to(device, dtype)

    @staticmethod
    def _image(device, dtype, h=50, w=80):
        gen = torch.Generator().manual_seed(7)
        return torch.rand(1, 3, h, w, generator=gen).to(device, dtype)

    @pytest.mark.parametrize("entry", ["forward", "detect", "describe"])
    def test_pad_is_the_mean_colour_in_every_entry_point_5730(self, device, dtype, entry):
        model = self._model(device, dtype)
        x = self._image(device, dtype)  # 50x80 -> 56x84
        if entry == "forward":
            model(x, n=4)
        elif entry == "detect":
            model.detect(x, n=4)
        else:
            model.describe(x)
        net = model.descriptor if entry == "describe" else model.detector
        seen = net.seen[0]
        assert seen.shape == (1, 3, 56, 84)
        normalized = model.normalizer(x)
        self.assert_close(seen[..., :50, :80], normalized)
        # padded after the normalisation: the pad is the ImageNet mean colour, 0 in normalised units
        assert seen[..., 50:, :].abs().max() == 0
        assert seen[..., :, 80:].abs().max() == 0

    def test_detect_and_describe_agree_with_forward_5730(self, device, dtype):
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        n = 40
        kpts, scores, descs = model(x, n=n)
        kpts_n, scores_d = model.detect(x, n=n)
        descs_d = model.describe(x, kpts_n)
        self.assert_close(dedode_denormalize_pixel_coordinates(kpts_n, 50, 80), kpts)
        self.assert_close(scores_d, scores)
        self.assert_close(descs_d, descs)

    def test_dense_descriptors_have_the_size_of_the_image_5730(self, device, dtype):
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        assert model.describe(x).shape == (1, 3, 50, 80)

    def test_multiple_of_14_is_not_padded(self, device, dtype):
        model = self._model(device, dtype)
        x = self._image(device, dtype, h=28, w=42)
        model(x, n=4)
        model.detect(x, n=4)
        model.describe(x)
        for seen in model.detector.seen + model.descriptor.seen:
            self.assert_close(seen, model.normalizer(x))

    def test_pad_can_be_disabled(self, device, dtype):
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        model.detect(x, n=4, pad_if_not_divisible=False)
        model.describe(x, pad_if_not_divisible=False)
        assert model.detector.seen[0].shape == (1, 3, 50, 80)
        assert model.descriptor.seen[0].shape == (1, 3, 50, 80)


@pytest.mark.skip(reason="DeDoDe is ummaintained")
class TestDeDoDe:
    @pytest.mark.slow
    @pytest.mark.parametrize("descriptor_model", ["B", "G"])
    @pytest.mark.parametrize("detector_model", ["L"])
    def test_smoke(self, dtype, device, descriptor_model, detector_model):
        if "G" in descriptor_model and device.type != "cuda" and dtype == torch.float16:
            pytest.skip('G descriptors do not support no cuda device. "LayerNormKernelImpl" not implemented for `Half`')
        dedode = DeDoDe(descriptor_model=descriptor_model, detector_model=detector_model, amp_dtype=dtype).to(
            device, dtype
        )
        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)
        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)
