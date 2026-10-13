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

    @pytest.mark.parametrize("entry", ["forward", "detect_describe"])
    def test_pad_can_be_disabled(self, device, dtype, entry):
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        if entry == "forward":
            model(x, n=4, pad_if_not_divisible=False)
        else:
            model.detect(x, n=4, pad_if_not_divisible=False)
            model.describe(x, pad_if_not_divisible=False)
        assert model.detector.seen[0].shape == (1, 3, 50, 80)
        assert model.descriptor.seen[0].shape == (1, 3, 50, 80)


class TestDeDoDeHalfPrecision(BaseTester):
    def test_detect_runs_in_the_model_dtype_5756(self, device, dtype):
        model = DeDoDe(detector_model="L", descriptor_model="B", amp_dtype=dtype).to(device, dtype).eval()
        image = torch.rand(1, 3, 56, 84, device=device, dtype=dtype)
        keypoints, scores = model.detect(image, n=10)
        assert keypoints.shape == (1, 10, 2)
        assert scores.shape == (1, 10)
        assert torch.isfinite(scores).all()


class TestDeDoDeNoGrad(BaseTester):
    @staticmethod
    def _model(device, dtype):
        # Random weights: the compiled graph does not depend on the pretrained values, and nothing is downloaded.
        torch.manual_seed(0)
        return DeDoDe(detector_model="L", descriptor_model="B", amp_dtype=dtype).to(device, dtype).eval()

    @staticmethod
    def _image(device, dtype):
        gen = torch.Generator().manual_seed(7)
        return torch.rand(1, 3, 56, 84, generator=gen).to(device, dtype)

    @staticmethod
    def _skip_half(dtype):
        if dtype in (torch.float16, torch.bfloat16):
            # Eager runs in half precision, but the compiled outputs differ from it by more than the tolerances below
            # (torch 2.14 CPU: descriptors by 4e-4 in float16 and 3e-3 in bfloat16; the last bfloat16 keypoints of the
            # top 100 move by a pixel).
            pytest.skip("compiled half-precision DeDoDe differs from eager by more than these tolerances")

    def test_outputs_are_ordinary_tensors_without_grad_5737(self, device, dtype):
        # `detect` and `describe` run under `torch.no_grad()`, not `torch.inference_mode()`: inference tensors
        # are what made inductor raise "Cannot set version_counter for inference tensor" on these two methods.
        model = TestDeDoDePadding._model(device, dtype)
        x = TestDeDoDePadding._image(device, dtype).requires_grad_()
        kpts, scores = model.detect(x, n=4)
        descs = model.describe(x, kpts)
        for out in (kpts, scores, descs, model.describe(x)):
            assert not out.is_inference()
            assert not out.requires_grad
        # `sample_keypoints` has a `no_grad` of its own, so the outputs of `detect` cannot show whether the network
        # ran with gradients off: check what the networks were fed.
        assert not model.detector.seen[0].requires_grad
        assert not model.descriptor.seen[0].requires_grad

    def test_dynamo_detect_5737(self, device, dtype, torch_optimizer, cudnn_tf32_follows_option):
        self._skip_half(dtype)
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        n = 100
        kpts, scores = model.detect(x, n=n)
        kpts_c, scores_c = torch_optimizer(model.detect)(x, n=n)
        # The compiled top-k may order equal scores differently and move a coordinate by one float32 ulp, so the
        # keypoints are compared as sets: each one has a neighbour in the other set far closer than a pixel.
        # (`torch.cdist` is not used: its matmul path returns 5e-4 for two identical float32 points.)
        dist = (kpts_c[:, :, None] - kpts[:, None]).abs().amax(dim=-1)
        assert dist.amin(dim=-1).max() < 1e-4
        assert dist.amin(dim=-2).max() < 1e-4
        self.assert_close(scores_c, scores, rtol=1e-3, atol=1e-6)

    def test_dynamo_describe_5737(self, device, dtype, torch_optimizer, cudnn_tf32_follows_option):
        self._skip_half(dtype)
        model = self._model(device, dtype)
        x = self._image(device, dtype)
        kpts, _ = model.detect(x, n=100)
        descs_c = torch_optimizer(model.describe)(x, kpts)
        self.assert_close(descs_c, model.describe(x, kpts), rtol=1e-3, atol=1e-4)


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


class _FixedMap(torch.nn.Module):
    """A DeDoDe network stand-in that returns ``fn(images)`` and records its input."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.seen = []

    def forward(self, images):
        self.seen.append(images)
        return self.fn(images)


class TestConventionsDeDoDe(BaseTester):
    @staticmethod
    def _model(monkeypatch, peaks):
        # DeDoDe with its networks replaced (no VGG is built, nothing is downloaded): the detector returns logits -20
        # with the given peaks (x, y, logit), the descriptor a two-channel map (column, row), so a description names
        # the pixel it was read at.
        def detector(images):
            b, _, h, w = images.shape
            logits = torch.full((b, 1, h, w), -20.0, device=images.device, dtype=images.dtype)
            for x, y, logit in peaks:
                if y < h and x < w:
                    logits[:, 0, y, x] = logit
            return logits

        def descriptor(images):
            b, _, h, w = images.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, device=images.device), torch.arange(w, device=images.device), indexing="ij"
            )
            return torch.stack([xx, yy])[None].expand(b, 2, h, w).to(images.dtype)

        monkeypatch.setattr("kornia.feature.dedode.dedode.get_detector", lambda *a, **k: _FixedMap(detector))
        monkeypatch.setattr("kornia.feature.dedode.dedode.get_descriptor", lambda *a, **k: _FixedMap(descriptor))
        return DeDoDe(detector_model="L", descriptor_model="B")

    def test_convention_dedode_forward_returns_edge_aligned_pixels_and_detect_normalised_ones(
        self, device, dtype, monkeypatch
    ):
        # Logit peaks at pixels (60, 40) and (21, 13) of a 56 x 84 image. forward returns pixel i as i + 0.5 (subtract
        # 0.5 for kornia's frame), detect returns (2 i + 1) / (W, H) - 1 in [-1, 1], and the description is read at
        # pixel i itself; keypoints and descriptions are float32 whatever the model dtype, keypoints come sorted by
        # score, N = min(n, H W) with no threshold or padding. The grid and its pixel conversion are the formulas of
        # Parskatt/DeDoDe 6d15618 (utils.get_grid, to_pixel_coords).
        peaks = [(21, 13, 18.0), (60, 40, 20.0)]
        pixels = torch.tensor([[[60.0, 40.0], [21.0, 13.0]]], device=device)
        for h, w, peaks_i, image_pixels in (
            (56, 84, peaks, pixels),
            (84, 56, [(y, x, v) for x, y, v in peaks], pixels.flip(-1)),  # relabelling: transpose
        ):
            model = self._model(monkeypatch, peaks_i).to(device, dtype)
            image = torch.rand(1, 3, h, w, device=device, dtype=dtype)
            keypoints, scores, descriptions = model(image, n=2)
            normalised, _ = model.detect(image, n=2)
            assert keypoints.dtype == normalised.dtype == descriptions.dtype == torch.float32
            self.assert_close(keypoints, image_pixels + 0.5)
            size = torch.tensor([w, h], device=device)
            self.assert_close(normalised, (2.0 * image_pixels + 1.0) / size - 1.0)
            self.assert_close(descriptions, image_pixels)
            assert scores[0, 0] > scores[0, 1]
        keypoints, scores, descriptions = model(torch.rand(2, 3, 14, 28, device=device, dtype=dtype), n=1000)
        assert keypoints.shape == (2, 392, 2)
        assert scores.shape == (2, 392)
        assert descriptions.shape == (2, 392, 2)

    def test_convention_dedode_takes_rgb_in_the_unit_range_and_imagenet_normalises_it(self, device, dtype, monkeypatch):
        # forward, detect and describe feed the networks (x - mean) / std with ImageNet's mean (0.485, 0.456, 0.406) and
        # std (0.229, 0.224, 0.225), the normalisation the pretrained weights expect (Parskatt/DeDoDe's image loader);
        # apply_imagenet_normalization=False feeds the input as given.
        model = self._model(monkeypatch, [(21, 13, 18.0), (60, 40, 20.0)]).to(device, dtype)
        image = torch.rand(1, 3, 56, 84, device=device, dtype=dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        model(image, n=2)
        model.detect(image, n=2)
        model.describe(image)
        model(image, n=2, apply_imagenet_normalization=False)
        seen = model.detector.seen + model.descriptor.seen
        assert len(seen) == 6
        for net in (model.detector, model.descriptor):
            for normalised in net.seen[:2]:
                self.assert_close(normalised, (image - mean) / std)
            self.assert_close(net.seen[2], image, rtol=0.0, atol=0.0)
