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

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import kornia
from kornia.core.utils import _clamp_range

from testing.base import BaseTester


class TestColorConventions(BaseTester):
    def test_convention_hls_hue_is_radians(self, device, dtype):
        # A pure green RGB sample has hue 2π/3 in the HLS convention.
        image = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]], device=device, dtype=dtype)

        hls = kornia.color.rgb_to_hls(image)

        self.assert_close(hls[:, 0], torch.full((1, 1, 1), 2.0 * math.pi / 3.0, device=device, dtype=dtype))
        self.assert_close(kornia.color.hls_to_rgb(hls), image)

    def test_convention_xyz_uses_linear_rgb(self, device, dtype):
        # rgb_to_xyz applies the matrix directly, without sRGB decoding: a 0.5 gray maps to 0.5 times the
        # row sums of the D65 matrix (0.950456, 1.0, 1.088754), not to the decoded 0.214 times them.
        srgb = torch.full((1, 3, 1, 1), 0.5, device=device, dtype=dtype)

        expected = torch.tensor([0.475228, 0.5, 0.544377], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.assert_close(kornia.color.rgb_to_xyz(srgb), expected)
        self.assert_close(kornia.color.xyz_to_rgb(expected), srgb)

    def test_convention_xyz_linear_rgb_matrix(self, device, dtype):
        image = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        xyz = kornia.color.rgb_to_xyz(image)

        expected = torch.tensor([[[[0.412453]], [[0.212671]], [[0.019334]]]], device=device, dtype=dtype)
        self.assert_close(xyz, expected)

    def test_convention_grayscale_to_rgb_5321_does_not_alias_input(self, device, dtype):
        image = torch.tensor([[[[0.2, 0.7]]]], device=device, dtype=dtype)
        before = image.clone()

        rgb = kornia.color.grayscale_to_rgb(image)
        assert rgb.shape == (1, 3, 1, 2)
        rgb[:, 0] = 0.9

        self.assert_close(image, before)
        self.assert_close(rgb[:, 1:], before.expand(1, 2, 1, 2))

        rgb = kornia.color.grayscale_to_rgb(image)
        rgb.add_(0.1)
        self.assert_close(image, before)
        self.assert_close(rgb, before.expand(1, 3, 1, 2) + 0.1)

    def test_convention_rgba_composites_over_white(self, device, dtype):
        rgba = torch.tensor([[[[0.2]], [[0.4]], [[0.6]], [[0.25]]]], device=device, dtype=dtype)
        background = (0.0, 0.5, 1.0)

        self.assert_close(
            kornia.color.rgba_to_rgb(rgba),
            torch.tensor([[[[0.8]], [[0.85]], [[0.9]]]], device=device, dtype=dtype),
        )
        self.assert_close(
            kornia.color.rgba_to_rgb(rgba, background),
            torch.tensor([[[[0.05]], [[0.475]], [[0.9]]]], device=device, dtype=dtype),
        )
        self.assert_close(kornia.color.rgba_to_bgr(rgba), kornia.color.rgb_to_bgr(kornia.color.rgba_to_rgb(rgba)))

    @pytest.mark.parametrize("per_pixel_background", [False, True])
    def test_convention_rgba_rank3_tensor_background_preserves_shape(self, device, dtype, per_pixel_background):
        rgba = torch.tensor([0.2, 0.4, 0.6, 0.25], device=device, dtype=dtype).view(4, 1, 1).expand(4, 2, 3)
        background = torch.tensor([0.0, 0.5, 1.0], device=device, dtype=dtype).view(3, 1, 1)
        if per_pixel_background:
            background = background.expand(3, 2, 3)

        result = kornia.color.rgba_to_rgb(rgba, background)

        assert result.shape == (3, 2, 3)
        expected = torch.tensor([0.05, 0.475, 0.9], device=device, dtype=dtype).view(3, 1, 1).expand(3, 2, 3)
        self.assert_close(result, expected)

    def test_convention_bayer_rg_layout_and_transpose_relabeling(self, device, dtype):
        image = torch.tensor(
            [
                [
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                    [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
                    [[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        raw_rg = kornia.color.rgb_to_raw(image, kornia.color.CFA.RG)
        transposed = kornia.color.rgb_to_raw(image.transpose(-2, -1), kornia.color.CFA.RG).transpose(-2, -1)

        expected = torch.tensor([[[[100.0, 20.0, 300.0, 40.0], [50.0, 6.0, 70.0, 8.0]]]], device=device, dtype=dtype)
        self.assert_close(raw_rg, expected)
        self.assert_close(transposed, raw_rg)

    def test_convention_bayer_horizontal_flip_relabels_bg_to_gb(self, device, dtype):
        image = torch.tensor(
            [
                [
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                    [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
                    [[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        raw_bg = kornia.color.rgb_to_raw(image, kornia.color.CFA.BG)
        flipped_gb = kornia.color.rgb_to_raw(image.flip(-1), kornia.color.CFA.GB).flip(-1)

        # CFA.BG is an RGGB sensor: red at (0, 0), blue at (1, 1).
        expected = torch.tensor([[[[1.0, 20.0, 3.0, 40.0], [50.0, 600.0, 70.0, 800.0]]]], device=device, dtype=dtype)
        self.assert_close(raw_bg, expected)
        self.assert_close(flipped_gb, raw_bg)

    def test_convention_sepia_5322_default_keeps_tint(self, device, dtype):
        # Gray maps to intensity * (1.351, 1.203, 0.937). Each image uses its red maximum,
        # independently of the other batch member's intensity.
        intensities = torch.tensor([0.25, 0.5], device=device, dtype=dtype).view(2, 1, 1, 1)
        image = intensities.expand(2, 3, 2, 2)
        coefficients = torch.tensor([1.351, 1.203, 0.937], device=device, dtype=dtype).view(1, 3, 1, 1)
        expected = (intensities * coefficients / (intensities * 1.351 + 1e-6)).expand_as(image)

        out = kornia.color.Sepia()(image)

        self.assert_close(out, expected)
        assert (out[:, 2] < 0.9 * out[:, 0]).all()

    def test_convention_rgb255_scaling_clipping_and_normalization(self, device, dtype):
        # Expected values follow the documented affine maps in rgb.py.
        rgb = torch.tensor([[[[-1.0]], [[0.5]], [[2.0]]]], device=device, dtype=dtype)
        encoded = kornia.color.rgb_to_rgb255(rgb)
        red = torch.tensor([[[[255.0]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        self.assert_close(encoded, torch.tensor([[[[0.0]], [[127.5]], [[255.0]]]], device=device, dtype=dtype))
        # (255, 0, 0) maps to (1, -1, -1) in [-1, 1], then to unit length.
        expected = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype).view(1, 3, 1, 1) / math.sqrt(3.0)
        self.assert_close(kornia.color.rgb255_to_normals(red), expected)

    def test_convention_ycbcr_to_rgb_clips_output(self, device, dtype):
        # Out-of-gamut YCbCr is clipped after conversion: (0, 0, 0) undershoots red and blue below 0,
        # and (1, 1, 1) overshoots them above 1 (unclipped red is 1 + 1.403 * 0.5).
        ycbcr = torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]]], device=device, dtype=dtype)

        rgb = kornia.color.ycbcr_to_rgb(ycbcr)

        self.assert_close(rgb[:, [0, 2], 0, 0], torch.zeros(1, 2, device=device, dtype=dtype))
        self.assert_close(rgb[:, [0, 2], 0, 1], torch.ones(1, 2, device=device, dtype=dtype))

    def test_convention_yuv420_plane_layout(self, device, dtype):
        image = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                    [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]],
                    [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        y, uv = kornia.color.rgb_to_yuv420(image)

        assert y.shape == (1, 1, 2, 4)
        assert uv.shape == (1, 2, 1, 2)
        expected = kornia.color.yuv_to_rgb(torch.cat((y, F.interpolate(uv, scale_factor=2.0, mode="nearest")), dim=-3))
        self.assert_close(kornia.color.yuv420_to_rgb(y, uv), expected)

    def test_convention_apply_colormap_5305_does_not_mutate_rank3_input(self, device, dtype):
        image = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]], device=device, dtype=dtype)
        before = image.clone()
        colormap = kornia.color.ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)

        kornia.color.apply_colormap(image, colormap)

        assert image.shape == before.shape
        self.assert_close(image, before)

    def test_convention_apply_colormap_5305_does_not_mutate_float32_values_or_leaf(self, device):
        # float32 is the dtype whose .float() returns the input itself, so an in-place division would reach it.
        image = torch.tensor([[[[0.0, 255.0]]]], device=device, dtype=torch.float32)
        before = image.clone()
        colormap = kornia.color.ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)

        kornia.color.apply_colormap(image, colormap)
        self.assert_close(image, before)
        leaf = torch.full((1, 1, 1, 1), 0.5, device=device, dtype=torch.float32, requires_grad=True)
        kornia.color.apply_colormap(leaf, colormap)

    def test_convention_apply_colormap_5306_is_batch_independent(self, device, dtype):
        colormap = kornia.color.ColorMap(
            base=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.0, 1.0, 1.0]],
            num_colors=4,
            device=device,
            dtype=dtype,
        )
        sample = torch.tensor([[[[0, 1, 0], [1, 0, 1]]]], device=device, dtype=torch.uint8)
        paired = torch.cat((sample, torch.full_like(sample, 255)), dim=0)

        self.assert_close(
            kornia.color.apply_colormap(paired, colormap)[0],
            kornia.color.apply_colormap(sample, colormap)[0],
        )

    def test_convention_apply_colormap_5307_reaches_last_palette_color(self, device, dtype):
        colormap = kornia.color.ColorMap(
            base=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.0, 1.0, 1.0]],
            num_colors=4,
            device=device,
            dtype=dtype,
        )
        output = kornia.color.apply_colormap(torch.ones((1, 1, 1, 1), device=device, dtype=dtype), colormap)

        self.assert_close(output, torch.ones_like(output))

    def test_convention_luv_5308_black_float16_is_finite(self, device):
        for conversion in (kornia.color.rgb_to_luv, kornia.color.luv_to_rgb):
            black = torch.zeros((1, 3, 1, 1), device=device, dtype=torch.float16, requires_grad=True)
            result = conversion(black)
            result.sum().backward()
            assert torch.isfinite(result).all()
            assert torch.isfinite(black.grad).all()

    def test_convention_apply_colormap_5317_module_to_migrates_palette(self, device):
        colormap = kornia.color.ColorMap(base="viridis", device=device, dtype=torch.float32)
        module = kornia.color.ApplyColorMap(colormap).to(dtype=torch.float16)

        output = module(torch.ones((1, 1, 2, 3), device=device, dtype=torch.float16))

        assert output.dtype == torch.float16
        assert len(module.state_dict()) > 0

    def test_convention_rgb_to_hsv_5309_module_default_matches_function(self, device, dtype):
        image = torch.tensor([[[[1e-6]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        self.assert_close(kornia.color.RgbToHsv()(image), kornia.color.rgb_to_hsv(image))

    def test_convention_rgb_to_raw_5310_rejects_invalid_cfa(self, device, dtype):
        image = torch.arange(18, device=device, dtype=dtype).reshape(1, 3, 2, 3)

        with pytest.raises(ValueError, match="Unsupported CFA"):
            kornia.color.rgb_to_raw(image, "invalid")  # type: ignore[arg-type]

    def test_convention_rgba_to_rgb_5323_rank3_background_keeps_rank(self, device, dtype):
        rgba = torch.rand(4, 2, 2, device=device, dtype=dtype)

        assert kornia.color.rgba_to_rgb(rgba, (0.0, 0.0, 1.0)).shape == (3, 2, 2)

    def test_convention_rgb_to_linear_rgb_5324_gradient_below_minus_0_055_is_finite(self, device, dtype):
        image = torch.full((1, 3, 1, 1), -0.1, device=device, dtype=dtype, requires_grad=True)

        kornia.color.rgb_to_linear_rgb(image).sum().backward()

        assert torch.isfinite(image.grad).all()


def _bound_row(
    device: torch.device, dtype: torch.dtype, values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
) -> torch.Tensor:
    # The three channels of a (1, 3, 1, N) image hold the same row of pixels.
    return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 1, -1).expand(1, 3, 1, -1).contiguous()


def _ycbcr_of_luma(image: torch.Tensor) -> torch.Tensor:
    # At Cb = Cr = 0.5 all three of R, G and B equal Y, so the output sits on a bound where Y does.
    half = torch.full_like(image[:, 0], 0.5)
    return kornia.color.ycbcr_to_rgb(torch.stack([image[:, 0], half, half], 1))


def _hue_on_a_bound(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # H = 5 pi / 3, L = 0.25, S = 1. H * 6 / pi is exactly 10 in the four float dtypes, so min(k - 3, 9 - k) sits on the
    # bound -1 of the clamp for red (k = 10) and blue (k = 2); green (k = 6) has 3, outside the range.
    return torch.tensor([5.0 * math.pi / 3.0, 0.25, 1.0], device=device, dtype=dtype).view(1, 3, 1, 1)


class TestClampGradientAtBounds(BaseTester):
    # A result of these conversions exactly on a bound uses the one-sided derivative from inside the range on every
    # torch version (#6600), the convention of _clamp_unit_range. torch.clamp gives 1 there up to 2.13 and 0 from 2.14,
    # so these pins fail with a plain clamp only on a recent torch; the 2.5.1 leg shows that the two sides agree.
    # A graph captured by torch.jit.trace or torch.export keeps the plain clamp.
    _OPS = {
        # Each pin is the gradient of out.sum() at the pixels 0, 0.25, 0.5, 1.
        # Y moves R, G and B by 1 each, so the gradient is 3 at every pixel, the bounds included.
        "ycbcr_to_rgb": (_ycbcr_of_luma, [3.0, 3.0, 3.0, 3.0]),
        # 255 x has slope 255. Each channel is a separate input, so the sum does not add them up.
        "rgb_to_rgb255": (kornia.color.rgb_to_rgb255, [255.0, 255.0, 255.0, 255.0]),
        # 0.5 (x + 1) * 255 has slope 127.5, and x = 1 lands on the bound 1 before the scale.
        "normals_to_rgb255": (kornia.color.normals_to_rgb255, [127.5, 127.5, 127.5, 127.5]),
    }

    @pytest.mark.parametrize("op", sorted(_OPS))
    def test_convention_result_on_a_bound_keeps_the_derivative_of_the_arithmetic(self, device, dtype, op):
        call, expected = self._OPS[op]
        image = _bound_row(device, dtype).requires_grad_()
        (grad,) = torch.autograd.grad(call(image).sum(), image)
        self.assert_close(grad[:, 0], image.new_tensor(expected).view(1, 1, 4))

    def test_convention_normals_at_the_lower_bound(self, device, dtype):
        # -1 lands on the bound 0, 1 on the bound 1; beyond them the value is the bound and the gradient is zero.
        image = _bound_row(device, dtype, (-2.0, -1.0, 0.0, 1.0, 2.0)).requires_grad_()
        out = kornia.color.normals_to_rgb255(image)
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(out[:, 0], image.new_tensor([0.0, 0.0, 127.5, 255.0, 255.0]).view(1, 1, 5))
        self.assert_close(grad[:, 0], image.new_tensor([0.0, 127.5, 127.5, 127.5, 0.0]).view(1, 1, 5))

    def test_convention_rgb255_outside_the_range_gets_the_bound_and_a_zero_gradient(self, device, dtype):
        image = _bound_row(device, dtype, (-0.5, 1.5)).requires_grad_()
        out = kornia.color.rgb_to_rgb255(image)
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(out[:, 0], image.new_tensor([0.0, 255.0]).view(1, 1, 2))
        assert torch.equal(grad, torch.zeros_like(grad))

    def test_convention_lab_round_trip_keeps_the_identity_gradient(self, device, dtype):
        # The round trip is the identity on grey, so each pixel's channel keeps a gradient of 1 summed over the outputs:
        # black lands exactly on the bound 0 (the issue's 0 on torch 2.15, 1 on 2.5.1).
        image = _bound_row(device, dtype).requires_grad_()
        out = kornia.color.lab_to_rgb(kornia.color.rgb_to_lab(image))
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(grad[:, 0, :, :3], image.new_tensor([1.0, 1.0, 1.0]).view(1, 1, 3))
        # White is not pinned: rounding puts its red just above 1 in float64 (bound value, zero gradient, a sum of 0)
        # and just below 1 in float32 (a sum of 1), on every torch version alike.
        assert torch.isfinite(grad).all()

    def test_convention_lab_black_keeps_the_gradient_of_the_lightness(self, device, dtype):
        # Lab (0, 0, 0) is black, on the bound 0 of the clip. Below the threshold each of X, Y, Z is its white times
        # (f - 4 / 29) / 7.787 with f = (L + 16) / 116, the matrix sends the D65 white to about (1, 1, 1), and
        # sRGB = 12.92 * linear, so d sum / d L = 3.00008 * 12.92 / (116 * 7.787) = 0.0429108.
        image = torch.zeros(1, 3, 1, 1, device=device, dtype=dtype, requires_grad=True)
        out = kornia.color.lab_to_rgb(image)
        assert torch.equal(out, torch.zeros_like(out))
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(grad[:, 0], image.new_tensor([[[0.0429108]]]), rtol=5e-3, atol=1e-6)

    def test_convention_lab_fz_on_its_bound_keeps_the_gradient(self, device, dtype):
        # L = 42, b = 100 gives fz = 58 / 116 - 100 / 200 = 0 exactly, on the bound of the clamp of a negative Z. From
        # inside, d Z / d b = 1.08883 / 7.787 * (-1 / 200) = -6.99130e-4, and each linear channel moves by its Z row of
        # the XYZ matrix (-0.498536, 0.0415559). Through the sRGB slope 1.055 / 2.4 * c^(1 / 2.4 - 1) at the linear
        # values c = 0.202469 and 0.118541, the slopes are 1.116 and 1.525. The tolerance covers float16's spacing here.
        image = torch.tensor([42.0, 0.0, 100.0], device=device, dtype=dtype).view(1, 3, 1, 1).requires_grad_()
        out = kornia.color.lab_to_rgb(image, clip=False)
        (grad_r,) = torch.autograd.grad(out[:, 0].sum(), image, retain_graph=True)
        (grad_g,) = torch.autograd.grad(out[:, 1].sum(), image)
        self.assert_close(grad_r[:, 2], image.new_tensor([[[3.88976e-4]]]), rtol=5e-3, atol=1e-6)
        self.assert_close(grad_g[:, 2], image.new_tensor([[[-4.43074e-5]]]), rtol=5e-3, atol=1e-6)

    def test_convention_hls_hue_on_a_bound_keeps_the_derivative(self, device, dtype):
        # Red is l - a * mink with a = s * min(l, 1 - l) = 0.25 and mink = 9 - k, so d red / d H = a * 6 / pi. Blue has
        # mink = k - 3, so d blue / d H = -a * 6 / pi. Green is outside the range and gets none.
        image = _hue_on_a_bound(device, dtype)
        assert (image[:, 0] * (6 / math.pi)).item() == 10.0  # the precondition: hue exactly on the bound
        image.requires_grad_()
        out = kornia.color.hls_to_rgb(image)
        slope = 0.25 * 6 / math.pi
        for channel, expected in enumerate((slope, 0.0, -slope)):
            (grad,) = torch.autograd.grad(out[:, channel].sum(), image, retain_graph=True)
            self.assert_close(grad[:, 0], image.new_tensor([[[expected]]]))

    @pytest.mark.parametrize("no_grad", [False, True], ids=["recorded", "no_grad"])
    def test_hls_leaves_its_input_alone(self, device, dtype, no_grad):
        # The clamp of hls_to_rgb is in place on a buffer of its own: the input must never see it, nor a view of it.
        image = _hue_on_a_bound(device, dtype).expand(2, 3, 4, 5).contiguous()
        before = image.clone()
        with torch.set_grad_enabled(not no_grad):
            kornia.color.hls_to_rgb(image.requires_grad_(not no_grad))
        assert torch.equal(image, before)

    def test_hls_without_a_derivative_reuses_its_buffer(self, device, dtype):
        # One image-sized temporary fewer per call: the clamp stays in place unless a derivative can be asked for.
        image = _hue_on_a_bound(device, dtype).expand(1, 3, 4, 5).contiguous()
        calls = []
        original = torch.Tensor.clamp_

        def spy(self: torch.Tensor, *args: float, **kwargs: float) -> torch.Tensor:
            calls.append(self)
            return original(self, *args, **kwargs)

        with patch.object(torch.Tensor, "clamp_", spy):
            with torch.no_grad():
                unrecorded = kornia.color.hls_to_rgb(image.clone().requires_grad_())
            assert len(calls) == 1
            plain = kornia.color.hls_to_rgb(image)
            assert len(calls) == 2
            recorded = kornia.color.hls_to_rgb(image.clone().requires_grad_())
            assert len(calls) == 2
        assert torch.equal(unrecorded, plain)
        assert torch.equal(unrecorded, recorded.detach())

    _SITES = {
        "ycbcr_to_rgb": kornia.color.ycbcr_to_rgb,
        "lab_to_rgb": kornia.color.lab_to_rgb,
        "rgb_to_rgb255": kornia.color.rgb_to_rgb255,
        "normals_to_rgb255": kornia.color.normals_to_rgb255,
        "hls_to_rgb": kornia.color.hls_to_rgb,
    }

    @pytest.mark.parametrize("op", sorted(_SITES))
    def test_recorded_values_match_unrecorded_ones(self, device, dtype, op):
        # Out of range on purpose: the selection returns what the clamp returns, NaN included.
        values = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 255.0, float("nan"), float("inf"), -float("inf"))
        image = _bound_row(device, dtype, values)
        with torch.no_grad():
            expected = self._SITES[op](image)
        recorded = self._SITES[op](image.clone().requires_grad_())
        assert torch.equal(torch.isnan(recorded), torch.isnan(expected))
        finite = ~torch.isnan(expected)
        assert torch.equal(recorded[finite].detach(), expected[finite])

    @pytest.mark.parametrize("forward_mode", ["jvp", "dual"])
    @pytest.mark.parametrize("grad_enabled", [False, True])
    @pytest.mark.parametrize("op", sorted(_OPS))
    def test_forward_mode_keeps_the_tangent_at_the_bounds(self, device, dtype, op, forward_mode, grad_enabled):
        call, expected = self._OPS[op]
        image = _bound_row(device, dtype)
        tangent = torch.zeros_like(image)
        tangent[:, 0] = 1.0
        with torch.set_grad_enabled(grad_enabled):
            if forward_mode == "jvp":
                _, actual = torch.func.jvp(call, (image,), (tangent,))
            else:
                with torch.autograd.forward_ad.dual_level():
                    dual = torch.autograd.forward_ad.make_dual(image, tangent)
                    _, actual = torch.autograd.forward_ad.unpack_dual(call(dual))
        self.assert_close(actual.sum(1), image.new_tensor(expected).view(1, 1, 4))

    @pytest.mark.parametrize("forward_mode", ["jvp", "dual"])
    @pytest.mark.parametrize("grad_enabled", [False, True])
    def test_forward_mode_hls_keeps_the_tangent_at_the_bounds(self, device, dtype, forward_mode, grad_enabled):
        # The tangent of the hue gives the column of the derivative pinned above, under no_grad too.
        image = _hue_on_a_bound(device, dtype)
        tangent = torch.zeros_like(image)
        tangent[:, 0] = 1.0
        with torch.set_grad_enabled(grad_enabled):
            if forward_mode == "jvp":
                _, actual = torch.func.jvp(kornia.color.hls_to_rgb, (image,), (tangent,))
            else:
                with torch.autograd.forward_ad.dual_level():
                    dual = torch.autograd.forward_ad.make_dual(image, tangent)
                    _, actual = torch.autograd.forward_ad.unpack_dual(kornia.color.hls_to_rgb(dual))
        slope = 0.25 * 6 / math.pi
        self.assert_close(actual.flatten(), image.new_tensor([slope, 0.0, -slope]))

    @pytest.mark.parametrize("op", ["rgb_to_rgb255", "normals_to_rgb255"])
    @pytest.mark.parametrize("int_dtype", [torch.uint8, torch.int64])
    def test_integer_image_keeps_its_result_dtype(self, device, op, int_dtype):
        # The float bounds promote an integer result to the default float dtype, with or without the selection.
        image = torch.tensor([0, 1, 2], device=device, dtype=int_dtype).view(1, 3, 1, 1)
        out = self._SITES[op](image)
        expected = (
            (image * 255).clip(0.0, 255.0) if op == "rgb_to_rgb255" else (0.5 * (image + 1.0)).clip(0.0, 1.0) * 255
        )
        assert out.dtype == expected.dtype
        assert torch.equal(out, expected)

    @pytest.mark.parametrize("name", sorted(_SITES))
    def test_scripted_call_keeps_the_gradient_at_the_bounds(self, device, dtype, name):
        scripted = torch.jit.script(self._SITES[name])
        if name == "hls_to_rgb":
            image = _hue_on_a_bound(device, dtype).requires_grad_()
            slope = 0.25 * 6 / math.pi
            (grad,) = torch.autograd.grad(scripted(image)[:, 0].sum(), image)
            self.assert_close(grad[:, 0], image.new_tensor([[[slope]]]))
            return
        if name == "lab_to_rgb":
            image = torch.zeros(1, 3, 1, 1, device=device, dtype=dtype, requires_grad=True)
            (grad,) = torch.autograd.grad(scripted(image).sum(), image)
            self.assert_close(grad[:, 0], image.new_tensor([[[0.0429108]]]), rtol=5e-3, atol=1e-6)
            return
        image = _bound_row(device, dtype)
        if name == "ycbcr_to_rgb":
            # Y on the bounds, neutral chroma.
            image[:, 1:] = 0.5
            expected = image.new_tensor([3.0, 3.0, 3.0, 3.0]).view(1, 1, 4)
        else:
            expected = image.new_tensor([self._OPS[name][1]]).view(1, 1, 4)
        image.requires_grad_()
        (grad,) = torch.autograd.grad(scripted(image).sum(), image)
        self.assert_close(grad[:, 0], expected)

    @pytest.mark.parametrize("op", sorted(_SITES))
    def test_trace_of_a_module_with_parameters_passes_its_check(self, device, dtype, op):
        # A parameter gives the op an input that requires grad, and torch.jit.trace checks its graph again under
        # no_grad: a clamp that followed the grad mode would record two different graphs and raise TracingCheckError.
        call = self._SITES[op]

        class ConvThenOp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
                torch.nn.init.dirac_(self.conv.weight)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return call(self.conv(x))

        module = ConvThenOp().to(device, dtype)
        image = _bound_row(device, dtype)
        traced = torch.jit.trace(module, (image,))
        self.assert_close(traced(image), module(image))
        graph = str(traced.inlined_graph)
        # The selection compares the clamped copy with its input; nothing else in these conversions tests for equality.
        assert "aten::eq" not in graph
        assert "aten::clamp" in graph

    @pytest.mark.parametrize("op", sorted(_OPS))
    def test_dynamo_gradient_at_the_bounds(self, device, dtype, op, torch_optimizer):
        call, expected = self._OPS[op]
        image = _bound_row(device, dtype).requires_grad_()
        (grad,) = torch.autograd.grad(torch_optimizer(call, fullgraph=True)(image).sum(), image)
        self.assert_close(grad[:, 0], image.new_tensor(expected).view(1, 1, 4))

    def test_dynamo_gradient_of_the_hue_at_a_bound(self, device, dtype, torch_optimizer):
        image = _hue_on_a_bound(device, dtype).requires_grad_()
        out = torch_optimizer(kornia.color.hls_to_rgb, fullgraph=True)(image)
        (grad,) = torch.autograd.grad(out[:, 0].sum(), image)
        self.assert_close(grad[:, 0], image.new_tensor([[[0.25 * 6 / math.pi]]]))

    def test_helper_keeps_the_derivative_on_both_bounds(self, device, dtype):
        image = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
        out = _clamp_range(image, -1.0, 1.0)
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(out, image.new_tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))
        self.assert_close(grad, image.new_tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

    def test_helper_without_an_upper_bound(self, device, dtype):
        image = torch.tensor([-2.0, 0.0, 3.0, float("inf")], device=device, dtype=dtype, requires_grad=True)
        out = _clamp_range(image, 0.0)
        (grad,) = torch.autograd.grad(out.sum(), image)
        self.assert_close(out[:3], image.new_tensor([0.0, 0.0, 3.0]))
        assert out[3].item() == float("inf")
        self.assert_close(grad, image.new_tensor([0.0, 1.0, 1.0, 1.0]))

    def test_helper_records_no_clamp(self, device, dtype):
        # The selection takes the clamp only outside the range, where the gradient is zero anyway, and the backward of a
        # recorded clamp made a forward and backward pass 1.7 to 2.5 times as slow on the CPU (#6594).
        image = torch.tensor([-2.0, 0.0, 3.0], device=device, dtype=dtype, requires_grad=True)
        recorded = {fn.name() for fn, _ in _clamp_range(image, 0.0, 1.0).grad_fn.next_functions if fn is not None}
        assert not any("Clamp" in name for name in recorded)
