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

import pytest
import torch
import torch.nn.functional as F

import kornia

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

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5306")
    def test_wart_apply_colormap_5306_is_batch_independent(self, device, dtype):
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
