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
        # rgb_to_xyz is a matrix transform; sRGB transfer conversion is an explicit prior step.
        srgb = torch.full((1, 3, 1, 1), 0.5, device=device, dtype=dtype)

        direct = kornia.color.rgb_to_xyz(srgb)
        linear = kornia.color.rgb_to_xyz(kornia.color.rgb_to_linear_rgb(srgb))

        assert not torch.allclose(direct, linear)
        self.assert_close(kornia.color.xyz_to_rgb(linear), kornia.color.rgb_to_linear_rgb(srgb))

    def test_convention_xyz_linear_rgb_matrix(self, device, dtype):
        image = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        xyz = kornia.color.rgb_to_xyz(image)

        expected = torch.tensor([[[[0.412453]], [[0.212671]], [[0.019334]]]], device=device, dtype=dtype)
        self.assert_close(xyz, expected)

    def test_convention_grayscale_to_rgb_is_expanded_view(self, device, dtype):
        image = torch.tensor([[[[0.2, 0.7]]]], device=device, dtype=dtype)

        rgb = kornia.color.grayscale_to_rgb(image)

        assert rgb.shape == (1, 3, 1, 2)
        assert rgb.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()

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

        self.assert_close(flipped_gb, raw_bg)

    def test_convention_sepia_rescale_is_per_channel_spatial(self, device, dtype):
        image = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]]], device=device, dtype=dtype)

        rescaled = kornia.color.sepia_from_rgb(image, rescale=True)

        self.assert_close(rescaled.amax(dim=(-2, -1)), torch.ones((1, 3), device=device, dtype=dtype))

    def test_convention_rgb255_scaling_clipping_and_normalization(self, device, dtype):
        # Expected values follow the documented affine maps in rgb.py.
        rgb = torch.tensor([[[[-1.0]], [[0.5]], [[2.0]]]], device=device, dtype=dtype)
        encoded = kornia.color.rgb_to_rgb255(rgb)
        white = torch.full((1, 3, 1, 1), 255.0, device=device, dtype=dtype)

        self.assert_close(encoded, torch.tensor([[[[0.0]], [[127.5]], [[255.0]]]], device=device, dtype=dtype))
        self.assert_close(kornia.color.rgb255_to_normals(white), torch.full_like(white, 1.0 / math.sqrt(3.0)))

    def test_convention_ycbcr_to_rgb_clips_output(self, device, dtype):
        # Out-of-gamut YCbCr is clipped after conversion by ycbcr_to_rgb.
        ycbcr = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        rgb = kornia.color.ycbcr_to_rgb(ycbcr)

        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

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

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5305")
    def test_wart_apply_colormap_5305_does_not_mutate_rank3_input(self, device, dtype):
        image = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]], device=device, dtype=dtype)
        before = image.clone()
        colormap = kornia.color.ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)

        kornia.color.apply_colormap(image, colormap)

        assert image.shape == before.shape
        self.assert_close(image, before)

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5305")
    def test_wart_apply_colormap_5305_does_not_mutate_rank4_values_or_leaf(self, device, dtype):
        image = torch.tensor([[[[0.0, 255.0]]]], device=device, dtype=dtype)
        before = image.clone()
        colormap = kornia.color.ColorMap(base=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)

        kornia.color.apply_colormap(image, colormap)
        self.assert_close(image, before)
        leaf = torch.full((1, 1, 1), 0.5, device=device, dtype=dtype, requires_grad=True)
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

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5307")
    def test_wart_apply_colormap_5307_reaches_last_palette_color(self, device, dtype):
        colormap = kornia.color.ColorMap(
            base=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [1.0, 1.0, 1.0]],
            num_colors=4,
            device=device,
            dtype=dtype,
        )
        output = kornia.color.apply_colormap(torch.ones((1, 1, 1, 1), device=device, dtype=dtype), colormap)

        self.assert_close(output, torch.ones_like(output))

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5308")
    def test_wart_luv_5308_black_float16_is_finite(self, device):
        for conversion in (kornia.color.rgb_to_luv, kornia.color.luv_to_rgb):
            black = torch.zeros((1, 3, 1, 1), device=device, dtype=torch.float16, requires_grad=True)
            result = conversion(black)
            result.sum().backward()
            assert torch.isfinite(result).all()
            assert torch.isfinite(black.grad).all()

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5317")
    def test_wart_apply_colormap_5317_module_to_migrates_palette(self, device):
        colormap = kornia.color.ColorMap(base="viridis", device=device, dtype=torch.float32)
        module = kornia.color.ApplyColorMap(colormap).to(dtype=torch.float16)

        output = module(torch.ones((1, 1, 2, 3), device=device, dtype=torch.float16))

        assert module.colormap.colors.dtype == torch.float16
        assert output.dtype == torch.float16

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5309")
    def test_wart_rgb_to_hsv_5309_module_default_matches_function(self, device, dtype):
        image = torch.tensor([[[[1e-6]], [[0.0]], [[0.0]]]], device=device, dtype=dtype)

        self.assert_close(kornia.color.RgbToHsv()(image), kornia.color.rgb_to_hsv(image))

    @pytest.mark.xfail(strict=True, reason="https://github.com/kornia/kornia/issues/5310")
    def test_wart_rgb_to_raw_5310_rejects_invalid_cfa(self, device, dtype):
        image = torch.arange(18, device=device, dtype=dtype).reshape(1, 3, 2, 3)

        with pytest.raises((TypeError, ValueError)):
            kornia.color.rgb_to_raw(image, "invalid")  # type: ignore[arg-type]
