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

import kornia

from testing.base import BaseTester


class TestPyrUp(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.pyrup, (img,), nondet_tol=1e-8)

    def test_convention_align_corners_and_border_type_change_output(self, device, dtype):
        # pyrup's align_corners (default False) and border_type (default 'reflect') defaults
        # actually change the output values -- existing tests only check output shape. pyrup is
        # an independent implementation (interpolate-then-blur, no delegation to pyrdown), so the
        # sibling pin on TestPyrDown gives this op no coverage on its own.
        x = torch.arange(0.0, 16.0, device=device, dtype=dtype).view(1, 1, 4, 4)

        out_ac_false = kornia.geometry.transform.pyrup(x, align_corners=False)
        out_ac_true = kornia.geometry.transform.pyrup(x, align_corners=True)
        out_default = kornia.geometry.transform.pyrup(x)
        self.assert_close(out_default, out_ac_false, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_ac_false, out_ac_true, atol=1e-2, rtol=1e-2)

        out_reflect = kornia.geometry.transform.pyrup(x, border_type="reflect")
        out_constant = kornia.geometry.transform.pyrup(x, border_type="constant")
        self.assert_close(out_default, out_reflect, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_reflect, out_constant, atol=1e-2, rtol=1e-2)


class TestPyrDown(BaseTester):
    def test_shape(self, device, dtype):
        inp = torch.zeros(1, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_custom_factor(self, device, dtype):
        inp = torch.zeros(1, 2, 9, 9, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown(factor=3.0)
        assert pyr(inp).shape == (1, 2, 3, 3)

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(2, 2, 4, 4, device=device, dtype=dtype)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_symmetry_preserving(self, device, dtype):
        inp = torch.zeros(1, 1, 6, 6, device=device, dtype=dtype)
        inp[:, :, 2:4, 2:4] = 1.0
        pyr_out = kornia.geometry.PyrDown()(inp).squeeze()
        self.assert_close(pyr_out, pyr_out.flip(0))
        self.assert_close(pyr_out, pyr_out.flip(1))

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 5, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.pyrdown, (img,), nondet_tol=1e-8)

    def test_convention_align_corners_and_border_type_change_output(self, device, dtype):
        # pyrdown's align_corners (default False) and border_type (default 'reflect') defaults
        # actually change the output values -- existing tests only check output shape, never a
        # discriminating-literal comparison of the defaults against their alternatives.
        x = torch.arange(0.0, 25.0, device=device, dtype=dtype).view(1, 1, 5, 5)

        out_ac_false = kornia.geometry.transform.pyrdown(x, align_corners=False)
        out_ac_true = kornia.geometry.transform.pyrdown(x, align_corners=True)
        out_default = kornia.geometry.transform.pyrdown(x)
        self.assert_close(out_default, out_ac_false, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_ac_false, out_ac_true, atol=1e-2, rtol=1e-2)

        out_reflect = kornia.geometry.transform.pyrdown(x, border_type="reflect")
        out_constant = kornia.geometry.transform.pyrdown(x, border_type="constant")
        self.assert_close(out_default, out_reflect, rtol=1e-2, atol=1e-2)
        assert not torch.allclose(out_reflect, out_constant, atol=1e-2, rtol=1e-2)

    def test_convention_floor_not_ceil_on_odd_size(self, device, dtype):
        # pyrdown uses floor(side / factor), diverging from OpenCV's ceil((side + 1) / 2) on
        # odd/non-exactly-divisible sizes: 5x5 at the default factor=2.0 gives 2x2, not 3x3.
        # (test_shape/test_shape_custom_factor only use exactly-divisible sizes, where floor and
        # ceil agree.)
        x = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        assert kornia.geometry.transform.pyrdown(x).shape == (1, 1, 2, 2)


class TestScalePyramid(BaseTester):
    def test_shape_tuple(self, device, dtype):
        inp = torch.zeros(3, 2, 41, 41, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1, min_size=30)
        out = SP(inp)
        assert len(out) == 3
        assert len(out[0]) == 1
        assert len(out[1]) == 1
        assert len(out[2]) == 1

    def test_shape_batch(self, device, dtype):
        inp = torch.zeros(3, 2, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (3, 2, 3 + 1, 31, 31)

    def test_shape_batch_double(self, device, dtype):
        inp = torch.zeros(3, 2, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=1, double_image=True)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (3, 2, 1 + 3, 62, 62)

    def test_n_levels_shape(self, device, dtype):
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        assert sp[0].shape == (1, 1, 3 + 3, 32, 32)

    def test_blur_order(self, device, dtype):
        inp = torch.rand(1, 1, 31, 31, device=device, dtype=dtype)
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        for _, pyr_level in enumerate(sp):
            for _, img in enumerate(pyr_level):
                img = img.squeeze().view(3, -1)
                max_per_blur_level_val, _ = img.max(dim=1)
                assert torch.argmax(max_per_blur_level_val).item() == 0

    def test_symmetry_preserving(self, device, dtype):
        PS = 16
        R = 2
        inp = torch.zeros(1, 1, PS, PS, device=device, dtype=dtype)
        inp[..., PS // 2 - R : PS // 2 + R, PS // 2 - R : PS // 2 + R] = 1.0
        SP = kornia.geometry.ScalePyramid(n_levels=3)
        sp, _, _ = SP(inp)
        for _, pyr_level in enumerate(sp):
            for _, img in enumerate(pyr_level):
                img = img.squeeze()
                self.assert_close(img, img.flip(1))
                self.assert_close(img, img.flip(2))

    def test_gradcheck(self, device):
        img = torch.rand(1, 2, 7, 9, device=device, dtype=torch.float64)
        from kornia.geometry import ScalePyramid as SP

        def sp_tuple(img):
            sp, _, _ = SP()(img)
            return tuple(sp)

        self.gradcheck(sp_tuple, (img,), nondet_tol=1e-4)

    def test_convention_octave0_sigmas_example(self, device, dtype):
        # Pins the Convention block's worked example: with n_levels=1 (default
        # extra_levels=3), init_sigma=0.25 below the assumed input blur (0.5) makes
        # octave 0's first sigma the input blur itself, while octave 1+ starts from
        # init_sigma as usual -- the two octaves diverge only in that first entry.
        inp = torch.rand(1, 1, 32, 32, device=device, dtype=dtype)
        sp = kornia.geometry.ScalePyramid(n_levels=1, init_sigma=0.25)
        _, sigmas, _ = sp(inp)
        # Snippet used to generate expected (requires only this module):
        # sp = kornia.geometry.ScalePyramid(n_levels=1, init_sigma=0.25)
        # _, sigmas, _ = sp(torch.rand(1, 1, 32, 32))
        # sigmas[0][0].tolist(), sigmas[1][0].tolist()
        expected_octave0 = torch.tensor([0.5, 0.5, 1.0, 2.0], device=device, dtype=dtype)
        expected_octave1 = torch.tensor([0.25, 0.5, 1.0, 2.0], device=device, dtype=dtype)
        self.assert_close(sigmas[0][0], expected_octave0, rtol=1e-3, atol=1e-3)
        self.assert_close(sigmas[1][0], expected_octave1, rtol=1e-3, atol=1e-3)


class TestBuildPyramid(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.ones(1, 2, 4, 5, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_pyramid(sample, max_level=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (1, 2, 4, 5)

    @pytest.mark.parametrize("batch_size", (1, 2, 3))
    @pytest.mark.parametrize("channels", (1, 3))
    @pytest.mark.parametrize("max_level", (2, 3, 4))
    def test_num_levels(self, batch_size, channels, max_level, device, dtype):
        height, width = 16, 20
        sample = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_pyramid(sample, max_level)
        assert len(pyramid) == max_level
        for i in range(1, max_level):
            img = pyramid[i]
            denom = 2**i
            expected_shape = (batch_size, channels, height // denom, width // denom)
            assert img.shape == expected_shape

    def test_gradcheck(self, device):
        max_level = 1
        batch_size, channels, height, width = 1, 2, 7, 9
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.build_pyramid, (img, max_level))


class TestBuildLaplacianPyramid(BaseTester):
    def test_smoke(self, device, dtype):
        sample = torch.ones(1, 2, 4, 5, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (1, 2, 4, 5)

    @pytest.mark.parametrize("height,width", ((5, 8), (8, 5)))
    def test_mixed_power_of_two_size_padding(self, height, width, device, dtype):
        sample = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level=2)
        assert [img.shape for img in pyramid] == [torch.Size((1, 2, 8, 8)), torch.Size((1, 2, 4, 4))]

    @pytest.mark.parametrize("batch_size", (1, 2, 3))
    @pytest.mark.parametrize("channels", (1, 3))
    @pytest.mark.parametrize("max_level", (2, 3, 4))
    def test_num_levels(self, batch_size, channels, max_level, device, dtype):
        height, width = 16, 32
        sample = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        pyramid = kornia.geometry.transform.build_laplacian_pyramid(sample, max_level)
        assert len(pyramid) == max_level
        for i in range(1, max_level):
            img = pyramid[i]
            denom = 2**i
            expected_shape = (batch_size, channels, height // denom, width // denom)
            assert img.shape == expected_shape

    def test_gradcheck(self, device):
        max_level = 1
        batch_size, channels, height, width = 1, 2, 7, 9
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.build_laplacian_pyramid, (img, max_level), nondet_tol=1e-8)
