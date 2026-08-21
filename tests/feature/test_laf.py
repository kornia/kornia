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
import kornia.geometry.transform.imgwarp

from testing.base import BaseTester
from testing.geometry.create import create_random_homography


class TestAngleToRotationMatrix(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4).to(device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self, device):
        ang_deg = torch.tensor([0, 90.0], device=device)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0, 1.0], [-1.0, 0]]], device=device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(ang_deg)
        self.assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix, (img,))

    @pytest.mark.skip("Problems with kornia.pi")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix
        model_jit = torch.jit.script(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix)
        self.assert_close(model(patches), model_jit(patches))


class TestGetLAFScale(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        rotmat = kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0], [1, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2]]]], device=device).float()
        rotmat = kornia.feature.get_laf_scale(inp)
        self.assert_close(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_scale, (img,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_scale
        model_jit = torch.jit.script(kornia.feature.get_laf_scale)
        self.assert_close(model(img), model_jit(img))


class TestGetLAFCenter(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        xy = kornia.feature.get_laf_center(inp)
        assert xy.shape == (1, 3, 2)

    def test_center(self, device):
        inp = torch.tensor([[5.0, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[2, 3]]], device=device).float()
        xy = kornia.feature.get_laf_center(inp)
        self.assert_close(xy, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_center, (img,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_center
        model_jit = torch.jit.script(kornia.feature.get_laf_center)
        self.assert_close(model(img), model_jit(img))


class TestGetLAFOri(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        ori = kornia.feature.get_laf_orientation(inp)
        assert ori.shape == (1, 3, 1)

    def test_ori(self, device):
        inp = torch.tensor([[1, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[45.0]]], device=device).float()
        angle = kornia.feature.get_laf_orientation(inp)
        self.assert_close(angle, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.get_laf_orientation, (img,))

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.get_laf_orientation
        model_jit = torch.jit.script(kornia.feature.get_laf_orientation)
        self.assert_close(model(img), model_jit(img))


class TestScaleLAF(BaseTester):
    def test_shape_float(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = 23.0
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = torch.zeros(7, 1, 1, 1, device=device).float()
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_scale(self, device):
        inp = torch.tensor([[5.0, 1, 0.8], [1, 1, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        scale = torch.tensor([[[[2.0]]]], device=device).float()
        out = kornia.feature.scale_laf(inp, scale)
        expected = torch.tensor([[[[10.0, 2, 0.8], [2, 2, -4.0]]]], device=device).float()
        self.assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        scale = torch.rand(batch_size, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.scale_laf, (laf, scale), atol=1e-4)

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        scale = torch.rand(batch_size, device=device)
        model = kornia.feature.scale_laf
        model_jit = torch.jit.script(kornia.feature.scale_laf)
        self.assert_close(model(laf, scale), model_jit(laf, scale))


class TestSetLAFOri(BaseTester):
    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        ori = torch.ones(7, 3, 1, 1, device=device).float()
        assert kornia.feature.set_laf_orientation(inp, ori).shape == inp.shape

    def test_ori(self, device):
        inp = torch.tensor([[0.0, 5.0, 0.8], [-5.0, 0, -4.0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        ori = torch.zeros(1, 1, 1, 1, device=device).float()
        out = kornia.feature.set_laf_orientation(inp, ori)
        expected = torch.tensor([[[[5.0, 0.0, 0.8], [0.0, 5.0, -4.0]]]], device=device).float()
        self.assert_close(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        ori = torch.rand(batch_size, channels, 1, 1, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.set_laf_orientation, (laf, ori), atol=1e-4)

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        ori = torch.rand(batch_size, channels, 1, 1, device=device)
        model = kornia.feature.set_laf_orientation
        model_jit = torch.jit.script(kornia.feature.set_laf_orientation)
        self.assert_close(model(laf, ori), model_jit(laf, ori))


class TestMakeUpright(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        rotmat = kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self, device):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[1, 0, 0], [0, 1, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        self.assert_close(laf, expected)

    def test_do_nothing_with_scalea(self, device):
        inp = torch.tensor([[2, 0, 0], [0, 2, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2, 0, 0], [0, 2, 0]]]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        self.assert_close(laf, expected)

    def test_check_zeros(self, device):
        inp = torch.rand(4, 5, 2, 3, device=device)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:, :, 0, 1]
        self.assert_close(must_be_zeros, torch.zeros_like(must_be_zeros))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 14, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.make_upright, (img,))

    @pytest.mark.skip("Union")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.make_upright
        model_jit = torch.jit.script(kornia.feature.make_upright)
        self.assert_close(model(img), model_jit(img))


class TestELL2LAF(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(5, 3, 5, device=device)
        inp[:, :, 3] = 0
        rotmat = kornia.feature.ellipse_to_laf(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self, device):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]], device=device).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.0], [0, 10, -20]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ellipse_to_laf(inp)
        self.assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device, dtype=torch.float64).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        # assure it is positive definite
        self.gradcheck(kornia.feature.ellipse_to_laf, (img,))

    def test_jit(self, device, dtype):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.0
        model = kornia.feature.ellipse_to_laf
        model_jit = torch.jit.script(kornia.feature.ellipse_to_laf)
        self.assert_close(model(img), model_jit(img))


class TestNormalizeLAF(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.normalize_laf(inp, img).shape

    def test_roundtrip_non_square_wide(self, device, dtype):
        # Wide image (W >> H): x/y coords are normalized differently from scale components.
        # The normalize→denormalize round-trip must be exact.
        laf = torch.tensor([[10.0, 0.0, 160.0], [0.0, 10.0, 60.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        img = torch.zeros(1, 1, 120, 320, device=device, dtype=dtype)
        laf_norm = kornia.feature.normalize_laf(laf, img)
        laf_back = kornia.feature.denormalize_laf(laf_norm, img)
        self.assert_close(laf_back, laf)

    def test_roundtrip_non_square_tall(self, device, dtype):
        # Tall image (H >> W): verify round-trip in the opposite aspect ratio.
        laf = torch.tensor([[5.0, 0.0, 40.0], [0.0, 5.0, 100.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        img = torch.zeros(1, 1, 240, 80, device=device, dtype=dtype)
        laf_norm = kornia.feature.normalize_laf(laf, img)
        laf_back = kornia.feature.denormalize_laf(laf_norm, img)
        self.assert_close(laf_back, laf)

    def test_conversion(self, device):
        w, h = 9, 5
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        expected = torch.tensor([[[[0.25, 0, 0.125], [0, 0.25, 0.25]]]]).float()
        lafn = kornia.feature.normalize_laf(laf, img)
        self.assert_close(lafn, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        img = torch.rand(batch_size, 3, 10, 32, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.normalize_laf, (laf, img))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.normalize_laf
        model_jit = torch.jit.script(kornia.feature.normalize_laf)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestLAF2pts(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        n_pts = 13
        assert kornia.feature.laf_to_boundary_points(inp, n_pts).shape == (5, 3, n_pts, 2)

    def test_conversion(self, device):
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        laf = laf.view(1, 1, 2, 3)
        n_pts = 6
        expected = torch.tensor([[[[1, 1], [1, 2], [2, 1], [1, 0], [0, 1], [1, 2]]]], device=device).float()
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        self.assert_close(pts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_to_boundary_points, (laf))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_boundary_points
        model_jit = torch.jit.script(kornia.feature.laf_to_boundary_points)
        self.assert_close(model(laf), model_jit(laf))


class TestDenormalizeLAF(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert inp.shape == kornia.feature.denormalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 9, 5
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w, device=device)
        lafn = torch.tensor([[0.25, 0, 0.125], [0, 0.25, 0.25]], device=device).float()
        laf = kornia.feature.denormalize_laf(lafn.view(1, 1, 2, 3), img)
        self.assert_close(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        img = torch.rand(batch_size, 3, 10, 32, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.denormalize_laf, (laf, img))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        model = kornia.feature.denormalize_laf
        model_jit = torch.jit.script(kornia.feature.denormalize_laf)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestGenPatchGrid(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        grid = generate_patch_grid_from_normalized_LAF(img, laf, PS)
        assert grid.shape == (15, 3, 3, 2)

    def test_gradcheck(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device, dtype=torch.float64)
        img = torch.rand(5, 3, 10, 10, device=device, dtype=torch.float64)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF

        self.gradcheck(generate_patch_grid_from_normalized_LAF, (img, laf, PS))


class TestClampGridToPixelCenters(BaseTester):
    """Pin the MPS ``padding_mode="border"`` emulation used by the patch extractors.

    MPS has no ``border`` padding for ``grid_sample``, so the extractors clamp the grid and use
    zero padding instead. The clamp must target the outermost pixel *center*; clamping to
    :math:`\\pm 1` lands on the outer *edge* of the border pixel and halves its value.

    The reference is always computed on CPU, because ``padding_mode="border"`` is exactly what the
    target device may not support.
    """

    @staticmethod
    def _make(h, w, dtype):
        torch.manual_seed(0)
        img = torch.rand(1, 2, h, w, dtype=dtype)
        # a grid that deliberately runs off every edge
        grid = (torch.rand(1, 9, 11, 2, dtype=dtype) * 3.0) - 1.5
        return img, grid

    def test_matches_border_padding(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("grid_sample reference comparison is meaningful only in full precision")
        h, w = 17, 23
        img, grid = self._make(h, w, dtype)
        expected = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=False)

        clamped = kornia.feature.laf._clamp_grid_to_pixel_centers(grid.to(device), h, w)
        actual = torch.nn.functional.grid_sample(img.to(device), clamped, padding_mode="zeros", align_corners=False)
        self.assert_close(actual.cpu(), expected)

    def test_naive_clamp_is_not_equivalent(self, device, dtype):
        """Guard against regressing to ``grid.clamp(-1, 1)``, which is off by half a pixel."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("grid_sample reference comparison is meaningful only in full precision")
        h, w = 17, 23
        img, grid = self._make(h, w, dtype)
        expected = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=False)
        naive = torch.nn.functional.grid_sample(
            img.to(device), grid.clamp(-1, 1).to(device), padding_mode="zeros", align_corners=False
        )
        assert (naive.cpu() - expected).abs().max() > 1e-3

    def test_shape(self, device, dtype):
        grid = torch.rand(4, 5, 6, 2, device=device, dtype=dtype)
        assert kornia.feature.laf._clamp_grid_to_pixel_centers(grid, 8, 9).shape == grid.shape

    def test_gradcheck(self, device):
        grid = ((torch.rand(1, 3, 3, 2, device=device, dtype=torch.float64) * 3.0) - 1.5).requires_grad_()
        self.gradcheck(lambda g: kornia.feature.laf._clamp_grid_to_pixel_centers(g, 7, 11), (grid,), fast_mode=False)

    @pytest.mark.parametrize(
        "extractor",
        [kornia.feature.extract_patches_simple, kornia.feature.extract_patches_from_pyramid],
    )
    def test_extractors_match_cpu_at_the_border(self, device, dtype, extractor):
        """Patches overlapping the image border must not depend on the device."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("cross-device comparison is meaningful only in full precision")
        torch.manual_seed(0)
        img = torch.rand(1, 1, 64, 64, dtype=dtype)
        # centered near the corner with a large scale, so the patch runs off two edges
        laf = kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[3.0, 3.0]]], dtype=dtype), torch.full((1, 1, 1, 1), 12.0, dtype=dtype)
        )
        expected = extractor(img, laf, 32)
        actual = extractor(img.to(device), laf.to(device), 32)
        self.assert_close(actual.cpu(), expected)


class TestExtractPatchesSimple(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 5, 1.0)
        self.assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_simple(img, laf, 4, 1.0)
        self.assert_close(img, patch[0])

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device, dtype=torch.float64)
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device, dtype=torch.float64)
        PS = 11
        self.gradcheck(kornia.feature.extract_patches_simple, (img, nlaf, PS, False), fast_mode=False)


class TestExtractPatchesPyr(BaseTester):
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_non_zero(self, device):
        img = torch.zeros(1, 1, 24, 24, device=device)
        img[:, :, 10:, 20:] = 1.0
        laf = torch.tensor([[8.0, 0, 14.0], [0, 8.0, 8.0]], device=device).reshape(1, 1, 2, 3)

        PS = 32
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.mean().item() > 0.01
        assert patches.shape == (1, 1, 1, PS, PS)

    def test_same_odd(self, device, dtype):
        img = torch.arange(5)[None].repeat(5, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[2.0, 0, 2.0], [0, 2.0, 2.0]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 5, 1.0)
        self.assert_close(img, patch[0])

    def test_same_even(self, device, dtype):
        img = torch.arange(4)[None].repeat(4, 1)[None, None].to(device, dtype)
        laf = torch.tensor([[1.5, 0, 1.5], [0, 1.5, 1.5]]).reshape(1, 1, 2, 3).to(device, dtype)

        patch = kornia.feature.extract_patches_from_pyramid(img, laf, 4, 1.0)
        self.assert_close(img, patch[0])

    def test_small_image_single_level(self, device, dtype):
        # When min(H, W) < 2 * PS, the pyramid cannot descend beyond level 0.
        # All patches must still have the correct shape and non-zero content.
        PS = 16
        img = torch.rand(1, 1, 24, 24, device=device, dtype=dtype)  # 24 < 2*16=32 → only level 0
        laf = torch.tensor([[6.0, 0.0, 12.0], [0.0, 6.0, 12.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (1, 1, 1, PS, PS)
        assert patches.abs().sum().item() > 0

    def test_multi_level_uses_correct_pyramid_level(self, device, dtype):
        # Two LAFs with very different scales should be extracted from different pyramid levels.
        # We verify the output shape and that the function runs without errors.
        PS = 8
        img = torch.rand(1, 1, 128, 128, device=device, dtype=dtype)
        # Small-scale LAF (extracted at level 0) and large-scale LAF (extracted at higher level).
        laf_small = torch.tensor([[2.0, 0.0, 64.0], [0.0, 2.0, 64.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf_large = torch.tensor([[32.0, 0.0, 64.0], [0.0, 32.0, 64.0]], device=device, dtype=dtype).view(1, 1, 2, 3)
        laf_both = torch.cat([laf_small, laf_large], dim=1)
        patches = kornia.feature.extract_patches_from_pyramid(img, laf_both, PS)
        assert patches.shape == (1, 2, 1, PS, PS)

    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device, dtype=torch.float64)
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device, dtype=torch.float64)
        PS = 11
        self.gradcheck(
            kornia.feature.extract_patches_from_pyramid,
            (img, nlaf, PS, False),
            nondet_tol=1e-8,
        )


class TestLAFIsTouchingBoundary(BaseTester):
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert (5, 3) == kornia.feature.laf_is_inside_image(inp, img).shape

    def test_touch(self, device):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        expected = torch.tensor([[False, True]], device=device)
        assert torch.all(kornia.feature.laf_is_inside_image(laf, img) == expected).item()

    @staticmethod
    def _radius_laf(cx, cy, radius, device, dtype):
        return kornia.feature.laf_from_center_scale_ori(
            torch.tensor([[[cx, cy]]], device=device, dtype=dtype),
            torch.full((1, 1, 1, 1), radius, device=device, dtype=dtype),
        )

    def test_boundary_is_the_last_valid_pixel(self, device, dtype):
        """Valid coordinates run 0 .. size-1, matching `get_laf_center`'s documented convention.

        A LAF reaching exactly `w - 1` is inside; anything past it is not (see #4064).
        """
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32  # valid coordinates 0 .. 31
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0

        # rightmost boundary lands exactly on 31.0 -> inside
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(29.0, 16.0, r, device, dtype), img)[0, 0])
        # ... and half a pixel past it -> outside
        assert not bool(kornia.feature.laf_is_inside_image(self._radius_laf(29.5, 16.0, r, device, dtype), img)[0, 0])
        # same for the bottom edge
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(16.0, 29.0, r, device, dtype), img)[0, 0])
        assert not bool(kornia.feature.laf_is_inside_image(self._radius_laf(16.0, 29.5, r, device, dtype), img)[0, 0])

    def test_boundary_is_symmetric(self, device, dtype):
        """The low and high edges must be equally strict."""
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0
        # negative offsets push the LAF past the edge, which is where an asymmetric
        # upper bound shows up: on the low side it is rejected, on the high side it was not.
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            low = kornia.feature.laf_is_inside_image(self._radius_laf(r + offset, 16.0, r, device, dtype), img)
            high = kornia.feature.laf_is_inside_image(
                self._radius_laf(float(w - 1) - r - offset, 16.0, r, device, dtype), img
            )
            assert bool(low[0, 0]) == bool(high[0, 0]), f"asymmetric at offset {offset}"

    def test_border_argument_shrinks_both_edges(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("boundary comparison needs full precision")
        w = h = 32
        img = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        r = 2.0
        # with border=3 the usable range is 3 .. 28, so a radius-2 LAF centred at 26 just fits
        assert bool(kornia.feature.laf_is_inside_image(self._radius_laf(26.0, 16.0, r, device, dtype), img, 3)[0, 0])
        assert not bool(
            kornia.feature.laf_is_inside_image(self._radius_laf(26.5, 16.0, r, device, dtype), img, 3)[0, 0]
        )

    def test_jit(self, device, dtype):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]], [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        model = kornia.feature.laf_is_inside_image
        model_jit = torch.jit.script(kornia.feature.laf_is_inside_image)
        self.assert_close(model(laf, img), model_jit(laf, img))


class TestGetCreateLAF(BaseTester):
    def test_shape(self, device):
        xy = torch.ones(1, 3, 2, device=device)
        ori = torch.ones(1, 3, 1, device=device)
        scale = torch.ones(1, 3, 1, 1, device=device)
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        assert laf.shape == (1, 3, 2, 3)

    def test_laf(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        ori = torch.zeros(1, 1, 1, device=device)
        scale = 5 * torch.ones(1, 1, 1, 1, device=device)
        expected = torch.tensor([[[[5, 0, 1], [0, 5, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        self.assert_close(laf, expected)

    def test_laf_def(self, device):
        xy = torch.ones(1, 1, 2, device=device)
        expected = torch.tensor([[[[1, 0, 1], [0, 1, 1]]]], device=device).float()
        laf = kornia.feature.laf_from_center_scale_ori(xy)
        self.assert_close(laf, expected)

    def test_cross_consistency(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        scale2 = kornia.feature.get_laf_scale(laf)
        self.assert_close(scale, scale2)
        xy2 = kornia.feature.get_laf_center(laf)
        self.assert_close(xy2, xy)
        ori2 = kornia.feature.get_laf_orientation(laf)
        self.assert_close(ori2, ori)

    def test_gradcheck(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device, dtype=torch.float64)
        ori = torch.rand(batch_size, channels, 1, device=device, dtype=torch.float64)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device, dtype=torch.float64))
        self.gradcheck(kornia.feature.laf_from_center_scale_ori, (xy, scale, ori))

    @pytest.mark.skip("Depends on angle-to-rotation-matric")
    def test_jit(self, device, dtype):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        model = kornia.feature.laf_from_center_scale_ori
        model_jit = torch.jit.script(kornia.feature.laf_from_center_scale_ori)
        self.assert_close(model(xy, scale, ori), model_jit(xy, scale, ori))


class TestGetLAF3pts(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_to_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        inp = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        expected = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_to_three_points(inp)
        self.assert_close(threepts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_to_three_points, (inp,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_to_three_points
        model_jit = torch.jit.script(kornia.feature.laf_to_three_points)
        self.assert_close(model(inp), model_jit(inp))


class TestGetLAFFrom3pts(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_batch_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        out = kornia.feature.laf_from_three_points(inp)
        assert out.shape == inp.shape

    def test_conversion(self, device):
        expected = torch.tensor([[1, 0, 2], [0, 1, 3]], device=device).float().view(1, 1, 2, 3)
        inp = torch.tensor([[3, 2, 2], [3, 4, 3]], device=device).float().view(1, 1, 2, 3)
        threepts = kornia.feature.laf_from_three_points(inp)
        self.assert_close(threepts, expected)

    def test_cross_consistency(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp_2 = kornia.feature.laf_from_three_points(inp)
        inp_2 = kornia.feature.laf_to_three_points(inp_2)
        self.assert_close(inp_2, inp)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.feature.laf_from_three_points, (inp,))

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        model = kornia.feature.laf_from_three_points
        model_jit = torch.jit.script(kornia.feature.laf_from_three_points)
        self.assert_close(model(inp), model_jit(inp))


class TestTransformLAFs(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    def test_transform_points(self, batch_size, num_points, device, dtype):
        # generate input data
        eye_size = 3
        lafs_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=dtype)

        dst_homo_src = create_random_homography(lafs_src, eye_size)
        # transform the points from dst to ref
        lafs_dst = kornia.feature.perspective_transform_lafs(dst_homo_src, lafs_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        lafs_dst_to_src = kornia.feature.perspective_transform_lafs(src_homo_dst, lafs_dst)

        # projected should be equal as initial
        self.assert_close(lafs_src, lafs_dst_to_src)

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points = 2, 3
        eye_size = 3
        points_src = torch.rand(batch_size, num_points, 2, 3, device=device, dtype=torch.float64)
        dst_homo_src = create_random_homography(points_src, eye_size)
        # evaluate function gradient
        self.gradcheck(kornia.feature.perspective_transform_lafs, (dst_homo_src, points_src))
