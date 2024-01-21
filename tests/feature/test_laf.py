import pytest
import torch

import kornia
import kornia.geometry.transform.imgwarp
from testing.base import BaseTester
from testing.laf import create_random_homography


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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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
    @pytest.mark.jit()
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

    @pytest.mark.jit()
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

    @pytest.mark.jit()
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
