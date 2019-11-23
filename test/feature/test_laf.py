import pytest

import kornia as kornia
import kornia.geometry.transform.imgwarp
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestAngleToRotationMatrix:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4).to(device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self, device):
        ang_deg = torch.tensor([0, 90.], device=device)
        expected = torch.tensor([[[1.0, 0.], [0., 1.0]],
                                 [[0, 1.0], [-1.0, 0]]], device=device)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(ang_deg)
        assert_allclose(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix,
                         (img,),
                         raise_exception=True)


class TestGetLAFScale:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        rotmat = kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self, device):
        inp = torch.tensor([[5., 1, 0], [1, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2]]]], device=device).float()
        rotmat = kornia.feature.get_laf_scale(inp)
        assert_allclose(rotmat, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_scale,
                         (img,),
                         raise_exception=True)


class TestGetLAFCenter:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        xy = kornia.feature.get_laf_center(inp)
        assert xy.shape == (1, 3, 2)

    def test_center(self, device):
        inp = torch.tensor([[5., 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[2, 3]]], device=device).float()
        xy = kornia.feature.get_laf_center(inp)
        assert_allclose(xy, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_center,
                         (img,),
                         raise_exception=True)


class TestGetLAFOri:
    def test_shape(self, device):
        inp = torch.ones(1, 3, 2, 3, device=device)
        ori = kornia.feature.get_laf_orientation(inp)
        assert ori.shape == (1, 3, 1)

    def test_ori(self, device):
        inp = torch.tensor([[1, 1, 2], [1, 1, 3]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[45.]]], device=device).float()
        angle = kornia.feature.get_laf_orientation(inp)
        assert_allclose(angle, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_orientation,
                         (img,),
                         raise_exception=True)


class TestScaleLAF:
    def test_shape_float(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = 23.
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_shape_tensor(self, device):
        inp = torch.ones(7, 3, 2, 3, device=device).float()
        scale = torch.zeros(7, 1, 1, 1, device=device).float()
        assert kornia.feature.scale_laf(inp, scale).shape == inp.shape

    def test_scale(self, device):
        inp = torch.tensor([[5., 1, 0.8], [1, 1, -4.]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        scale = torch.tensor([[[[2.]]]], device=device).float()
        out = kornia.feature.scale_laf(inp, scale)
        expected = torch.tensor([[[[10., 2, 0.8], [2, 2, -4.]]]], device=device).float()
        assert_allclose(out, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        scale = torch.rand(batch_size, device=device)
        scale = utils.tensor_to_gradcheck_var(scale)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.scale_laf,
                         (laf, scale),
                         raise_exception=True, atol=1e-4)


class TestMakeUpright:
    def test_shape(self, device):
        inp = torch.ones(5, 3, 2, 3, device=device)
        rotmat = kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self, device):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        assert_allclose(laf, expected)

    def test_do_nothing_with_scalea(self, device):
        inp = torch.tensor([[2, 0, 0], [0, 2, 0]], device=device).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[2, 0, 0], [0, 2, 0]], device=device).float()
        laf = kornia.feature.make_upright(inp)
        assert_allclose(laf, expected)

    def test_check_zeros(self, device):
        inp = torch.rand(4, 5, 2, 3, device=device)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:, :, 0, 1]
        assert_allclose(must_be_zeros, torch.zeros_like(must_be_zeros))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 14, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.make_upright,
                         (img,),
                         raise_exception=True)


class TestELL2LAF:
    def test_shape(self, device):
        inp = torch.ones(5, 3, 5, device=device)
        inp[:, :, 3] = 0
        rotmat = kornia.feature.ellipse_to_laf(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self, device):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]], device=device).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.], [0, 10, -20]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ellipse_to_laf(inp)
        assert_allclose(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height, device=device).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.
        # assure it is positive definite
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.ellipse_to_laf,
                         (img,),
                         raise_exception=True)


class TestNormalizeLAF:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.normalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 10, 5
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        expected = torch.tensor([[0.2, 0, 0.1], [0, 0.2, 0.2]]).float()
        lafn = kornia.feature.normalize_laf(laf, img)
        assert_allclose(lafn, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.normalize_laf,
                         (laf, img,),
                         raise_exception=True)


class TestLAF2pts:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        n_pts = 13
        assert kornia.feature.laf_to_boundary_points(inp, n_pts).shape == (5, 3, n_pts, 2)

    def test_conversion(self, device):
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        laf = laf.view(1, 1, 2, 3)
        n_pts = 6
        expected = torch.tensor([[[[1, 1],
                                   [1, 2],
                                   [2, 1],
                                   [1, 0],
                                   [0, 1],
                                   [1, 2]]]], device=device).float()
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        assert_allclose(pts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width, device=device)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.laf_to_boundary_points,
                         (laf),
                         raise_exception=True)


class TestDenormalizeLAF:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert inp.shape == kornia.feature.denormalize_laf(inp, img).shape

    def test_conversion(self, device):
        w, h = 10, 5
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device).float()
        expected = expected.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w, device=device)
        lafn = torch.tensor([[0.2, 0, 0.1], [0, 0.2, 0.2]], device=device).float()
        laf = kornia.feature.denormalize_laf(lafn.view(1, 1, 2, 3), img)
        assert_allclose(laf, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width, device=device)
        img = torch.rand(batch_size, 3, 10, 32, device=device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.denormalize_laf,
                         (laf, img,),
                         raise_exception=True)


class TestGenPatchGrid:
    def test_shape(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF
        grid = generate_patch_grid_from_normalized_LAF(img, laf, PS)
        assert grid.shape == (15, 3, 3, 2)

    def test_gradcheck(self, device):
        laf = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(generate_patch_grid_from_normalized_LAF,
                         (img, laf, PS,),
                         raise_exception=True)


class TestExtractPatchesSimple:
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    # TODO: check what to do to improve timing
    # @pytest.mark.skip("The test takes too long to finish.")
    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(kornia.feature.extract_patches_simple,
                         (img, nlaf, PS, False),
                         raise_exception=True)


class TestExtractPatchesPyr:
    def test_shape(self, device):
        laf = torch.rand(5, 4, 2, 3, device=device)
        img = torch.rand(5, 3, 100, 30, device=device)
        PS = 10
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    # TODO: check what to do to improve timing
    # @pytest.mark.skip("The test takes too long to finish.")
    def test_gradcheck(self, device):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]], device=device).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 20, 30, device=device)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(kornia.feature.extract_patches_from_pyramid,
                         (img, nlaf, PS, False),
                         raise_exception=True)


class TestLAFIsTouchingBoundary:
    def test_shape(self, device):
        inp = torch.rand(5, 3, 2, 3, device=device)
        img = torch.rand(5, 3, 10, 10, device=device)
        assert (5, 3) == kornia.feature.laf_is_inside_image(inp, img).shape

    def test_touch(self, device):
        w, h = 10, 5
        img = torch.rand(1, 3, h, w, device=device)
        laf = torch.tensor([[[[10, 0, 3], [0, 10, 3]],
                             [[1, 0, 5], [0, 1, 2]]]], device=device).float()
        expected = torch.tensor([[False, True]], device=device)
        assert torch.all(kornia.feature.laf_is_inside_image(laf, img) == expected).item()


class TestGetCreateLAF:
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
        assert_allclose(laf, expected)

    def test_cross_consistency(self, device):
        batch_size, channels = 3, 2
        xy = torch.rand(batch_size, channels, 2, device=device)
        ori = torch.rand(batch_size, channels, 1, device=device)
        scale = torch.abs(torch.rand(batch_size, channels, 1, 1, device=device))
        laf = kornia.feature.laf_from_center_scale_ori(xy, scale, ori)
        scale2 = kornia.feature.get_laf_scale(laf)
        assert_allclose(scale, scale2)
        xy2 = kornia.feature.get_laf_center(laf)
        assert_allclose(xy2, xy)
        ori2 = kornia.feature.get_laf_orientation(laf)
        assert_allclose(ori2, ori)

    def test_gradcheck(self, device):
        batch_size, channels = 3, 2
        xy = utils.tensor_to_gradcheck_var(torch.rand(batch_size, channels, 2, device=device))
        ori = utils.tensor_to_gradcheck_var(torch.rand(batch_size, channels, 1, device=device))
        scale = utils.tensor_to_gradcheck_var(torch.abs(torch.rand(batch_size, channels, 1, 1, device=device)))
        assert gradcheck(kornia.feature.laf_from_center_scale_ori,
                         (xy, scale, ori,),
                         raise_exception=True)


class TestGetLAF3pts:
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
        assert_allclose(threepts, expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(kornia.feature.laf_to_three_points,
                         (inp,),
                         raise_exception=True)


class TestGetLAFFrom3pts:
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
        assert_allclose(threepts, expected)

    def test_cross_consistency(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp_2 = kornia.feature.laf_from_three_points(inp)
        inp_2 = kornia.feature.laf_to_three_points(inp_2)
        assert_allclose(inp_2, inp)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 3, 2, 2, 3
        inp = torch.rand(batch_size, channels, height, width, device=device)
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(kornia.feature.laf_from_three_points,
                         (inp,),
                         raise_exception=True)
