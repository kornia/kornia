import pytest

import kornia as kornia
import kornia.geometry.transform.imgwarp
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck
import math


class TestAngleToRotationMatrix:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self):
        ang_deg = torch.tensor([0, 90.])
        expected = torch.tensor([[[1.0, 0.], [0., 1.0]],
                                 [[0, 1.0], [-1.0, 0]]])
        rotmat = kornia.geometry.transform.imgwarp.angle_to_rotation_matrix(ang_deg)
        assert_allclose(rotmat, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.transform.imgwarp.angle_to_rotation_matrix,
                         (img,),
                         raise_exception=True)


class TestGetLAFScale:
    def test_shape(self):
        inp = torch.ones(1, 3, 2, 3)
        rotmat = kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self):
        inp = torch.tensor([[5., 1, 0], [1, 1, 0]]).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[[[2]]]]).float()
        rotmat = kornia.feature.get_laf_scale(inp)
        assert_allclose(rotmat, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.get_laf_scale,
                         (img,),
                         raise_exception=True)


class TestMakeUpright:
    def test_shape(self):
        inp = torch.ones(5, 3, 2, 3)
        rotmat = kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
        inp = inp.view(1, 1, 2, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
        laf = kornia.feature.make_upright(inp)
        assert_allclose(laf, expected)

    def test_check_zeros(self):
        inp = torch.rand(4, 5, 2, 3)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:, :, 0, 1]
        assert_allclose(must_be_zeros, torch.zeros_like(must_be_zeros))

    def test_gradcheck(self):
        batch_size, channels, height, width = 14, 2, 2, 3
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.make_upright,
                         (img,),
                         raise_exception=True)


class TestELL2LAF:
    def test_shape(self):
        inp = torch.ones(5, 3, 5)
        inp[:, :, 3] = 0
        rotmat = kornia.feature.ellipse_to_laf(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]]).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.], [0, 10, -20]]).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ellipse_to_laf(inp)
        assert_allclose(laf, expected)

    def test_gradcheck(self):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height).abs()
        img[:, :, 2] = img[:, :, 3].abs() + 0.3
        img[:, :, 4] += 1.
        # assure it is positive definite
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.ellipse_to_laf,
                         (img,),
                         raise_exception=True)


class TestNormalizeLAF:
    def test_shape(self):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.normalize_laf(inp, img).shape

    def test_conversion(self):
        w, h = 10, 5
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        expected = torch.tensor([[0.2, 0, 0.1], [0, 0.2, 0.2]]).float()
        lafn = kornia.feature.normalize_laf(laf, img)
        assert_allclose(lafn, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.normalize_laf,
                         (laf, img,),
                         raise_exception=True)


class TestLAF2pts:
    def test_shape(self):
        inp = torch.rand(5, 3, 2, 3)
        n_pts = 13
        assert kornia.feature.laf_to_boundary_points(inp, n_pts).shape == (5, 3, n_pts, 2)

    def test_conversion(self):
        laf = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        laf = laf.view(1, 1, 2, 3)
        n_pts = 6
        expected = torch.tensor([[[[1, 1],
                                   [1, 2],
                                   [2, 1],
                                   [1, 0],
                                   [0, 1],
                                   [1, 2]]]]).float()
        pts = kornia.feature.laf_to_boundary_points(laf, n_pts)
        assert_allclose(pts, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 3, 2, 2, 3
        laf = torch.rand(batch_size, channels, height, width)
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.laf_to_boundary_points,
                         (laf),
                         raise_exception=True)


class TestDenormalizeLAF:
    def test_shape(self):
        inp = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        assert inp.shape == kornia.feature.denormalize_laf(inp, img).shape

    def test_conversion(self):
        w, h = 10, 5
        expected = torch.tensor([[1, 0, 1], [0, 1, 1]]).float()
        expected = expected.view(1, 1, 2, 3)
        img = torch.rand(1, 3, h, w)
        lafn = torch.tensor([[0.2, 0, 0.1], [0, 0.2, 0.2]]).float()
        laf = kornia.feature.denormalize_laf(lafn.view(1, 1, 2, 3), img)
        assert_allclose(laf, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 2, 3

        laf = torch.rand(batch_size, channels, height, width)
        img = torch.rand(batch_size, 3, 10, 32)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(kornia.feature.denormalize_laf,
                         (laf, img,),
                         raise_exception=True)


class TestGenPatchGrid:
    def test_shape(self):
        laf = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF
        grid = generate_patch_grid_from_normalized_LAF(img, laf, PS)
        assert grid.shape == (15, 3, 3, 2)

    def test_gradcheck(self):
        laf = torch.rand(5, 3, 2, 3)
        img = torch.rand(5, 3, 10, 10)
        PS = 3
        from kornia.feature.laf import generate_patch_grid_from_normalized_LAF
        img = utils.tensor_to_gradcheck_var(img)  # to var
        laf = utils.tensor_to_gradcheck_var(laf)  # to var
        assert gradcheck(generate_patch_grid_from_normalized_LAF,
                         (img, laf, PS,),
                         raise_exception=True)


class TestExtractPatchesSimple:
    def test_shape(self):
        laf = torch.rand(5, 4, 2, 3)
        img = torch.rand(5, 3, 100, 30)
        PS = 10
        patches = kornia.feature.extract_patches_simple(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_gradcheck(self):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]]).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 100, 120)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(kornia.feature.extract_patches_simple,
                         (img, nlaf, PS,),
                         raise_exception=True)


class TestExtractPatchesPyr:
    def test_shape(self):
        laf = torch.rand(5, 4, 2, 3)
        img = torch.rand(5, 3, 100, 30)
        PS = 10
        patches = kornia.feature.extract_patches_from_pyramid(img, laf, PS)
        assert patches.shape == (5, 4, 3, PS, PS)

    def test_gradcheck(self):
        nlaf = torch.tensor([[0.1, 0.001, 0.5], [0, 0.1, 0.5]]).float()
        nlaf = nlaf.view(1, 1, 2, 3)
        img = torch.rand(1, 3, 100, 120)
        PS = 11
        img = utils.tensor_to_gradcheck_var(img)  # to var
        nlaf = utils.tensor_to_gradcheck_var(nlaf)  # to var
        assert gradcheck(kornia.feature.extract_patches_from_pyramid,
                         (img, nlaf, PS,),
                         raise_exception=True)


class TestConvertFromOpenCVKpts:
    def test_shape(self):
        class KP():
            def __init__(self, x, y, scale, angle):
                self.pt = (x, y)
                self.size = scale
                self.angle = angle
        kps = [KP(10, 23, 3.0, 0), KP(5, 8, 1.0, 90)]
        lafs = kornia.feature.create_lafs_from_opencv_kps(kps)
        assert lafs.shape == (1, 2, 2, 3)

    def test_conversion(self):
        class KP():
            def __init__(self, x, y, scale, angle):
                self.pt = (x, y)
                self.size = scale
                self.angle = angle
        kps = [KP(10, 23, 3.0, 0), KP(5, 8, 1.0, 90)]
        lafs = kornia.feature.create_lafs_from_opencv_kps(kps, 12.0)
        expected = torch.tensor(
            [[[[0, 18., 10.],
              [-18., 0.0, 23.]],
             [[6., 0., 5],
              [-0., 6.0, 8]]]])
        assert_allclose(lafs, expected)
