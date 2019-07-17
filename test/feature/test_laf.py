import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck
import math


class TestAngleToRotationMatrix:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        rotmat= kornia.feature.angle_to_rotation_matrix(inp)
        assert rotmat.shape == (1, 3, 4, 4, 2, 2)

    def test_angles(self):
        inp = torch.tensor([0, math.pi/2.0])

        expected = torch.tensor([[[1.0, 0.], [0., 1.0]],
                                 [[0, 1.0],[-1.0, 0]]])
        rotmat= kornia.feature.angle_to_rotation_matrix(inp)
        assert_allclose(rotmat, expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.angle_to_rotation_matrix,
                         (img,),
                         raise_exception=True)


class TestGetLAFScale:
    def test_shape(self):
        inp = torch.ones(1, 3, 2, 3)
        rotmat= kornia.feature.get_laf_scale(inp)
        assert rotmat.shape == (1, 3, 1, 1)

    def test_scale(self):
        inp = torch.tensor([[5., 1, 0], [1, 1, 0]]).float()
        inp = inp.view(1,1,2,3)
        expected = torch.tensor([[[[2]]]]).float()
        rotmat= kornia.feature.get_laf_scale(inp)
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
        rotmat= kornia.feature.make_upright(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_do_nothing(self):
        inp = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
        inp = inp.view(1,1,2,3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
        laf= kornia.feature.make_upright(inp)
        assert_allclose(laf, expected)

    def test_check_zeros(self):
        inp = torch.rand(4, 5, 2, 3)
        laf = kornia.feature.make_upright(inp)
        must_be_zeros = laf[:,:,0,1]
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
        inp[:,:,3] = 0
        rotmat= kornia.feature.ell2LAF(inp)
        assert rotmat.shape == (5, 3, 2, 3)

    def test_conversion(self):
        inp = torch.tensor([[10, -20, 0.01, 0, 0.01]]).float()
        inp = inp.view(1, 1, 5)
        expected = torch.tensor([[10, 0, 10.], [0, 10, -20]]).float()
        expected = expected.view(1, 1, 2, 3)
        laf = kornia.feature.ell2LAF(inp)
        assert_allclose(laf, expected)


    def test_gradcheck(self):
        batch_size, channels, height = 1, 2, 5
        img = torch.rand(batch_size, channels, height).abs()
        img[:,:,2] = img[:,:,3].abs()+0.3
        img[:,:,4] += 1.
        # assure it is positive definite  
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.ell2LAF,
                         (img,),
                         raise_exception=True)
