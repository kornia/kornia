import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestAdjustColor:
    def test_adjust_bightness_factor_zero(self):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        expected = torch.tensor([[[0., 0.],
                                  [0., 0.]],

                                 [[0., 0.],
                                  [0., 0.]],

                                 [[0., 0.],
                                  [0., 0.]]])  # 3x2x2

        f = kornia.color.AdjustBrightness()

        assert_allclose(f(data, 0), expected)

    def test_adjust_bightness_factor_one(self):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        expected = data

        f = kornia.color.AdjustBrightness()

        assert_allclose(f(data, 1), expected)

    def test_adjust_bightness_factor_two(self):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        expected = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[1., 1.],
                                  [1., 1.]],

                                 [[.5, .5],
                                  [.5, .5]]])  # 3x2x2

        f = kornia.color.AdjustBrightness()

        assert_allclose(f(data, 2), expected)

        
    def test_adjust_bightness_factor_tensor(self):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]],
                            
                             [[.5, .5],
                              [.5, .5]]])  # 4x2x2

        expected = torch.tensor([[[0., 0.],
                                  [0., 0.]],

                                 [[.5, .5],
                                  [.5, .5]],

                                 [[.375, .375],
                                  [.375, .375]],
                                 
                                 [[1., 1.],
                                  [1., 1.]]])  # 4x2x2

        f = kornia.color.AdjustBrightness()
        assert_allclose(f(data, torch.Tensor([0, 1, 1.5, 2])), expected)
