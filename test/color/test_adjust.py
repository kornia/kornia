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

    def test_adjust_bightness_factor_tensor_color(self):
        # prepare input data
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]],

                             [[[0., 0.],
                               [0., 0.]],

                              [[.3, .3],
                               [.3, .3]],

                              [[.6, .6],
                               [.6, .6]]]])  # 2x3x2x2

        expected = torch.tensor([[[[1., 1.],
                                   [1., 1.]],

                                  [[.5, .5],
                                   [.5, .5]],

                                  [[.25, .25],
                                   [.25, .25]]],

                                 [[[0., 0.],
                                   [0., 0.]],

                                  [[.6, .6],
                                   [.6, .6]],

                                  [[1., 1.],
                                   [1., 1.]]]])  # 2x3x2x2

        f = kornia.color.AdjustBrightness()
        assert_allclose(f(data, torch.Tensor([1, 2])), expected)

    def test_adjust_bightness_factor_tensor_shape(self):
        # prepare input data
        data = torch.tensor([[[[1., 1., .5],
                               [1., 1., .5]],

                              [[.5, .5, .25],
                               [.5, .5, .25]],

                              [[.25, .25, .25],
                               [.6, .6, .3]]],

                             [[[0., 0., 1.],
                               [0., 0., .25]],

                              [[.3, .3, .4],
                               [.3, .3, .4]],

                              [[.6, .6, 0.],
                               [.3, .2, .1]]]])  # 2x3x2x3

        expected = torch.tensor([[[[1., 1., .75],
                                   [1., 1., .75]],

                                  [[.75, .75, .375],
                                   [.75, .75, .375]],

                                  [[.375, .375, .375],
                                   [.9, .9, .45]]],

                                 [[[0., 0., 1.],
                                   [0., 0., .5]],

                                  [[.6, .6, .8],
                                     [.6, .6, .8]],

                                  [[1., 1., 0.],
                                     [.6, .4, .2]]]])  # 2x3x2x3

        f = kornia.color.AdjustBrightness()
        assert_allclose(f(data, torch.Tensor([1.5, 2])), expected)
