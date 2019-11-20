import pytest

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry import pi
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestAdjustSaturation:
    def test_saturation_one(self, device):
        data = torch.tensor([[[.5, .5],
                              [.5, .5]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = data.clone()

        f = kornia.color.AdjustSaturation(1.)
        assert_allclose(f(data), expected)

    def test_saturation_one_batch(self):
        data = torch.tensor([[[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]],

                             [[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]]])  # 2x3x2x2

        expected = data
        f = kornia.color.AdjustSaturation(torch.ones(2))
        assert_allclose(f(data), expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_saturation, (img, 2.),
                         raise_exception=True)


class TestAdjustHue:
    def test_hue_one(self, device):
        data = torch.tensor([[[.5, .5],
                              [.5, .5]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = data.clone()

        f = kornia.color.AdjustHue(0.)
        assert_allclose(f(data), expected)

    def test_hue_one_batch(self):
        data = torch.tensor([[[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]],

                             [[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]]])  # 2x3x2x2

        expected = data
        f = kornia.color.AdjustHue(torch.tensor([0, 0]))
        assert_allclose(f(data), expected)

    def test_hue_flip_batch(self):
        data = torch.tensor([[[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]],

                             [[[.5, .5],
                               [.5, .5]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]]])  # 2x3x2x2

        f = kornia.color.AdjustHue(torch.tensor([-pi, pi]))
        result = f(data)
        assert_allclose(result, result.flip(0))

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_hue, (img, 2.),
                         raise_exception=True)


class TestAdjustGamma:
    def test_gamma_zero(self, device):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = torch.ones_like(data)

        f = kornia.color.AdjustGamma(0.)
        assert_allclose(f(data), expected)

    def test_gamma_one(self, device):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = data.clone()

        f = kornia.color.AdjustGamma(1.)
        assert_allclose(f(data), expected)

    def test_gamma_one_gain_two(self, device):
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

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.AdjustGamma(1., 2.)
        assert_allclose(f(data), expected)

    def test_gamma_two(self, device):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        expected = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[.25, .25],
                                  [.25, .25]],

                                 [[.0625, .0625],
                                  [.0625, .0625]]])  # 3x2x2

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.AdjustGamma(2.)
        assert_allclose(f(data), expected)

    def test_gamma_two_batch(self):
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]],

                             [[[1., 1.],
                               [1., 1.]],

                              [[.5, .5],
                               [.5, .5]],

                              [[.25, .25],
                               [.25, .25]]]])  # 2x3x2x2

        expected = torch.tensor([[[[1., 1.],
                                   [1., 1.]],

                                  [[.25, .25],
                                   [.25, .25]],

                                  [[.0625, .0625],
                                   [.0625, .0625]]],

                                 [[[1., 1.],
                                   [1., 1.]],

                                  [[.25, .25],
                                   [.25, .25]],

                                  [[.0625, .0625],
                                   [.0625, .0625]]]])  # 2x3x2x2

        f = kornia.color.AdjustGamma(torch.tensor([2., 2.]), gain=torch.ones(2))
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = img.to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_gamma, (img, 1., 2.),
                         raise_exception=True)


class TestAdjustContrast:
    def test_factor_zero(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = torch.zeros_like(data)

        f = kornia.color.AdjustContrast(0.)

        assert_allclose(f(data), expected)

    def test_factor_one(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2
        data = data.to(device)
        expected = data.clone()

        f = kornia.color.AdjustContrast(1.)

        assert_allclose(f(data), expected)

    def test_factor_two(self, device):
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

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.AdjustContrast(2.)

        assert_allclose(f(data), expected)

    def test_factor_tensor(self, device):
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

        factor = torch.tensor([0, 1, 1.5, 2])

        data = data.to(device)
        expected = expected.to(device)
        factor = factor.to(device)

        f = kornia.color.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_color(self, device):
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

        factor = torch.tensor([1, 2])

        data = data.to(device)
        expected = expected.to(device)
        factor = factor.to(device)

        f = kornia.color.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_shape(self, device):
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

        factor = torch.tensor([1.5, 2.])

        data = data.to(device)
        expected = expected.to(device)
        factor = factor.to(device)

        f = kornia.color.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = img.to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_contrast, (img, 2.),
                         raise_exception=True)


class TestAdjustBrightness:
    def test_factor_zero(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = data.clone()

        f = kornia.color.AdjustBrightness(0.)
        assert_allclose(f(data), expected)

    def test_factor_one(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        data = data.to(device)
        expected = torch.ones_like(data)

        f = kornia.color.AdjustBrightness(1.)
        assert_allclose(f(data), expected)

    def test_factor_minus(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.75, .75],
                              [.75, .75]],

                             [[.25, .25],
                              [.25, .25]]])  # 3x2x2

        expected = torch.tensor([[[.5, .5],
                                  [.5, .5]],

                                 [[.25, .25],
                                  [.25, .25]],

                                 [[0., 0.],
                                  [0., 0.]]])  # 3x2x2

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.AdjustBrightness(-0.5)
        assert_allclose(f(data), expected)

    def test_factor_tensor(self, device):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]],

                             [[.5, .5],
                              [.5, .5]]])  # 4x2x2

        factor = torch.tensor([0, 0.5, 0.75, 2])

        data = data.to(device)
        expected = torch.ones_like(data)
        factor = factor.to(device)

        f = kornia.color.AdjustBrightness(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_color(self, device):
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

                                  [[.75, .75],
                                   [.75, .75]],

                                  [[.5, .5],
                                   [.5, .5]]],

                                 [[[.1, .1],
                                   [.1, .1]],

                                  [[.4, .4],
                                   [.4, .4]],

                                  [[.7, .7],
                                   [.7, .7]]]])  # 2x3x2x2

        factor = torch.tensor([0.25, 0.1])

        data = data.to(device)
        expected = expected.to(device)
        factor = factor.to(device)

        f = kornia.color.AdjustBrightness(factor)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = img.to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_brightness, (img, 2.),
                         raise_exception=True)
