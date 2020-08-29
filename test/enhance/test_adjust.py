import pytest

import kornia
import kornia.testing as utils  # test utils
from kornia.constants import pi

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

        f = kornia.enhance.AdjustSaturation(1.)
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
        f = kornia.enhance.AdjustSaturation(torch.ones(2))
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

        f = kornia.enhance.AdjustHue(0.)
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
        f = kornia.enhance.AdjustHue(torch.tensor([0, 0]))
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

        f = kornia.enhance.AdjustHue(torch.tensor([-pi, pi]))
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

        f = kornia.enhance.AdjustGamma(0.)
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

        f = kornia.enhance.AdjustGamma(1.)
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

        f = kornia.enhance.AdjustGamma(1., 2.)
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

        f = kornia.enhance.AdjustGamma(2.)
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

        f = kornia.enhance.AdjustGamma(torch.tensor([2., 2.]), gain=torch.ones(2))
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

        f = kornia.enhance.AdjustContrast(0.)

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

        f = kornia.enhance.AdjustContrast(1.)

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

        f = kornia.enhance.AdjustContrast(2.)

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

        f = kornia.enhance.AdjustContrast(factor)
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

        f = kornia.enhance.AdjustContrast(factor)
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

        f = kornia.enhance.AdjustContrast(factor)
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

        f = kornia.enhance.AdjustBrightness(0.)
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

        f = kornia.enhance.AdjustBrightness(1.)
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

        f = kornia.enhance.AdjustBrightness(-0.5)
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

        f = kornia.enhance.AdjustBrightness(factor)
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

        f = kornia.enhance.AdjustBrightness(factor)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = img.to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_brightness, (img, 2.),
                         raise_exception=True)


class TestEqualize:
    def test_shape_equalize(self, device):
        bs, channels, height, width = 1, 3, 4, 5

        inputs = torch.ones(channels, height, width)
        inputs = inputs.to(device)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size([bs, channels, height, width])

    def test_shape_equalize_batch(self, device):
        bs, channels, height, width = 2, 3, 4, 5

        inputs = torch.ones(bs, channels, height, width)
        inputs = inputs.to(device)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size([bs, channels, height, width])

    def test_equalize(self, device):
        bs, channels, depth, height, width = 1, 3, 6, 20, 20

        inputs = self.build_input(channels, height, width).squeeze(dim=0)
        inputs = inputs.to(device)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs=1, row=row_expected)
        expected = expected.to(device)

        f = kornia.enhance.equalize

        assert_allclose(f(inputs), expected)

    def test_equalize_batch(self, device):
        bs, channels, depth, height, width = 2, 3, 6, 20, 20

        inputs = self.build_input(channels, height, width, bs)
        inputs = inputs.to(device)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs, row=row_expected)
        expected = expected.to(device)

        f = kornia.enhance.equalize

        assert_allclose(f(inputs), expected)

    def test_gradcheck(self, device):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.ones(bs, channels, height, width)
        inputs = inputs.to(device)
        inputs = utils.tensor_to_gradcheck_var(inputs)
        assert gradcheck(kornia.enhance.equalize, (inputs,),
                         raise_exception=True)

    @staticmethod
    def build_input(channels, height, width, bs=1, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        batch = torch.stack([image] * bs)

        return batch


class TestEqualize3D:
    def test_shape_equalize3d(self, device):
        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = torch.ones(channels, depth, height, width)
        inputs3d = inputs3d.to(device)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size([bs, channels, depth, height, width])

    def test_shape_equalize3d_batch(self, device):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = torch.ones(bs, channels, depth, height, width)
        inputs3d = inputs3d.to(device)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size([bs, channels, depth, height, width])

    def test_equalize3d(self, device):
        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width).squeeze(dim=0)
        inputs3d.to(device)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ])
        expected = self.build_input(channels, depth, height, width, bs=1, row=row_expected)
        expected = expected.to(device)

        f = kornia.enhance.equalize3d

        assert_allclose(f(inputs3d), expected)

    def test_equalize3d_batch(self, device):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs)
        inputs3d = inputs3d.to(device)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ])
        expected = self.build_input(channels, depth, height, width, bs, row=row_expected)
        expected = expected.to(device)

        f = kornia.enhance.equalize3d

        assert_allclose(f(inputs3d), expected)

    def test_gradcheck(self, device):
        bs, channels, depth, height, width = 2, 3, 6, 4, 5
        inputs3d = torch.ones(bs, channels, depth, height, width)
        inputs3d = inputs3d.to(device)
        inputs3d = utils.tensor_to_gradcheck_var(inputs3d)
        assert gradcheck(kornia.enhance.equalize3d, (inputs3d,),
                         raise_exception=True)

    @staticmethod
    def build_input(channels, depth, height, width, bs=1, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        image3d = torch.stack([image] * depth).transpose(0, 1)
        batch = torch.stack([image3d] * bs)

        return batch
