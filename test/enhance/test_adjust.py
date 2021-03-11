import pytest

import kornia
from kornia.testing import tensor_to_gradcheck_var, BaseTester
import kornia.testing as utils
from kornia.constants import pi

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestAdjustSaturation:
    def test_saturation_one(self, device, dtype):
        data = torch.tensor([[[.5, .5],
                              [.5, .5]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustSaturation(1.)
        assert_allclose(f(data), expected)

    def test_saturation_one_batch(self, device, dtype):
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
                               [.25, .25]]]], device=device, dtype=dtype)  # 2x3x2x2

        expected = data
        f = kornia.enhance.AdjustSaturation(torch.ones(2))
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_saturation, (img, 2.),
                         raise_exception=True)


class TestAdjustHue:
    def test_hue_one(self, device, dtype):
        data = torch.tensor([[[.5, .5],
                              [.5, .5]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustHue(0.)
        assert_allclose(f(data), expected)

    def test_hue_one_batch(self, device, dtype):
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
                               [.25, .25]]]], device=device, dtype=dtype)  # 2x3x2x2

        expected = data
        f = kornia.enhance.AdjustHue(torch.tensor([0, 0]))
        assert_allclose(f(data), expected)

    def test_hue_flip_batch(self, device, dtype):
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
                               [.25, .25]]]], device=device, dtype=dtype)  # 2x3x2x2

        pi_t = torch.tensor([-pi, pi], device=device, dtype=dtype)
        f = kornia.enhance.AdjustHue(pi_t)

        result = f(data)
        assert_allclose(result, result.flip(0))

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_hue, (img, 2.),
                         raise_exception=True)


class TestAdjustGamma:
    def test_gamma_zero(self, device, dtype):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.ones_like(data)

        f = kornia.enhance.AdjustGamma(0.)
        assert_allclose(f(data), expected)

    def test_gamma_one(self, device, dtype):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustGamma(1.)
        assert_allclose(f(data), expected)

    def test_gamma_one_gain_two(self, device, dtype):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[1., 1.],
                                  [1., 1.]],

                                 [[.5, .5],
                                  [.5, .5]]], device=device, dtype=dtype)  # 3x2x2

        f = kornia.enhance.AdjustGamma(1., 2.)
        assert_allclose(f(data), expected)

    def test_gamma_two(self, device, dtype):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[.25, .25],
                                  [.25, .25]],

                                 [[.0625, .0625],
                                  [.0625, .0625]]], device=device, dtype=dtype)  # 3x2x2

        f = kornia.enhance.AdjustGamma(2.)
        assert_allclose(f(data), expected)

    def test_gamma_two_batch(self, device, dtype):
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
                               [.25, .25]]]], device=device, dtype=dtype)  # 2x3x2x2

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
                                   [.0625, .0625]]]], device=device, dtype=dtype)  # 2x3x2x2

        p1 = torch.tensor([2., 2.], device=device, dtype=dtype)
        p2 = torch.ones(2, device=device, dtype=dtype)

        f = kornia.enhance.AdjustGamma(p1, gain=p2)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_gamma, (img, 1., 2.),
                         raise_exception=True)


class TestAdjustContrast:
    def test_factor_zero(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.zeros_like(data)

        f = kornia.enhance.AdjustContrast(0.)
        assert_allclose(f(data), expected)

    def test_factor_one(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustContrast(1.)
        assert_allclose(f(data), expected)

    def test_factor_two(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[1., 1.],
                                  [1., 1.]],

                                 [[.5, .5],
                                  [.5, .5]]], device=device, dtype=dtype)  # 3x2x2

        f = kornia.enhance.AdjustContrast(2.)
        assert_allclose(f(data), expected)

    def test_factor_tensor(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]],

                             [[.5, .5],
                              [.5, .5]]], device=device, dtype=dtype)  # 4x2x2

        expected = torch.tensor([[[0., 0.],
                                  [0., 0.]],

                                 [[.5, .5],
                                  [.5, .5]],

                                 [[.375, .375],
                                  [.375, .375]],

                                 [[1., 1.],
                                  [1., 1.]]], device=device, dtype=dtype)  # 4x2x2

        factor = torch.tensor([0, 1, 1.5, 2], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_color(self, device, dtype):
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
                               [.6, .6]]]], device=device, dtype=dtype)  # 2x3x2x2

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
                                   [1., 1.]]]], device=device, dtype=dtype)  # 2x3x2x2

        factor = torch.tensor([1, 2], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_shape(self, device, dtype):
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
                               [.3, .2, .1]]]], device=device, dtype=dtype)  # 2x3x2x3

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
                                   [.6, .4, .2]]]], device=device, dtype=dtype)  # 2x3x2x3

        factor = torch.tensor([1.5, 2.], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_contrast, (img, 2.),
                         raise_exception=True)


class TestAdjustBrightness:
    def test_factor_zero(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustBrightness(0.)
        assert_allclose(f(data), expected)

    def test_factor_one(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.ones_like(data)

        f = kornia.enhance.AdjustBrightness(1.)
        assert_allclose(f(data), expected)

    def test_factor_minus(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.75, .75],
                              [.75, .75]],

                             [[.25, .25],
                              [.25, .25]]], device=device, dtype=dtype)  # 3x2x2

        expected = torch.tensor([[[.5, .5],
                                  [.5, .5]],

                                 [[.25, .25],
                                  [.25, .25]],

                                 [[0., 0.],
                                  [0., 0.]]], device=device, dtype=dtype)  # 3x2x2

        f = kornia.enhance.AdjustBrightness(-0.5)
        assert_allclose(f(data), expected)

    def test_factor_tensor(self, device, dtype):
        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[.5, .5],
                              [.5, .5]],

                             [[.25, .25],
                              [.25, .25]],

                             [[.5, .5],
                              [.5, .5]]], device=device, dtype=dtype)  # 4x2x2

        factor = torch.tensor([0, 0.5, 0.75, 2], device=device, dtype=dtype)

        expected = torch.ones_like(data)

        f = kornia.enhance.AdjustBrightness(factor)
        assert_allclose(f(data), expected)

    def test_factor_tensor_color(self, device, dtype):
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
                               [.6, .6]]]], device=device, dtype=dtype)  # 2x3x2x2

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
                                   [.7, .7]]]], device=device, dtype=dtype)  # 2x3x2x2

        factor = torch.tensor([0.25, 0.1], device=device, dtype=dtype)

        f = kornia.enhance.AdjustBrightness(factor)
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.adjust_brightness, (img, 2.),
                         raise_exception=True)


class TestEqualize:
    def test_shape_equalize(self, device, dtype):
        bs, channels, height, width = 1, 3, 4, 5

        inputs = torch.ones(channels, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size([bs, channels, height, width])

    def test_shape_equalize_batch(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5

        inputs = torch.ones(bs, channels, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size([bs, channels, height, width])

    def test_equalize(self, device, dtype):
        bs, channels, height, width = 1, 3, 20, 20

        inputs = self.build_input(bs, channels, height, width, device=device, dtype=dtype)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ], device=device, dtype=dtype)

        expected = self.build_input(bs, channels, height, width,
                                    device=device, dtype=dtype, row=row_expected)

        f = kornia.enhance.equalize

        assert_allclose(f(inputs), expected, rtol=1e-4, atol=1e-4)

    def test_equalize_batch(self, device, dtype):
        bs, channels, height, width = 2, 3, 20, 20

        inputs = self.build_input(bs, channels, height, width, device=device, dtype=dtype)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ], device=device, dtype=dtype)

        expected = self.build_input(bs, channels, height, width,
                                    device=device, dtype=dtype, row=row_expected)

        f = kornia.enhance.equalize

        assert_allclose(f(inputs), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 1, 2, 3, 3
        inputs = torch.ones(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(kornia.enhance.equalize, (inputs,),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 3
        inp = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize
        op_script = torch.jit.script(op)

        assert_allclose(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        batch = torch.stack([image] * batch_size).to(device, dtype)

        return batch


class TestEqualize3D:
    def test_shape_equalize3d(self, device, dtype):
        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = torch.ones(channels, depth, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size([bs, channels, depth, height, width])

    def test_shape_equalize3d_batch(self, device, dtype):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = torch.ones(bs, channels, depth, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size([bs, channels, depth, height, width])

    def test_equalize3d(self, device, dtype):
        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(bs, channels, depth, height, width, device, dtype)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ], device=device, dtype=dtype)

        expected = self.build_input(
            bs, channels, depth, height, width, device, dtype, row=row_expected)

        f = kornia.enhance.equalize3d

        assert_allclose(f(inputs3d), expected, atol=1e-4, rtol=1e-4)

    def test_equalize3d_batch(self, device, dtype):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(
            bs, channels, depth, height, width, device, dtype)

        row_expected = torch.tensor([
            0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000
        ], device=device, dtype=dtype)

        expected = self.build_input(
            bs, channels, depth, height, width, device, dtype, row=row_expected)

        f = kornia.enhance.equalize3d

        assert_allclose(f(inputs3d), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        bs, channels, depth, height, width = 1, 2, 3, 4, 5
        inputs3d = torch.ones(bs, channels, depth, height, width, device=device, dtype=dtype)
        inputs3d = tensor_to_gradcheck_var(inputs3d)
        assert gradcheck(kornia.enhance.equalize3d, (inputs3d,),
                         raise_exception=True)

    def test_jit(self, device, dtype):
        batch_size, channels, depth, height, width = 1, 2, 1, 3, 3
        inp = torch.ones(batch_size, channels, depth, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize3d
        op_script = torch.jit.script(op)

        assert_allclose(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, depth, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        image3d = torch.stack([image] * depth).transpose(0, 1).to(device, dtype)
        batch = torch.stack([image3d] * batch_size).to(device, dtype)

        return batch


class TestSharpness(BaseTester):

    f = kornia.enhance.sharpness

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestSharpness.f(img, 0.8), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width, factor", [
        (1, 4, 5, 0.0),
        (1, 4, 5, 0.8),
        (2, 4, 5, 0.8),
        (1, 4, 5, torch.tensor(0.8)),
        (2, 4, 5, torch.tensor(0.8)),
        (2, 4, 5, torch.tensor([0.8, 0.7]))
    ])
    @pytest.mark.parametrize("channels", [1, 3, 5])
    def test_cardinality(self, batch_size, channels, height, width, factor, device, dtype):
        inputs = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        assert TestSharpness.f(inputs, factor).shape == torch.Size([batch_size, channels, height, width])

    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, [0.8, 0.9, 0.6])
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, torch.tensor([0.8, 0.9, 0.6]))
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, torch.tensor([0.8]))

    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray(arr)
        # en = ImageEnhance.Sharpness(img).enhance(0.8)
        # np.array(en) / 255.
        expected = torch.tensor([
            [[[0.4963, 0.7682, 0.0885],
              [0.1320, 0.3305, 0.6341],
              [0.4901, 0.8964, 0.4556]]]], device=device, dtype=dtype)

        # If factor == 1, shall return original
        # TODO(jian): add test for this case
        # assert_allclose(TestSharpness.f(inputs, 0.), inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(TestSharpness.f(inputs, 1.), inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(TestSharpness.f(inputs, 0.8), expected, rtol=1e-4, atol=1e-4)

    def test_value_batch(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray(arr)
        # en = ImageEnhance.Sharpness(img).enhance(0.8)
        # np.array(en) / 255.
        expected_08 = torch.tensor([
            [[[0.4963, 0.7682, 0.0885],
              [0.1320, 0.3305, 0.6341],
              [0.4901, 0.8964, 0.4556]]],
            [[[0.6323, 0.3489, 0.4017],
              [0.0223, 0.2052, 0.2939],
              [0.5185, 0.6977, 0.8000]]]], device=device, dtype=dtype)

        expected_08_13 = torch.tensor([
            [[[0.4963, 0.7682, 0.0885],
              [0.1320, 0.3305, 0.6341],
              [0.4901, 0.8964, 0.4556]]],
            [[[0.6323, 0.3489, 0.4017],
              [0.0223, 0.1143, 0.2939],
              [0.5185, 0.6977, 0.8000]]]], device=device, dtype=dtype)

        # If factor == 1, shall return original
        tol_val: float = utils._get_precision(device, dtype)
        assert_allclose(TestSharpness.f(inputs, 1), inputs, rtol=tol_val, atol=tol_val)
        assert_allclose(TestSharpness.f(inputs, torch.tensor([1., 1.])), inputs, rtol=tol_val, atol=tol_val)
        assert_allclose(TestSharpness.f(inputs, 0.8), expected_08, rtol=tol_val, atol=tol_val)
        assert_allclose(TestSharpness.f(inputs, torch.tensor([0.8, 1.3])), expected_08_13, rtol=tol_val, atol=tol_val)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestSharpness.f, (inputs, 0.8), raise_exception=True)

    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = torch.jit.script(kornia.enhance.adjust.sharpness)
        inputs = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(input, 0.8)
        actual = op_script(input, 0.8)
        assert_allclose(actual, expected)

    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        # gray_ops = kornia.enhance.sharpness().to(device, dtype)
        # assert_allclose(gray_ops(img), f(img))


@pytest.mark.skipif(kornia.xla_is_available(), reason="issues with xla device")
class TestSolarize(BaseTester):

    f = kornia.enhance.solarize

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestSolarize.f(img, 0.8), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width, thresholds, additions", [
        (1, 4, 5, 0.8, None),
        (1, 4, 5, 0.8, 0.4),
        (2, 4, 5, 0.8, None),
        (1, 4, 5, torch.tensor(0.8), None),
        (2, 4, 5, torch.tensor(0.8), None),
        (2, 4, 5, torch.tensor([0.8, 0.7]), None),
        (2, 4, 5, torch.tensor([0.8, 0.7]), torch.tensor([0., 0.4]))
    ])
    @pytest.mark.parametrize("channels", [1, 3, 5])
    def test_cardinality(self, batch_size, channels, height, width, thresholds, additions, device, dtype):
        inputs = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        assert TestSolarize.f(inputs, thresholds, additions).shape == torch.Size([batch_size, channels, height, width])

    # TODO(jian): add better assertions
    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert TestSolarize.f([1.], 0.)

        with pytest.raises(TypeError):
            assert TestSolarize.f(img, 1)

        with pytest.raises(TypeError):
            assert TestSolarize.f(img, 0.8, 1)

    # TODO: add better cases
    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray((255*inputs[0,0]).byte().numpy())
        # en = ImageOps.Solarize(img, 128)
        # np.array(en) / 255.
        expected = torch.tensor([
            [[[0.49411765, 0.23529412, 0.08627451],
              [0.12941176, 0.30588235, 0.36862745],
              [0.48627451, 0.10588235, 0.45490196]]]], device=device, dtype=dtype)

        # TODO(jian): precision is very bad compared to PIL
        assert_allclose(TestSolarize.f(inputs, 0.5), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestSolarize.f, (inputs, 0.8), raise_exception=True)

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = torch.jit.script(kornia.enhance.adjust.solarize)
        inputs = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(input, 0.8)
        actual = op_script(input, 0.8)
        assert_allclose(actual, expected)

    # TODO: implement me
    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        # gray_ops = kornia.enhance.sharpness().to(device, dtype)
        # assert_allclose(gray_ops(img), f(img))


class TestPosterize(BaseTester):

    f = kornia.enhance.posterize

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestPosterize.f(img, 8), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width, bits", [
        (1, 4, 5, 8),
        (2, 4, 5, 1),
        (2, 4, 5, 0),
        (1, 4, 5, torch.tensor(8)),
        (2, 4, 5, torch.tensor(8)),
        (2, 4, 5, torch.tensor([0, 8])),
        (3, 4, 5, torch.tensor([0, 1, 8]))
    ])
    @pytest.mark.parametrize("channels", [1, 3, 5])
    def test_cardinality(self, batch_size, channels, height, width, bits, device, dtype):
        inputs = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        assert TestPosterize.f(inputs, bits).shape == torch.Size([batch_size, channels, height, width])

    # TODO(jian): add better assertions
    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert TestPosterize.f([1.], 0.)

        with pytest.raises(TypeError):
            assert TestPosterize.f(img, 1.)

    # TODO(jian): add better cases
    @pytest.mark.skipif(kornia.xla_is_available(), reason="issues with xla device")
    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray((255*inputs[0,0]).byte().numpy())
        # en = ImageOps.posterize(img, 1)
        # np.array(en) / 255.
        expected = torch.tensor([
            [[[0., 0.50196078, 0.],
              [0., 0., 0.50196078],
              [0., 0.50196078, 0.]]]], device=device, dtype=dtype)

        assert_allclose(TestPosterize.f(inputs, 1), expected)
        assert_allclose(TestPosterize.f(inputs, 0), torch.zeros_like(inputs))
        assert_allclose(TestPosterize.f(inputs, 8), inputs)

    @pytest.mark.skip(reason="IndexError: tuple index out of range")
    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestPosterize.f, (inputs, 0), raise_exception=True)

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = torch.jit.script(kornia.enhance.adjust.posterize)
        inputs = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(input, 8)
        actual = op_script(input, 8)
        assert_allclose(actual, expected)

    # TODO: implement me
    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        # gray_ops = kornia.enhance.sharpness().to(device, dtype)
        # assert_allclose(gray_ops(img), f(img))
