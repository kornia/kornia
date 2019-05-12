import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import torchgeometry.color as color
import utils


class TestNormalize:
    def test_smoke(self):
        mean = [0.5]
        std = [0.1]
        repr = 'Normalize(mean=[0.5], std=[0.1])'
        assert str(color.Normalize(mean, std)) == repr

    def test_normalize(self):

        # prepare input data
        data = torch.ones(1, 2, 2)
        mean = torch.tensor([0.5])
        std = torch.tensor([2.0])

        # expected output
        expected = torch.tensor([0.25]).repeat(1, 2, 2).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_broadcast_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0])
        std = torch.tensor([2.0, 2.0, 2.0])

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_batch_normalize(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2

        mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
        std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

        # expected output
        expected = torch.tensor([1.25, 1, 0.5]).repeat(2, 1, 1).view_as(data)

        f = color.Normalize(mean, std)
        assert_allclose(f(data), expected)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:

            return color.normalize(data, mean, std)

            data = torch.ones(2, 3, 1, 1)
            data += 2

            mean = torch.tensor([0.5, 1.0, 2.0]).repeat(2, 1)
            std = torch.tensor([2.0, 2.0, 2.0]).repeat(2, 1)

            actual = op_script(data, mean, std)
            expected = color.normalize(data, mean, std)
            assert_allclose(actual, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.ones(2, 3, 1, 1)
        data += 2
        mean = torch.tensor([0.5, 1.0, 2.0]).double()
        std = torch.tensor([2., 2., 2.]).double()

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(color.Normalize(mean, std), (data,),
                         raise_exception=True)


class TestRgbToHsv:

    def test_rgb_to_hsv(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])

        expected = torch.tensor([[[0.0641, 0.0714],
                                  [0.0714, 0.0714]],

                                 [[0.6190, 0.6364],
                                  [0.6364, 0.6364]],

                                 [[21.0000, 22.0000],
                                  [22.0000, 22.0000]]])

        f = color.RgbToHsv()
        assert_allclose(f(data), expected, atol=1e-4, rtol=1e-5)

    def test_batch_rgb_to_hsv(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])  # 3x2x2

        expected = torch.tensor([[[0.0641, 0.0714],
                                  [0.0714, 0.0714]],

                                 [[0.6190, 0.6364],
                                  [0.6364, 0.6364]],

                                 [[21.0000, 22.0000],
                                  [22.0000, 22.0000]]])  # 3x2x2
        f = color.RgbToHsv()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        print(data.shape)
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        print(expected.shape)
        print(f(data).shape)
        assert_allclose(f(data), expected, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self):

        data = torch.tensor([[[[21., 22.],
                               [22., 22.]],

                              [[13., 14.],
                               [14., 14.]],

                              [[8., 8.],
                               [8., 8.]]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(color.RgbToHsv(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return color.rgb_to_hsv(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = color.rgb_to_hsv(data)
            assert_allclose(actual, expected)


class TestBgrToRgb:

    def test_bgr_to_rgb(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        expected = torch.tensor([[[3., 3.],
                                  [3., 3.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[1., 1.],
                                  [1., 1.]]])  # 3x2x2

        f = color.BgrToRgb()
        assert_allclose(f(data), expected)

    def test_batch_bgr_to_rgb(self):

        # prepare input data
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]],

                             [[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]]])  # 2x3x2x2

        expected = torch.tensor([[[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]],

                                 [[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]]])  # 2x3x2x2

        f = color.BgrToRgb()
        out = f(data)
        assert_allclose(out, expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(color.BgrToRgb(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return color.bgr_to_rgb(data)

            data = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[3., 3.],
                                  [3., 3.]]])  # 3x2x2

            actual = op_script(data)
            expected = color.bgr_to_rgb(data)
            assert_allclose(actual, expected)


class TestRgbToBgr:

    def test_rgb_to_bgr(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        expected = torch.tensor([[[3., 3.],
                                  [3., 3.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[1., 1.],
                                  [1., 1.]]])  # 3x2x2

        f = color.RgbToBgr()
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(color.RgbToBgr(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return color.rgb_to_bgr(data)

            data = torch.tensor([[[1., 1.],
                                  [1., 1.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[3., 3.],
                                  [3., 3.]]])  # 3x2x2

            actual = op_script(data)
            expected = color.rgb_to_bgr(data)
            assert_allclose(actual, expected)

    def test_batch_rgb_to_bgr(self):

        # prepare input data
        data = torch.tensor([[[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]],

                             [[[1., 1.],
                               [1., 1.]],

                              [[2., 2.],
                               [2., 2.]],

                              [[3., 3.],
                               [3., 3.]]]])  # 2x3x2x2

        expected = torch.tensor([[[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]],

                                 [[[3., 3.],
                                   [3., 3.]],

                                  [[2., 2.],
                                   [2., 2.]],

                                  [[1., 1.],
                                   [1., 1.]]]])  # 2x3x2x2

        f = color.RgbToBgr()
        out = f(data)
        assert_allclose(out, expected)


class TestRgbToGrayscale:
    def test_rgb_to_grayscale(self):
        channels, height, width = 3, 4, 5
        img = torch.ones(channels, height, width)
        assert color.RgbToGrayscale()(img).shape == (1, height, width)

    def test_rgb_to_grayscale_batch(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        assert color.RgbToGrayscale()(img).shape == \
            (batch_size, 1, height, width)

    def test_gradcheck(self):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(color.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self):
        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        gray = color.RgbToGrayscale()
        gray_traced = torch.jit.trace(color.RgbToGrayscale(), img)
        assert_allclose(gray(img), gray_traced(img))
