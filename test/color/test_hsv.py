import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import kornia.color as color
import utils


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

                                 [[21.0000 / 255, 22.0000 / 255],
                                  [22.0000 / 255, 22.0000 / 255]]])

        f = color.RgbToHsv()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

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

                                 [[21.0000 / 255, 22.0000 / 255],
                                  [22.0000 / 255, 22.0000 / 255]]])  # 3x2x2
        f = color.RgbToHsv()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

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


class TestHsvToRgb:

    def test_hsv_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])

        data = torch.tensor([[[0.0641, 0.0714],
                              [0.0714, 0.0714]],

                             [[0.6190, 0.6364],
                              [0.6364, 0.6364]],

                             [[21.0000 / 255, 22.0000 / 255],
                              [22.0000 / 255, 22.0000 / 255]]])

        f = color.HsvToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_batch_hsv_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])  # 3x2x2

        data = torch.tensor([[[0.0641, 0.0714],
                              [0.0714, 0.0714]],

                             [[0.6190, 0.6364],
                              [0.6364, 0.6364]],

                             [[21.0000 / 255, 22.0000 / 255],
                              [22.0000 / 255, 22.0000 / 255]]])  # 3x2x2

        f = color.HsvToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return color.hsv_to_rgb(data)

            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = color.hsv_to_rgb(data)
            assert_allclose(actual, expected)
