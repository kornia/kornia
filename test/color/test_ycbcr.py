import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToYcbcr:

    def test_rgb_to_ycbcr(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])

        expected = torch.tensor([[[0.05882353, 0.0627451],
                                  [0.0627451, 0.0627451]],

                                 [[0.4862745, 0.48235294],
                                  [0.48235294, 0.48235294]],

                                 [[0.5176471, 0.5176471],
                                  [0.5176471, 0.5176471]]])

        f = kornia.color.RgbToYcbcr()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

    def test_batch_rgb_to_ycbcr(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])  # 3x2x2

        expected = torch.tensor([[[0.05882353, 0.0627451],
                                  [0.0627451, 0.0627451]],

                                 [[0.4862745, 0.48235294],
                                  [0.48235294, 0.48235294]],

                                 [[0.5176471, 0.5176471],
                                  [0.5176471, 0.5176471]]])  # 3x2x2
        f = kornia.color.RgbToYcbcr()
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

        assert gradcheck(kornia.color.RgbToYcbcr(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rgb_to_ycbcr(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.rgb_to_ycbcr(data)
            assert_allclose(actual, expected)


class TestYcbcrToRgb:

    def test_ycbcr_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])

        data = torch.tensor([[[0.05882353, 0.0627451],
                              [0.0627451, 0.0627451]],

                             [[0.4862745, 0.48235294],
                                 [0.48235294, 0.48235294]],

                             [[0.5176471, 0.5176471],
                                 [0.5176471, 0.5176471]]])

        f = kornia.color.YcbcrToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_batch_ycbcr_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])  # 3x2x2

        data = torch.tensor([[[0.05882353, 0.0627451],
                              [0.0627451, 0.0627451]],

                             [[0.4862745, 0.48235294],
                                 [0.48235294, 0.48235294]],

                             [[0.5176471, 0.5176471],
                                 [0.5176471, 0.5176471]]])  # 3x2x2

        f = kornia.color.YcbcrToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.ycbcr_to_rgb(data)

            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.ycbcr_to_rgb(data)
            assert_allclose(actual, expected)
