import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToYcbcr:

    def test_rgb_to_ycbcr(self):

        data = torch.tensor([[[255., 0., 0.],
                              [255., 0., 0.],
                              [255., 0., 0.]],

                             [[0., 255., 0.],
                              [0., 255., 0.],
                              [0., 255., 0.]],

                             [[0., 0., 255.],
                              [0., 0., 255.],
                              [0., 0., 255.]]]) / 255

        expected = torch.tensor([[[76, 149, 29],
                                  [76, 149, 29],
                                  [76, 149, 29]],

                                 [[84, 43, 255],
                                  [84, 43, 255],
                                  [84, 43, 255]],

                                 [[255, 21, 107],
                                  [255, 21, 107],
                                  [255, 21, 107]]]) / 255

        f = kornia.color.RgbToYcbcr()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

    def test_batch_rgb_to_ycbcr(self):

        data = torch.tensor([[[255., 0., 0.],
                              [255., 0., 0.],
                              [255., 0., 0.]],

                             [[0., 255., 0.],
                              [0., 255., 0.],
                              [0., 255., 0.]],

                             [[0., 0., 255.],
                              [0., 0., 255.],
                              [0., 0., 255.]]]) / 255

        expected = torch.tensor([[[76, 149, 29],
                                  [76, 149, 29],
                                  [76, 149, 29]],

                                 [[84, 43, 255],
                                  [84, 43, 255],
                                  [84, 43, 255]],

                                 [[255, 21, 107],
                                  [255, 21, 107],
                                  [255, 21, 107]]]) / 255

        f = kornia.color.RgbToYcbcr()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self):

        data = torch.tensor([[[255., 0., 0.],
                              [255., 0., 0.],
                              [255., 0., 0.]],

                             [[0., 255., 0.],
                              [0., 255., 0.],
                              [0., 255., 0.]],

                             [[0., 0., 255.],
                              [0., 0., 255.],
                              [0., 0., 255.]]]) / 255

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToYcbcr(), (data,),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rgb_to_ycbcr(data)
            data = torch.tensor([[[255., 0., 0.],
                                  [255., 0., 0.],
                                  [255., 0., 0.]],

                                 [[0., 255., 0.],
                                  [0., 255., 0.],
                                  [0., 255., 0.]],

                                 [[0., 0., 255.],
                                  [0., 0., 255.],
                                  [0., 0., 255.]]]) / 255

            actual = op_script(data)
            expected = kornia.rgb_to_ycbcr(data)
            assert_allclose(actual, expected)


class TestYcbcrToRgb:

    def test_ycbcr_to_rgb(self):

        expected = torch.tensor([[[255., 0., 0.],
                                  [255., 0., 0.],
                                  [255., 0., 0.]],

                                 [[0., 255., 0.],
                                  [0., 255., 0.],
                                  [0., 255., 0.]],

                                 [[0., 0., 255.],
                                  [0., 0., 255.],
                                  [0., 0., 255.]]]) / 255

        data = torch.tensor([[[76, 149, 29],
                              [76, 149, 29],
                              [76, 149, 29]],

                             [[84, 43, 255],
                              [84, 43, 255],
                              [84, 43, 255]],

                             [[255, 21, 107],
                              [255, 21, 107],
                              [255, 21, 107]]]) / 255

        f = kornia.color.YcbcrToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_batch_ycbcr_to_rgb(self):

        expected = torch.tensor([[[255., 0., 0.],
                                  [255., 0., 0.],
                                  [255., 0., 0.]],

                                 [[0., 255., 0.],
                                  [0., 255., 0.],
                                  [0., 255., 0.]],

                                 [[0., 0., 255.],
                                  [0., 0., 255.],
                                  [0., 0., 255.]]]) / 255

        data = torch.tensor([[[76, 149, 29],
                              [76, 149, 29],
                              [76, 149, 29]],

                             [[84, 43, 255],
                              [84, 43, 255],
                              [84, 43, 255]],

                             [[255, 21, 107],
                              [255, 21, 107],
                              [255, 21, 107]]]) / 255

        f = kornia.color.YcbcrToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.ycbcr_to_rgb(data)

            data = torch.tensor([[[255., 0., 0.],
                                  [255., 0., 0.],
                                  [255., 0., 0.]],

                                 [[0., 255., 0.],
                                  [0., 255., 0.],
                                  [0., 255., 0.]],

                                 [[0., 0., 255.],
                                  [0., 0., 255.],
                                  [0., 0., 255.]]]) / 255

            actual = op_script(data)
            expected = kornia.ycbcr_to_rgb(data)
            assert_allclose(actual, expected)
