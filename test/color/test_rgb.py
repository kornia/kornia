import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from common import device_type

import kornia.color as color
import utils


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
