import kornia
import kornia.testing as utils
from test.common import device_type
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
import pytest


class TestRgbToRgba:
    def test_rgb_to_rgba(self):

        data = torch.Tensor([[[1., 1.], [1., 1.]], [[2., 2.], [2., 2.]], [[3., 3.], [3., 3.]]])  # 3x2x2
        aval = 0.4

        expected = torch.Tensor([[[1.0000e+00, 1.0000e+00], [1.0000e+00, 1.0000e+00]],
                                 [[2.0000e+00, 2.0000e+00], [2.0000e+00, 2.0000e+00]],
                                 [[3.0000e+00, 3.0000e+00], [3.0000e+00, 3.0000e+00]],
                                 [[1.3563e-19, 1.5686e-03],
                                  [2.0706e-19, 7.2939e+22]]])  # 4x2x2

        f = kornia.color.RgbToRgba()
        assert_allclose(f(data, aval), expected)

    def test_batch_rgb_to_rgba(self):

        data = torch.Tensor([[[[1., 1.], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]],
                             [[[1., 1.], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]]])  # 2x3x2x2
        aval = 45

        expected = torch.Tensor([[[[1.0000e+00, 1.0000e+00], [1.0000e+00, 1.0000e+00]],
                                  [[2.0000e+00, 2.0000e+00], [2.0000e+00, 2.0000e+00]],
                                  [[3.0000e+00, 3.0000e+00], [3.0000e+00, 3.0000e+00]],
                                  [[8.9683e-44, 0.0000e+00], [8.9683e-44, 0.0000e+00]]],
                                 [[[1.0000e+00, 1.0000e+00], [1.0000e+00, 1.0000e+00]],
                                  [[2.0000e+00, 2.0000e+00], [2.0000e+00, 2.0000e+00]],
                                  [[3.0000e+00, 3.0000e+00], [3.0000e+00, 3.0000e+00]],
                                  [[7.6751e+27, 3.0872e-41], [1.7647e-01, 1.7647e-01]]]])  # 2x4x2x2

        f = kornia.color.RgbToRgba()
        out = f(data, aval)
        assert_allclose(out, expected)

    def test_gradcheck(self):

        data = torch.Tensor([[[1., 1.], [1., 1.]], [[2., 2.], [2., 2.]], [[3., 3.], [3., 3.]]])  # 3x2x2
        aval = 0.4

        data = utils.Tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToRgba(), (data, aval), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor, aval: int) -> torch.Tensor:
            return kornia.rgb_to_rgba(data, aval)

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2
        aval = 0.4
        actual = op_script(data, aval)
        expected = kornia.rgb_to_rgba(data, aval)
        assert_allclose(actual, expected)


class TestBgrToRgb:

    def test_bgr_to_rgb(self):

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2

        expected = torch.Tensor([[[3., 3.], [3., 3.]],
                                 [[2., 2], [2., 2.]],
                                 [[1., 1.], [1., 1.]]])  # 3x2x2

        f = kornia.color.BgrToRgb()
        assert_allclose(f(data), expected)

    def test_batch_bgr_to_rgb(self):

        data = torch.Tensor([[[[1., 1], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]],
                             [[[1., 1.], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]]])  # 2x3x2x2

        expected = torch.Tensor([[[[3., 3.], [3., 3.]],
                                  [[2., 2.], [2., 2.]],
                                  [[1., 1.], [1., 1.]]],
                                 [[[3., 3.], [3., 3.]],
                                  [[2., 2.], [2., 2.]],
                                  [[1., 1.], [1., 1.]]]])  # 2x3x2x2

        f = kornia.color.BgrToRgb()
        out = f(data)
        assert_allclose(out, expected)

    def test_gradcheck(self):

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2

        data = utils.Tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.BgrToRgb(), (data,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.bgr_to_rgb(data)

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2
        actual = op_script(data)
        expected = kornia.bgr_to_rgb(data)
        assert_allclose(actual, expected)


class TestRgbToBgr:

    def test_rgb_to_bgr(self):

        # prepare input data
        data = torch.Tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        expected = torch.Tensor([[[3., 3.],
                                  [3., 3.]],

                                 [[2., 2.],
                                  [2., 2.]],

                                 [[1., 1.],
                                  [1., 1.]]])  # 3x2x2

        f = kornia.color.RgbToBgr()
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        # prepare input data
        data = torch.Tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = utils.Tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToBgr(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.rgb_to_bgr(data)

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x
        actual = op_script(data)
        expected = kornia.rgb_to_bgr(data)
        assert_allclose(actual, expected)

    def test_batch_rgb_to_bgr(self):

        # prepare input data
        data = torch.Tensor([[[[1., 1.],
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

        expected = torch.Tensor([[[[3., 3.],
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

        f = kornia.color.RgbToBgr()
        out = f(data)
        assert_allclose(out, expected)
