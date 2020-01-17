import kornia
import kornia.testing as utils  # test utils
from test.common import device

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
                                 [[0.4, 0.4],
                                  [0.4, 0.4]]])  # 4x2x2

        f = kornia.color.RgbToRgba(aval)
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
                                  [[45., 45.], [45., 45.]]],
                                 [[[1.0000e+00, 1.0000e+00], [1.0000e+00, 1.0000e+00]],
                                  [[2.0000e+00, 2.0000e+00], [2.0000e+00, 2.0000e+00]],
                                  [[3.0000e+00, 3.0000e+00], [3.0000e+00, 3.0000e+00]],
                                  [[45., 45.], [45., 45.]]]])  # 2x4x2x2

        f = kornia.color.RgbToRgba(aval)
        out = f(data, aval)
        assert_allclose(out, expected)

        
class TestBgrToRgb:

    def test_bgr_to_rgb(self, device):

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
                             
        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.BgrToRgb()
        assert_allclose(f(data), expected)

    def test_gradcheck(self):
        aval = 0.4
        data = torch.Tensor([[[1., 1.], [1., 1.]], [[2., 2.], [2., 2.]], [[3., 3.], [3., 3.]]])  # 3x2x2
        data = utils.tensor_to_gradcheck_var(data)  # to var
        assert gradcheck(kornia.color.RgbToRgba(aval), (data, aval), raise_exception=True)

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

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.BgrToRgb()
        out = f(data)
        assert_allclose(out, expected)

    def test_gradcheck(self, device):

        data = torch.Tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2

        data = data.to(device)
        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.BgrToRgb(), (data,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
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

    def test_rgb_to_bgr(self, device):

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

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.RgbToBgr()
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):

        # prepare input data
        data = torch.Tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2

        data = data.to(device)
        data = utils.tensor_to_gradcheck_var(data)  # to var

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

    def test_batch_rgb_to_bgr(self, device):

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

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.RgbToBgr()
        out = f(data)
        assert_allclose(out, expected)
