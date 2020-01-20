import kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
import pytest


class TestRgbToRgba:
    def test_smoke(self, device):
        data = torch.rand(3, 4, 4).to(device)
        assert kornia.rgb_to_rgba(data, 0.).shape == (4, 4, 4)

    def test_back_and_forth_rgb(self, device):
        a_val: float = 1.
        x_rgb = torch.rand(3, 4, 4).to(device)
        x_rgba = kornia.rgb_to_rgba(x_rgb, a_val)
        x_rgb_new = kornia.rgba_to_rgb(x_rgba)
        assert_allclose(x_rgb, x_rgb_new)

    def test_back_and_forth_bgr(self, device):
        a_val: float = 1.
        x_bgr = torch.rand(3, 4, 4).to(device)
        x_rgba = kornia.bgr_to_rgba(x_bgr, a_val)
        x_bgr_new = kornia.rgba_to_bgr(x_rgba)
        assert_allclose(x_bgr, x_bgr_new)

    def test_bgr(self, device):
        a_val: float = 1.
        x_rgb = torch.rand(3, 4, 4).to(device)
        x_bgr = kornia.rgb_to_bgr(x_rgb)
        x_rgba = kornia.rgb_to_rgba(x_rgb, a_val)
        x_rgba_new = kornia.bgr_to_rgba(x_bgr, a_val)
        assert_allclose(x_rgba, x_rgba_new)

    def test_single(self, device):
        data = torch.tensor([[[1., 1.],
                              [1., 1.]],

                             [[2., 2.],
                              [2., 2.]],

                             [[3., 3.],
                              [3., 3.]]])  # 3x2x2
        data = data.to(device)

        aval: float = 0.4
        expected = torch.tensor([[[1.0, 1.0],
                                  [1.0, 1.0]],

                                 [[2.0, 2.0],
                                  [2.0, 2.0]],

                                 [[3.0, 3.0],
                                  [3.0, 3.0]],

                                 [[0.4, 0.4],
                                  [0.4, 0.4]]])  # 4x2x2
        expected = expected.to(device)

        assert_allclose(kornia.rgb_to_rgba(data, aval), expected)

    def test_batch(self, device):

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
        data = data.to(device)

        aval: float = 45.

        expected = torch.tensor([[[[1.0, 1.0],
                                   [1.0, 1.0]],

                                  [[2.0, 2.0],
                                   [2.0, 2.0]],

                                  [[3.0, 3.0],
                                   [3.0, 3.0]],

                                  [[45., 45.],
                                   [45., 45.]]],

                                 [[[1.0, 1.0],
                                   [1.0, 1.0]],

                                  [[2.0, 2.0],
                                   [2.0, 2.0]],

                                  [[3.0, 3.0],
                                   [3.0, 3.0]],

                                  [[45., 45.],
                                   [45., 45.]]]])
        expected = expected.to(device)

        assert_allclose(kornia.rgb_to_rgba(data, aval), expected)

    def test_gradcheck(self, device):
        data = torch.rand(1, 3, 2, 2).to(device)
        data = utils.tensor_to_gradcheck_var(data)  # to var
        assert gradcheck(kornia.color.RgbToRgba(1.), (data,), raise_exception=True)


class TestBgrToRgb:

    def test_back_and_forth(self, device):
        data_bgr = torch.rand(1, 3, 3, 2).to(device)
        data_rgb = kornia.bgr_to_rgb(data_bgr)
        data_bgr_new = kornia.rgb_to_bgr(data_rgb)
        assert_allclose(data_bgr, data_bgr_new)

    def test_bgr_to_rgb(self, device):

        data = torch.tensor([[[1., 1.], [1., 1.]],
                             [[2., 2.], [2., 2.]],
                             [[3., 3.], [3., 3.]]])  # 3x2x2

        expected = torch.tensor([[[3., 3.], [3., 3.]],
                                 [[2., 2.], [2., 2.]],
                                 [[1., 1.], [1., 1.]]])  # 3x2x2

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.BgrToRgb()
        assert_allclose(f(data), expected)

    def test_batch_bgr_to_rgb(self, device):

        data = torch.tensor([[[[1., 1.], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]],
                             [[[1., 1.], [1., 1.]],
                              [[2., 2.], [2., 2.]],
                              [[3., 3.], [3., 3.]]]])  # 2x3x2x2

        expected = torch.tensor([[[[3., 3.], [3., 3.]],
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

        data = torch.tensor([[[1., 1.], [1., 1.]],
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

    def test_back_and_forth(self, device):
        data_rgb = torch.rand(1, 3, 3, 2).to(device)
        data_bgr = kornia.rgb_to_bgr(data_rgb)
        data_rgb_new = kornia.bgr_to_rgb(data_bgr)
        assert_allclose(data_rgb, data_rgb_new)

    def test_rgb_to_bgr(self, device):

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

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.RgbToBgr()
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):

        # prepare input data
        data = torch.tensor([[[1., 1.],
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

        # move data to the device
        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.RgbToBgr()
        out = f(data)
        assert_allclose(out, expected)
