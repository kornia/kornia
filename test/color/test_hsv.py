import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToHsv:

    def test_rgb_to_hsv(self, device):

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

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.RgbToHsv()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

    def test_batch_rgb_to_hsv(self, device):

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

        data = data.to(device)
        expected = expected.to(device)

        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2

        f = kornia.color.RgbToHsv()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self, device):

        data = torch.tensor([[[[21., 22.],
                               [22., 22.]],

                              [[13., 14.],
                               [14., 14.]],

                              [[8., 8.],
                               [8., 8.]]]])  # 3x2x2

        data = data.to(device)
        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToHsv(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rgb_to_hsv(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            data = data.to(device)
            actual = op_script(data)
            expected = kornia.rgb_to_hsv(data)
            assert_allclose(actual, expected)


class TestHsvToRgb:

    def test_hsv_to_rgb(self, device):

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

        data = data.to(device)
        expected = expected.to(device)

        f = kornia.color.HsvToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    def test_batch_hsv_to_rgb(self, device):

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

        data = data.to(device)
        expected = expected.to(device)

        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2

        f = kornia.color.HsvToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-3)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.hsv_to_rgb(data)

            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            data = data.to(device)
            actual = op_script(data)
            expected = kornia.hsv_to_rgb(data)
            assert_allclose(actual, expected)
