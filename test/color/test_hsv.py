import pytest

import kornia
from kornia.geometry import pi
import kornia.testing as utils  # test utils
from test.common import device

import cv2
import numpy as np
import math

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToHsv:

    def test_rgb_to_hsv(self, device):

        data = torch.rand(3, 5, 5).to(device)

        # OpenCV
        data_cv = kornia.tensor_to_image(data.clone())
        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2HSV)
        expected = kornia.image_to_tensor(expected, True).to(device)

        h_expected = 2 * math.pi * expected[0] / 360.
        s_expected = expected[1]
        v_expected = expected[2]

        f = kornia.color.RgbToHsv()
        result = f(data)

        h = result[0, :, :]
        s = result[1, :, :]
        v = result[2, :, :]

        assert_allclose(h, h_expected)
        assert_allclose(s, s_expected)
        assert_allclose(v, v_expected)

    def test_batch_rgb_to_hsv(self, device):

        data = torch.rand(3, 5, 5).to(device)  # 3x5x5

        # OpenCV
        data_cv = kornia.tensor_to_image(data.clone())

        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2HSV)
        expected = kornia.image_to_tensor(expected, False).to(device)
        expected[:, 0] = 2 * math.pi * expected[:, 0] / 360.

        # Kornia
        f = kornia.color.RgbToHsv()

        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x5x5

        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):

        data = torch.rand(3, 5, 5).to(device)  # 3x2x2
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

        data = torch.rand(3, 5, 5).to(device)  # 3x5x5

        # OpenCV
        data_cv = kornia.tensor_to_image(data.clone())
        data_cv[:, :, 0] = 360 * data_cv[:, :, 0]

        expected = cv2.cvtColor(data_cv, cv2.COLOR_HSV2RGB)
        expected = kornia.image_to_tensor(expected, True).to(device)

        r_expected = expected[0]
        g_expected = expected[1]
        b_expected = expected[2]

        # Kornia
        f = kornia.color.HsvToRgb()
        data[0] = 2 * pi * data[0]
        result = f(data)

        r = result[0, :, :]
        g = result[1, :, :]
        b = result[2, :, :]

        assert_allclose(r, r_expected)
        assert_allclose(g, g_expected)
        assert_allclose(b, b_expected)

    def test_batch_hsv_to_rgb(self, device):

        data = torch.rand(3, 5, 5).to(device)  # 3x5x5

        # OpenCV
        data_cv = kornia.tensor_to_image(data.clone())
        data_cv[:, :, 0] = 360 * data_cv[:, :, 0]

        expected = cv2.cvtColor(data_cv, cv2.COLOR_HSV2RGB)
        expected = kornia.image_to_tensor(expected, False).to(device)

        # Kornia
        f = kornia.color.HsvToRgb()

        data[0] = 2 * pi * data[0]
        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x5x5

        assert_allclose(f(data), expected)

        data[:, 0] += 2 * pi
        assert_allclose(f(data), expected)

        data[:, 0] -= 4 * pi
        assert_allclose(f(data), expected)

    def test_gradcheck(self, device):

        data = torch.rand(3, 5, 5).to(device)  # 3x5x5
        data[0] = 2 * pi * data[0]

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.HsvToRgb(), (data,),
                         raise_exception=True)

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
