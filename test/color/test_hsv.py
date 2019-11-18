import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type
from kornia.geometry import pi

import cv2
import numpy as np
import math

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToHsv:

    def test_rgb_to_hsv(self):

        data = torch.rand(3, 5, 5)

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2HSV)

        h_expected = 2 * math.pi * expected[:, :, 0] / 360.
        s_expected = expected[:, :, 1]
        v_expected = expected[:, :, 2]

        # Kornia
        f = kornia.color.RgbToHsv()
        result = f(data)

        h = result[0, :, :]
        s = result[1, :, :]
        v = result[2, :, :]

        assert_allclose(h, h_expected)
        assert_allclose(s, s_expected)
        assert_allclose(v, v_expected)

    def test_batch_rgb_to_hls(self):

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2HSV)

        expected[:, :, 0] = 2 * math.pi * expected[:, :, 0] / 360.
        expected = expected.transpose(2, 0, 1)

        # Kornia
        f = kornia.color.RgbToHsv()

        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5

        expected = np.expand_dims(expected, 0)
        expected = expected.repeat(2, 0)  # 2x3x5x5

        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        data = torch.rand(3, 5, 5)  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToHsv(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rgb_to_hsv(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.rgb_to_hsv(data)
            assert_allclose(actual, expected)


class TestHsvToRgb:

    def test_hsv_to_rgb(self):

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        data_cv[:, :, 0] = 360 * data_cv[:, :, 0]
        expected = cv2.cvtColor(data_cv, cv2.COLOR_HSV2RGB)

        r_expected = expected[:, :, 0]
        g_expected = expected[:, :, 1]
        b_expected = expected[:, :, 2]

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

    def test_batch_hsv_to_rgb(self):

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        data_cv[:, :, 0] = 360 * data_cv[:, :, 0]
        expected = cv2.cvtColor(data_cv, cv2.COLOR_HSV2RGB)

        expected = expected.transpose(2, 0, 1)

        # Kornia
        f = kornia.color.HsvToRgb()

        data[0] = 2 * pi * data[0]
        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5

        expected = np.expand_dims(expected, 0)
        expected = expected.repeat(2, 0)  # 2x3x5x5

        assert_allclose(f(data), expected)

        data[:, 0] += 2 * pi
        assert_allclose(f(data), expected)

        data[:, 0] -= 4 * pi
        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        data = torch.rand(3, 5, 5)  # 3x5x5
        data[0] = 2 * pi * data[0]

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.HsvToRgb(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.hsv_to_rgb(data)

            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.hsv_to_rgb(data)
            assert_allclose(actual, expected)

    def test_nan(self):
        np.random.seed(0)

        img = (np.random.randint(0, 256, [1000, 1000, 3], np.uint8) / 255.0).astype(np.float32)

        cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        k_img = torch.from_numpy(np.transpose(img, [2, 0, 1]))

        k_img = kornia.rgb_to_hsv(k_img.float())
        k_img[0] = k_img[0] * 360 / (2 * pi)
        k_img = np.transpose(k_img.numpy(), [1, 2, 0])

        assert_allclose(cv_img, k_img)
