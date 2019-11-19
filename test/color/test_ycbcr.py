import pytest

import kornia
import kornia.testing as utils  # test utils

import cv2
import numpy as np

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToYcbcr:

    def test_rgb_to_ycbcr(self):

        data = torch.rand(3, 5, 5)

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2YCrCb)

        y_expected = expected[:, :, 0]
        cr_expected = expected[:, :, 1]
        cb_expected = expected[:, :, 2]

        # Kornia
        f = kornia.color.RgbToYcbcr()
        result = f(data)

        y = result[0, :, :]
        cb = result[1, :, :]
        cr = result[2, :, :]

        assert_allclose(y, y_expected)
        assert_allclose(cb, cb_expected)
        assert_allclose(cr, cr_expected)

    def test_batch_rgb_to_ycbcr(self):

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        expected = cv2.cvtColor(data_cv, cv2.COLOR_RGB2YCrCb)
        expected = expected.transpose(2, 0, 1)
        expected[1:] = expected[-1:0:-1]

        # Kornia
        f = kornia.color.RgbToYcbcr()
        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5

        expected = np.expand_dims(expected, 0)
        expected = expected.repeat(2, 0)  # 2x3x5x5

        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        data = torch.rand(3, 5, 5)  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToYcbcr(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
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

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        data_cv[..., 1:] = data_cv[..., -1:0:-1]
        expected = cv2.cvtColor(data_cv, cv2.COLOR_YCrCb2RGB)

        r_expected = expected[:, :, 0]
        g_expected = expected[:, :, 1]
        b_expected = expected[:, :, 2]

        # Kornia
        f = kornia.color.YcbcrToRgb()
        result = f(data)

        r = result[0, :, :]
        g = result[1, :, :]
        b = result[2, :, :]

        assert_allclose(r, r_expected)
        assert_allclose(g, g_expected)
        assert_allclose(b, b_expected)

    def test_batch_ycbcr_to_rgb(self):

        data = torch.rand(3, 5, 5)  # 3x5x5

        # OpenCV
        data_cv = data.numpy().transpose(1, 2, 0).copy()
        data_cv[..., 1:] = data_cv[..., -1:0:-1]
        expected = cv2.cvtColor(data_cv, cv2.COLOR_YCrCb2RGB)

        expected = expected.transpose(2, 0, 1)

        # Kornia
        f = kornia.color.YcbcrToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x5x5

        expected = np.expand_dims(expected, 0)
        expected = expected.repeat(2, 0)  # 2x3x5x5

        assert_allclose(f(data), expected)

    def test_gradcheck(self):

        data = torch.rand(3, 5, 5)  # 3x5x5
        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.YcbcrToRgb(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
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
