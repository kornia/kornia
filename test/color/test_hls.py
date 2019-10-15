import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToHls:

    def test_rgb_to_hls(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])

        expected = torch.tensor([[[0.0641, 0.07138],
                                  [0.07138, 0.07138]],

                                 [[0.0569, 0.0588],
                                  [0.0588, 0.0588]],

                                 [[0.4483, 0.4667],
                                  [0.4667, 0.4667]]])

        f = kornia.color.RgbToHls()
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-7)

    def test_batch_rgb_to_hls(self):

        data = torch.tensor([[[21., 22.],
                              [22., 22.]],

                             [[13., 14.],
                              [14., 14.]],

                             [[8., 8.],
                              [8., 8.]]])  # 3x2x2

        expected = torch.tensor([[[0.0641, 0.07138],
                                  [0.07138, 0.07138]],

                                 [[0.0569, 0.0588],
                                  [0.0588, 0.0588]],

                                 [[0.4483, 0.4667],
                                  [0.4667, 0.4667]]])  # 3x2x2
        f = kornia.color.RgbToHls()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data / 255), expected, atol=1e-4, rtol=1e-7)

    def test_gradcheck(self):

        data = torch.tensor([[[[21., 22.],
                               [22., 22.]],

                              [[13., 14.],
                               [14., 14.]],

                              [[8., 8.],
                               [8., 8.]]]])  # 3x2x2

        data = utils.tensor_to_gradcheck_var(data)  # to var

        assert gradcheck(kornia.color.RgbToHls(), (data,),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:

            return kornia.rgb_to_hls(data)
            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.rgb_to_hls(data)
            assert_allclose(actual, expected)


class TestHlsToRgb:

    def test_hls_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])

        data = torch.tensor(      [[[0.0641, 0.07138],
                                  [0.07138, 0.07138]],

                                 [[0.0569, 0.0588],
                                  [0.0588, 0.0588]],

                                 [[0.4483, 0.4667],
                                  [0.4667, 0.4667]]])

        f = kornia.color.HlsToRgb()
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-7)

    def test_batch_hls_to_rgb(self):

        expected = torch.tensor([[[21., 22.],
                                  [22., 22.]],

                                 [[13., 14.],
                                  [14., 14.]],

                                 [[8., 8.],
                                  [8., 8.]]])  # 3x2x2

        data = torch.tensor([[[0.0641, 0.07138],
                                  [0.07138, 0.07138]],

                                 [[0.0569, 0.0588],
                                  [0.0588, 0.0588]],

                                 [[0.4483, 0.4667],
                                  [0.4667, 0.4667]]])  # 3x2x2

        f = kornia.color.HlsToRgb()
        data = data.repeat(2, 1, 1, 1)  # 2x3x2x2
        expected = expected.repeat(2, 1, 1, 1)  # 2x3x2x2
        assert_allclose(f(data), expected / 255, atol=1e-3, rtol=1e-7)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> torch.Tensor:
            return kornia.hls_to_rgb(data)

            data = torch.tensor([[[[21., 22.],
                                   [22., 22.]],

                                  [[13., 14.],
                                   [14., 14.]],

                                  [[8., 8.],
                                   [8., 8.]]]])  # 3x2x2

            actual = op_script(data)
            expected = kornia.hls_to_rgb(data)
            assert_allclose(actual, expected)
