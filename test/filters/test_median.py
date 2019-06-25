from typing import Tuple

import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestMedianBlur:
    def test_shape(self):
        inp = torch.zeros(1, 3, 4, 4)
        median = kornia.filters.MedianBlur((3, 3))
        assert median(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        blur = kornia.filters.BoxBlur((3, 3))
        assert blur(inp).shape == (2, 6, 4, 4)

    def test_kernel_3x3(self):
        inp = torch.tensor([[
            [0., 0., 0., 0., 0.],
            [0., 3., 7., 5., 0.],
            [0., 3., 1., 1., 0.],
            [0., 6., 9., 2., 0.],
            [0., 0., 0., 0., 0.]
        ], [
            [36., 7.0, 25., 0., 0.],
            [3.0, 14., 1.0, 0., 0.],
            [65., 59., 2.0, 0., 0.],
            [0.0, 0.0, 0.0, 0., 0.],
            [0.0, 0.0, 0.0, 0., 0.]
        ]]).repeat(2, 1, 1, 1)

        kernel_size = (3, 3)
        actual = kornia.filters.median_blur(inp, kernel_size)
        assert_allclose(actual[0, 0, 2, 2], torch.tensor(3.))
        assert_allclose(actual[0, 1, 1, 1], torch.tensor(14.))

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.median_blur, (img, (5, 3),),
                         raise_exception=True)

    @pytest.mark.skip("")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      kernel_size: Tuple[int, int]) -> torch.Tensor:
            return kornia.filters.median_blur(input, kernel_size)
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, kernel_size)
        expected = kornia.filters.median_blur(img, kernel_size)
        assert_allclose(actual, expected)
