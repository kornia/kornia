from typing import Tuple

import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestMedianBlur:
    def test_shape(self, device):
        inp = torch.zeros(1, 3, 4, 4).to(device)
        median = kornia.filters.MedianBlur((3, 3))
        assert median(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 6, 4, 4).to(device)
        blur = kornia.filters.BoxBlur((3, 3))
        assert blur(inp).shape == (2, 6, 4, 4)

    def test_kernel_3x3(self, device):
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
        ]]).repeat(2, 1, 1, 1).to(device)

        kernel_size = (3, 3)
        actual = kornia.filters.median_blur(inp, kernel_size)
        assert_allclose(actual[0, 0, 2, 2], torch.tensor(3.).to(device))
        assert_allclose(actual[0, 1, 1, 1], torch.tensor(14.).to(device))

    def test_noncontiguous(self, device):
        batch_size = 3
        inp = torch.rand(3, 5, 5).expand(batch_size, -1, -1, -1).to(device)

        kernel_size = (3, 3)
        actual = kornia.filters.median_blur(inp, kernel_size)
        expected = actual
        assert_allclose(actual, actual)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.median_blur, (img, (5, 3),),
                         raise_exception=True)

    @pytest.mark.skip("")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      kernel_size: Tuple[int, int]) -> torch.Tensor:
            return kornia.filters.median_blur(input, kernel_size)
        kernel_size = (3, 5)
        img = torch.rand(2, 3, 4, 5).to(device)
        actual = op_script(img, kernel_size)
        expected = kornia.filters.median_blur(img, kernel_size)
        assert_allclose(actual, expected)
