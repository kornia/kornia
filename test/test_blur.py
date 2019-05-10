import pytest
from typing import Tuple

import torch
import torchgeometry as tgm
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import utils  # test utils


class TestBoxBlur:
    def test_shape(self):
        inp = torch.zeros(1, 3, 4, 4)
        blur = tgm.image.BoxBlur((3, 3))
        assert blur(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        blur = tgm.image.BoxBlur((3, 3))
        assert blur(inp).shape == (2, 6, 4, 4)

    def test_kernel_3x3(self):
        inp = torch.tensor([[[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.]
        ]]])

        kernel_size = (3, 3)
        actual = tgm.image.box_blur(inp, kernel_size)
        assert_allclose(actual[0, 0, 1, 1:4], torch.tensor(1.))

    def test_kernel_5x5(self):
        inp = torch.tensor([[[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.]
        ]]])

        kernel_size = (5, 5)
        actual = tgm.image.box_blur(inp, kernel_size)
        assert_allclose(actual[0, 0, 1, 2], torch.tensor(1.))

    def test_kernel_5x5_batch(self):
        batch_size = 3
        inp = torch.tensor([[[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.]
        ]]]).repeat(batch_size, 1, 1, 1)

        kernel_size = (5, 5)
        actual = tgm.image.box_blur(inp, kernel_size)
        assert_allclose(actual[0, 0, 1, 2], torch.tensor(1.))

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(tgm.image.box_blur, (img, (3, 3),),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      kernel_size: Tuple[int, int]) -> torch.Tensor:
            return tgm.image.box_blur(input, kernel_size)
        kernel_size = (3, 3)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, kernel_size)
        expected = tgm.image.box_blur(img, kernel_size)
        assert_allclose(actual, expected)
