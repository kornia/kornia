from typing import Tuple

import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestBoxBlur:
    def test_shape(self):
        inp = torch.zeros(1, 3, 4, 4)
        blur = kornia.filters.BoxBlur((3, 3))
        assert blur(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        blur = kornia.filters.BoxBlur((3, 3))
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
        actual = kornia.filters.box_blur(inp, kernel_size)
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
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        actual = kornia.filters.box_blur(inp, kernel_size)
        assert_allclose(actual[:, 0, 2, 2], expected)

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
        expected = inp.sum((1, 2, 3)) / torch.mul(*kernel_size)

        actual = kornia.filters.box_blur(inp, kernel_size)
        assert_allclose(actual[:, 0, 2, 2], expected)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.filters.box_blur, (img, (3, 3),),
                         raise_exception=True)

    @pytest.mark.skip(reason="undefined value BoxBlur")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        op = kornia.filters.box_blur
        op_script = torch.jit.script(op)

        kernel_size = (3, 3)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, kernel_size)
        expected = op(img, kernel_size)
        assert_allclose(actual, expected)
