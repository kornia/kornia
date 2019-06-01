import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestPyrUp:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(kornia.geometry.pyrup, (img,), raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrup(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.geometry.pyrup(img)
        assert_allclose(actual, expected)


class TestPyrDown:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(kornia.geometry.pyrdown, (img,), raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrdown(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.geometry.pyrdown(img)
        assert_allclose(actual, expected)
