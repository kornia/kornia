import pytest

import torch
import torchgeometry as tgm
from torch.testing import assert_allclose
from torch.autograd import gradcheck

import utils  # test utils


class TestPyrUp:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = tgm.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = tgm.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(tgm.geometry.pyrup, (img,), raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return tgm.geometry.pyrup(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = tgm.geometry.pyrup(img)
        assert_allclose(actual, expected)


class TestPyrDown:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = tgm.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = tgm.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var

        assert gradcheck(tgm.geometry.pyrdown, (img,), raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return tgm.geometry.pyrdown(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = tgm.geometry.pyrdown(img)
        assert_allclose(actual, expected)
