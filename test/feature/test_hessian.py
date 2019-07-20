import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestHessianResp:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        out = kornia.feature.HessianResp()
        assert out(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        out = kornia.feature.HessianResp()
        assert out(inp).shape == (2, 6, 4, 4)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.hessian, (img),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input, k):
            return kornia.feature.hessian(input)
        k = torch.tensor(0.04)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.feature.hessian(img)
        assert_allclose(actual, expected)
