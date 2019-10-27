import pytest
import random

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


def random_shape(dim, min_elem=1, max_elem=10):
    return tuple(random.randint(min_elem, max_elem) for _ in range(dim))


class TestAddWeighted:
    def get_input(self, size, max_elem=10):
        shape = random_shape(size, max_elem)
        src1 = torch.randn(shape)
        src2 = torch.randn(shape)
        alpha = random.random()
        beta = random.random()
        gamma = random.random()
        return src1, src2, alpha, beta, gamma

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_addweighted(self, size):
        src1, src2, alpha, beta, gamma = self.get_input(3)

        f = kornia.color.AddWeighted(alpha, beta, gamma)
        assert_allclose(f(src1, src2), src1 * alpha + src2 * beta + gamma)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(src1: torch.Tensor, alpha: float, src2: torch.Tensor,
                      beta: float, gamma: float) -> torch.Tensor:
            return kornia.color.add_weighted(src1, alpha, src2, beta, gamma)

        src1, src2, alpha, beta, gamma = self.get_input(3)

        actual = op_script(src1, alpha, src2, beta, gamma)
        expected = kornia.color.add_weighted(src1, alpha, src2, beta, gamma)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize("size", [2, 3])
    def test_gradcheck(self, size):
        shape = random_shape(size, max_elem=5)  # to shave time on gradcheck
        src1 = torch.randn(shape)
        src2 = torch.randn(shape)
        alpha = random.random()
        beta = random.random()
        gamma = random.random()

        src1 = utils.tensor_to_gradcheck_var(src1)  # to var
        src2 = utils.tensor_to_gradcheck_var(src2)  # to var

        assert gradcheck(kornia.color.AddWeighted(alpha, beta, gamma), (src1, src2),
                         raise_exception=True)
