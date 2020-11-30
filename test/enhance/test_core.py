import pytest
import random

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


def random_shape(dim, min_elem=1, max_elem=10):
    return tuple(random.randint(min_elem, max_elem) for _ in range(dim))


class TestAddWeighted:

    fcn = kornia.enhance.add_weighted

    def get_input(self, device, dtype, size, max_elem=10):
        shape = random_shape(size, max_elem)
        src1 = torch.randn(shape, device=device, dtype=dtype)
        src2 = torch.randn(shape, device=device, dtype=dtype)
        alpha = random.random()
        beta = random.random()
        gamma = random.random()
        return src1, src2, alpha, beta, gamma

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_smoke(self, device, dtype, size):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, size=3)
        assert_allclose(
            TestAddWeighted.fcn(src1, alpha, src2, beta, gamma),
            src1 * alpha + src2 * beta + gamma
        )

    def test_jit(self, device, dtype):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, size=3)
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_script = torch.jit.script(op)

        assert_allclose(op(*inputs), op_script(*inputs))

    @pytest.mark.parametrize("size", [2, 3])
    def test_gradcheck(self, size, device, dtype):
        src1, src2, alpha, beta, gamma = self.get_input(
            device, dtype, size=3, max_elem=5)  # to shave time on gradcheck
        src1 = utils.tensor_to_gradcheck_var(src1)  # to var
        src2 = utils.tensor_to_gradcheck_var(src2)  # to var
        assert gradcheck(kornia.enhance.AddWeighted(alpha, beta, gamma), (src1, src2),
                         raise_exception=True)

    def test_module(self, device, dtype):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, size=3)
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_module = kornia.enhance.AddWeighted(alpha, beta, gamma)

        assert_allclose(op(*inputs), op_module(src1, src2))
