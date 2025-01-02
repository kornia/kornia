# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import random

import pytest
import torch

import kornia
from kornia.core import Tensor

from testing.base import BaseTester


def random_shape(dim, min_elem=1, max_elem=10):
    return tuple(random.randint(min_elem, max_elem) for _ in range(dim))


class TestAddWeighted(BaseTester):
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
        self.assert_close(TestAddWeighted.fcn(src1, alpha, src2, beta, gamma), src1 * alpha + src2 * beta + gamma)

    @pytest.mark.parametrize("size1, size2", [((2, 5, 5), (4, 5, 5)), ((2, 5, 5), (2, 3, 5, 5))])
    def test_shape_mismatch(self, device, dtype, size1, size2):
        src1 = torch.randn(size1, device=device, dtype=dtype)
        src2 = torch.randn(size2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            TestAddWeighted.fcn(src1, 1.0, src2, 1.0, 0.0)

    @pytest.mark.parametrize("size1, size2", [((2, 3, 5, 5), (2, 3, 5, 5)), ((2, 3, 5, 5), (2, 3, 5, 5))])
    @pytest.mark.parametrize("alpha", [torch.randn(2, 3, 5, 5), 1.0])
    @pytest.mark.parametrize("beta", [torch.randn(2, 3, 5, 5), 1.0])
    @pytest.mark.parametrize("gamma", [torch.randn(2, 3, 5, 5), 1.0])
    def test_shape(self, device, dtype, size1, size2, alpha, beta, gamma):
        src1 = torch.randn(size1, device=device, dtype=dtype)
        src2 = torch.randn(size2, device=device, dtype=dtype)
        if isinstance(alpha, Tensor):
            alpha = alpha.to(src1)
        if isinstance(beta, Tensor):
            beta = beta.to(src2)
        if isinstance(gamma, Tensor):
            gamma = gamma.to(src1)
        self.assert_close(TestAddWeighted.fcn(src1, alpha, src2, beta, gamma), src1 * alpha + src2 * beta + gamma)

    def test_dynamo(self, device, dtype, torch_optimizer):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, size=3)
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_optimized = torch_optimizer(op)

        self.assert_close(op(*inputs), op_optimized(*inputs), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("size", [2, 3])
    def test_gradcheck(self, size, device):
        src1, src2, alpha, beta, gamma = self.get_input(
            device, torch.float64, size=3, max_elem=5
        )  # to shave time on gradcheck
        self.gradcheck(kornia.enhance.AddWeighted(alpha, beta, gamma), (src1, src2))

    def test_module(self, device, dtype):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, size=3)
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_module = kornia.enhance.AddWeighted(alpha, beta, gamma)

        self.assert_close(op(*inputs), op_module(src1, src2))
