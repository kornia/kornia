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

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestAddWeighted(BaseTester):
    fcn = kornia.enhance.add_weighted

    def get_input(self, device, dtype, shape):
        src1 = torch.randn(shape, device=device, dtype=dtype)
        src2 = torch.randn(shape, device=device, dtype=dtype)
        alpha, beta, gamma = 0.3, 0.7, 0.2
        return src1, src2, alpha, beta, gamma

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 1, 3, 4, 5)])
    def test_smoke(self, device, dtype, shape):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, shape)
        expected = src1 * alpha + src2 * beta + gamma
        self.assert_close(TestAddWeighted.fcn(src1, alpha, src2, beta, gamma), expected)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_python_scalars_keep_opmath_precision(self, dtype):
        src1 = torch.linspace(-2.0, 2.0, 1000, dtype=dtype)
        src2 = torch.linspace(2.0, -2.0, 1000, dtype=dtype)

        actual = TestAddWeighted.fcn(src1, 0.1, src2, 0.3, 0.2)
        expected = src1 * 0.1 + src2 * 0.3 + 0.2
        downcast = (
            src1 * torch.tensor(0.1, dtype=dtype)
            + src2 * torch.tensor(0.3, dtype=dtype)
            + torch.tensor(0.2, dtype=dtype)
        )

        assert torch.equal(actual, expected)
        assert not torch.equal(expected, downcast)

    @pytest.mark.device_agnostic
    def test_python_scalars_promote_integer_inputs(self):
        src1 = torch.full((2, 2), 2, dtype=torch.uint8)
        src2 = torch.full((2, 2), 4, dtype=torch.uint8)

        actual = TestAddWeighted.fcn(src1, 0.5, src2, 0.5, 0.0)

        assert actual.dtype == torch.float32
        assert torch.equal(actual, torch.full((2, 2), 3.0))

    @pytest.mark.parametrize("size1, size2", [((2, 5, 5), (4, 5, 5)), ((2, 5, 5), (2, 3, 5, 5))])
    def test_shape_mismatch(self, device, dtype, size1, size2):
        src1 = torch.randn(size1, device=device, dtype=dtype)
        src2 = torch.randn(size2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            TestAddWeighted.fcn(src1, 1.0, src2, 1.0, 0.0)

    @pytest.mark.parametrize("size1, size2", [((2, 3, 5, 5), (2, 3, 5, 5))])
    @pytest.mark.parametrize("alpha", [torch.randn(2, 3, 5, 5), 1.0])
    @pytest.mark.parametrize("beta", [torch.randn(2, 3, 5, 5), 1.0])
    @pytest.mark.parametrize("gamma", [torch.randn(2, 3, 5, 5), 1.0])
    def test_shape(self, device, dtype, size1, size2, alpha, beta, gamma):
        src1 = torch.randn(size1, device=device, dtype=dtype)
        src2 = torch.randn(size2, device=device, dtype=dtype)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(src1)
        if isinstance(beta, torch.Tensor):
            beta = beta.to(src2)
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.to(src1)
        self.assert_close(TestAddWeighted.fcn(src1, alpha, src2, beta, gamma), src1 * alpha + src2 * beta + gamma)

    @pytest.mark.parametrize("coef_shape", [(), (1,), (5,), (2, 1, 1, 1), (1, 3, 1, 1), (3, 5, 5)])
    def test_broadcastable_tensor_coefficients(self, device, dtype, coef_shape):
        src1 = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        src2 = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        alpha = torch.rand(coef_shape, device=device, dtype=dtype)
        beta = torch.rand(coef_shape, device=device, dtype=dtype)
        gamma = torch.rand(coef_shape, device=device, dtype=dtype)

        actual = TestAddWeighted.fcn(src1, alpha, src2, beta, gamma)

        assert actual.shape == src1.shape
        self.assert_close(actual, src1 * alpha + src2 * beta + gamma)
        self.assert_close(kornia.enhance.AddWeighted(alpha, beta, gamma)(src1, src2), actual)

    @pytest.mark.parametrize("name", ["alpha", "beta", "gamma"])
    @pytest.mark.parametrize("coef_shape", [(4,), (3, 1, 1, 1), (1, 2, 3, 5, 5)])
    def test_non_broadcastable_tensor_coefficient(self, device, dtype, name, coef_shape):
        src1 = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        src2 = torch.randn(2, 3, 5, 5, device=device, dtype=dtype)
        coefs = {"alpha": 0.5, "beta": 0.5, "gamma": 0.0}
        coefs[name] = torch.rand(coef_shape, device=device, dtype=dtype)
        with pytest.raises(Exception, match=name):
            TestAddWeighted.fcn(src1, coefs["alpha"], src2, coefs["beta"], coefs["gamma"])

    def test_dynamo(self, device, dtype, torch_optimizer):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, (2, 3, 4))
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_optimized = torch_optimizer(op)

        self.assert_close(op(*inputs), op_optimized(*inputs), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_gradcheck(self, shape, device):
        src1, src2, alpha, beta, gamma = self.get_input(device, torch.float64, shape)
        self.gradcheck(kornia.enhance.AddWeighted(alpha, beta, gamma), (src1, src2))

    def test_module(self, device, dtype):
        src1, src2, alpha, beta, gamma = self.get_input(device, dtype, (2, 3, 4))
        inputs = (src1, alpha, src2, beta, gamma)

        op = TestAddWeighted.fcn
        op_module = kornia.enhance.AddWeighted(alpha, beta, gamma)

        self.assert_close(op(*inputs), op_module(src1, src2))
