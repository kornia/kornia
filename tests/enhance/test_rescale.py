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

from kornia.enhance.rescale import Rescale

from testing.base import BaseTester


class TestRescale(BaseTester):
    def test_smoke(self, device, dtype):
        r = Rescale(0.5).to(device, dtype)
        x = torch.ones(1, 3, 4, 4, device=device, dtype=dtype)
        assert isinstance(r(x), torch.Tensor)

    def test_cardinality(self, device, dtype):
        r = Rescale(2.0).to(device, dtype)
        x = torch.ones(2, 3, 5, 5, device=device, dtype=dtype)
        assert r(x).shape == x.shape

    def test_float_factor(self, device, dtype):
        r = Rescale(0.5).to(device, dtype)
        x = torch.ones(1, 1, 2, 2, device=device, dtype=dtype) * 4.0
        expected = torch.ones(1, 1, 2, 2, device=device, dtype=dtype) * 2.0
        self.assert_close(r(x), expected)

    def test_tensor_factor(self, device, dtype):
        factor = torch.tensor(3.0, device=device, dtype=dtype)
        r = Rescale(factor)
        x = torch.ones(1, 1, 2, 2, device=device, dtype=dtype) * 2.0
        expected = torch.ones(1, 1, 2, 2, device=device, dtype=dtype) * 6.0
        self.assert_close(r(x), expected)

    def test_zero_factor(self, device, dtype):
        r = Rescale(0.0).to(device, dtype)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        expected = torch.zeros_like(x)
        self.assert_close(r(x), expected)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            Rescale(torch.ones(2))  # not a 0-d tensor

        with pytest.raises(TypeError):
            Rescale([0.5])  # list is not valid either

    def test_gradcheck(self, device):
        factor = torch.tensor(2.0)
        r = Rescale(factor)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(r, (x,))

    def test_module(self, device, dtype):
        r1 = Rescale(2.0).to(device, dtype)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        self.assert_close(r1(x), x * 2.0)

    def test_dynamo(self, device, dtype, torch_optimizer):
        r = Rescale(0.5).to(device, dtype)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        op = torch_optimizer(r)
        self.assert_close(op(x), r(x))
