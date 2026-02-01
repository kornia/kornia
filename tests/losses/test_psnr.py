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


class TestPSNRLoss(BaseTester):
    def test_smoke(self, device, dtype):
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(pred, target)

        assert loss is not None

    def test_type(self, device, dtype):
        # Expecting an exception
        # since we pass integers instead of torch tensors
        criterion = kornia.losses.PSNRLoss(1.0).to(device, dtype)
        with pytest.raises(Exception):
            criterion(1, 2)

    def test_shape(self, device, dtype):
        # Expecting an exception
        # since we pass tensors of different shapes
        criterion = kornia.losses.PSNRLoss(1.0).to(device, dtype)
        with pytest.raises(Exception):
            criterion(torch.rand(2, 3, 3, 2), torch.rand(2, 3, 3))

    def test_loss(self, device, dtype):
        pred = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(-20.0, device=device, dtype=dtype)
        actual = kornia.losses.psnr_loss(pred, 1.2 * pred, 2.0)
        self.assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (pred, target, 1.0)

        op = kornia.losses.psnr_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(*args), op_optimized(*args))

    def test_module(self, device, dtype):
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (pred, target, 1.0)

        op = kornia.losses.psnr_loss
        op_module = kornia.losses.PSNRLoss(1.0)

        self.assert_close(op(*args), op_module(pred, target))

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        dtype = torch.float64
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        self.gradcheck(kornia.losses.psnr_loss, (pred, target, 1.0))
