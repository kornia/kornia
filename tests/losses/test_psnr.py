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

import math

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

    @pytest.mark.parametrize("scalar_range", [True, False])
    def test_scripted_tensor_range(self, device, scalar_range):
        image = torch.tensor([32.0], device=device)
        target = torch.zeros_like(image)
        values = 512.0 if scalar_range else [512.0, 1024.0]
        max_val = torch.tensor(values, device=device, requires_grad=True)
        expected = 20 * torch.log10(max_val.detach() / 32)
        metric = torch.jit.script(kornia.metrics.psnr)
        loss = torch.jit.script(kornia.losses.psnr_loss)
        module = torch.jit.script(kornia.losses.PSNRLoss(max_val))
        actuals = [(metric(image, target, max_val), 1), (loss(image, target, max_val), -1), (module(image, target), -1)]
        for actual, sign in actuals:
            assert actual.shape == max_val.shape
            self.assert_close(actual, sign * expected)
            (range_grad,) = torch.autograd.grad(actual.sum(), max_val)
            self.assert_close(range_grad, sign * 20 / (math.log(10) * max_val.detach()))

    @pytest.mark.parametrize("max_val", [512, 512.0])
    def test_scripted_numeric_range(self, device, max_val):
        image = torch.tensor([32.0], device=device)
        target = torch.zeros_like(image)
        expected = kornia.metrics.psnr(image, target, max_val)
        metric = torch.jit.script(kornia.metrics.psnr)
        loss = torch.jit.script(kornia.losses.psnr_loss)
        module = torch.jit.script(kornia.losses.PSNRLoss(max_val))
        self.assert_close(metric(image, target, max_val), expected)
        self.assert_close(loss(image, target, max_val), -expected)
        self.assert_close(module(image, target), -expected)

    def test_integer_images(self, device):
        # psnr_loss and PSNRLoss inherit the float32 computation of integer images from psnr (#5536).
        generator = torch.Generator().manual_seed(0)
        pred = (torch.rand(2, 3, 3, 2, generator=generator) * 255).to(torch.uint8).to(device)
        target = (torch.rand(2, 3, 3, 2, generator=generator) * 255).to(torch.uint8).to(device)
        expected = kornia.losses.psnr_loss(pred.float(), target.float(), 255.0)
        self.assert_close(kornia.losses.psnr_loss(pred, target, 255.0), expected, rtol=0, atol=0)
        self.assert_close(kornia.losses.PSNRLoss(255.0)(pred, target), expected, rtol=0, atol=0)

    def test_gradcheck(self, device, dtype):
        dtype = torch.float64
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        self.gradcheck(kornia.losses.psnr_loss, (pred, target, 1.0))


class TestConventionsPSNRLoss(BaseTester):
    def test_convention_psnr_loss_is_minus_psnr(self, device, dtype):
        # psnr_loss is -psnr on the MSE pooled over the batch, PSNRLoss is the same function, and identical images give
        # -inf: here -10 log10(2 / (0.125**2 + 0.375**2)) = -11.0721
        a = torch.zeros(2, 1, 4, 6, device=device, dtype=dtype)
        b = a.clone()
        b[0] += 0.125
        b[1] += 0.375
        expected = torch.tensor(-10.0 * math.log10(2.0 / (0.125**2 + 0.375**2)), device=device, dtype=dtype)
        self.assert_close(kornia.losses.psnr_loss(a, b, 1.0), expected)
        self.assert_close(kornia.losses.PSNRLoss(1.0)(a, b), expected)
        assert kornia.losses.psnr_loss(a, a, 1.0).item() == -math.inf
