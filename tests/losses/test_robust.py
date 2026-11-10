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


@pytest.mark.parametrize(
    "loss_fn,loss_module",
    [
        (kornia.losses.charbonnier_loss, kornia.losses.CharbonnierLoss),
        (kornia.losses.cauchy_loss, kornia.losses.CauchyLoss),
        (kornia.losses.geman_mcclure_loss, kornia.losses.GemanMcclureLoss),
    ],
)
class TestRobustLossPrecision(BaseTester):
    @pytest.mark.parametrize("reduction", ["none", None, "mean", "sum"])
    @pytest.mark.parametrize("large", [False, True], ids=["ordinary", "overflow_5601"])
    def test_values_and_gradients(self, device, dtype, loss_fn, loss_module, reduction, large):
        residuals = [180.0, 181.0, 255.0, 256.0, 1000.0] if large else [0.0, 0.125, 0.25, 1.0, 3.0]
        img1 = torch.tensor([residuals, [-r for r in residuals]], device=device, dtype=dtype, requires_grad=True)
        img2 = torch.zeros_like(img1, requires_grad=True)
        # Evaluate the defining functions and analytical derivatives independently on CPU in float64.
        r = img1.detach().cpu().double()
        q = r.square()
        if loss_fn is kornia.losses.charbonnier_loss:
            expected = q / ((q + 1).sqrt() + 1)
            expected_grad = r / (q + 1).sqrt()
        elif loss_fn is kornia.losses.cauchy_loss:
            expected = (1 + q / 2).log()
            expected_grad = 2 * r / (q + 2)
        else:
            expected = 2 * q / (q + 4)
            expected_grad = 16 * r / (q + 4).square()
        if reduction == "mean":
            expected = expected.mean()
            expected_grad = expected_grad / img1.numel()
        elif reduction == "sum":
            expected = expected.sum()
        expected = expected.to(device=device, dtype=dtype)
        expected_grad = expected_grad.to(device=device, dtype=dtype)

        actual = loss_fn(img1, img2, reduction)
        assert actual.dtype == dtype
        assert actual.device == img1.device
        assert torch.isfinite(actual).all()
        self.assert_close(actual, expected, rtol=4 * torch.finfo(dtype).eps, atol=0)
        self.assert_close(loss_module(reduction)(img1, img2), actual, rtol=0, atol=0)
        self.assert_close(loss_fn(img2, img1, reduction), actual, rtol=0, atol=0)
        with torch.autograd.detect_anomaly():
            grad1, grad2 = torch.autograd.grad(actual.sum(), (img1, img2))
        assert torch.isfinite(grad1).all()
        assert torch.isfinite(grad2).all()
        self.assert_close(grad2, -grad1, rtol=0, atol=0)
        # Geman-McClure's quotient backward subtracts nearly equal terms at large residuals.
        # Allow its existing float32 cancellation error, and one float16 subnormal rounding step.
        grad_rtol = 4 * torch.finfo(dtype).eps
        if large and loss_fn is kornia.losses.geman_mcclure_loss:
            grad_rtol = max(grad_rtol, 0.01 if dtype != torch.float64 else 1e-8)
        grad_atol = 2**-24 if dtype == torch.float16 else 0
        self.assert_close(grad1, expected_grad, rtol=grad_rtol, atol=grad_atol)
        representable = expected_grad != 0
        assert (grad1[representable].sign() == expected_grad[representable].sign()).all()
        # At r=1000 the Geman-McClure derivative is below half a float16 subnormal.
        self.assert_close(grad1[~representable], expected_grad[~representable], rtol=0, atol=0)

    def test_float16_limit(self, device, dtype, loss_fn, loss_module):
        if dtype != torch.float16:
            pytest.skip("the float16 overflow regression has a finite float32 square")
        limit = torch.finfo(dtype).max
        img1 = torch.tensor([limit, -limit], device=device, dtype=dtype, requires_grad=True)
        actual = loss_fn(img1, torch.zeros_like(img1))
        if loss_fn is kornia.losses.charbonnier_loss:
            value, derivative = limit - 1, 1.0
        elif loss_fn is kornia.losses.cauchy_loss:
            value = torch.tensor(1 + limit**2 / 2, dtype=torch.float64).log().item()
            derivative = 2 * limit / (limit**2 + 2)
        else:
            value, derivative = 2.0, 16 * limit / (limit**2 + 4) ** 2
        assert torch.isfinite(actual).all()
        self.assert_close(actual, img1.new_tensor([value, value]), rtol=4 * torch.finfo(dtype).eps, atol=0)
        with torch.autograd.detect_anomaly():
            gradient = torch.autograd.grad(actual.sum(), img1)[0]
        assert torch.isfinite(gradient).all()
        self.assert_close(gradient, img1.new_tensor([derivative, -derivative]), rtol=4 * torch.finfo(dtype).eps, atol=0)

    def test_half_precision_is_evaluated_in_float32(self, device, dtype, loss_fn, loss_module):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("only half-precision inputs are evaluated in float32")
        # Half-precision inputs are evaluated in float32 and rounded once. bfloat16 does not overflow at these
        # residuals, but evaluated in bfloat16 instead, 6 to 19 percent of these losses round differently.
        residual = torch.linspace(-40.0, 40.0, 641, device=device, dtype=dtype)
        img1 = residual.clone().requires_grad_(True)
        actual = loss_fn(img1, torch.zeros_like(img1))
        (grad,) = torch.autograd.grad(actual.sum(), img1)
        img1_float = residual.float().requires_grad_(True)
        expected = loss_fn(img1_float, torch.zeros_like(img1_float))
        (expected_grad,) = torch.autograd.grad(expected.sum(), img1_float)
        self.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
        self.assert_close(grad, expected_grad.to(dtype), rtol=0, atol=0)
