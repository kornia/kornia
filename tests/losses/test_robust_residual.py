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


class TestRobustResidual(BaseTester):
    @pytest.mark.parametrize("loss_name", ["cauchy_loss", "charbonnier_loss", "geman_mcclure_loss"])
    def test_opposite_large_inputs(self, device, dtype, loss_name):
        pred = torch.tensor([40000.0, -40000.0], device=device, dtype=dtype, requires_grad=True)
        target = (-pred.detach()).requires_grad_()
        delta = pred.detach().cpu().double() - target.detach().cpu().double()
        if loss_name == "cauchy_loss":
            expected = torch.log1p(delta.square() / 2)
            expected_grad = delta / (1 + delta.square() / 2)
        elif loss_name == "charbonnier_loss":
            norm = torch.hypot(delta, torch.ones_like(delta))
            expected = norm - 1
            expected_grad = delta / norm
        else:
            expected = 2 / (1 + 4 / delta.square())
            expected_grad = 16 * delta / (delta.square() + 4).square()

        actual = getattr(kornia.losses, loss_name)(pred, target)
        self.assert_close(actual, expected.to(pred))
        assert actual.dtype == dtype
        actual.sum().backward()
        assert torch.isfinite(pred.grad).all()
        assert torch.isfinite(target.grad).all()
        self.assert_close(pred.grad, expected_grad.to(pred))
        self.assert_close(target.grad, -expected_grad.to(pred))

    @pytest.mark.parametrize("loss_name", ["cauchy_loss", "charbonnier_loss", "geman_mcclure_loss"])
    def test_integer_inputs_keep_a_floating_loss(self, device, loss_name):
        # Only half-precision losses are cast back to the input dtype; an integer residual gives a float loss.
        delta = torch.tensor([1.0, 3.0], dtype=torch.float64)
        expected = {
            "cauchy_loss": torch.log1p(delta.square() / 2),
            "charbonnier_loss": torch.hypot(delta, torch.ones_like(delta)) - 1,
            "geman_mcclure_loss": 2 / (1 + 4 / delta.square()),
        }[loss_name]
        loss = getattr(kornia.losses, loss_name)(
            torch.tensor([1, 3], device=device), torch.tensor([0, 0], device=device)
        )
        assert loss.dtype.is_floating_point
        self.assert_close(loss.cpu().double(), expected, rtol=1e-6, atol=1e-6)
