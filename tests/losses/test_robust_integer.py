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

from kornia import losses

from testing.base import BaseTester


@pytest.mark.parametrize(
    "loss_fn,loss_module",
    [
        (losses.charbonnier_loss, losses.CharbonnierLoss),
        (losses.cauchy_loss, losses.CauchyLoss),
        (losses.geman_mcclure_loss, losses.GemanMcclureLoss),
        (losses.welsch_loss, losses.WelschLoss),
    ],
)
class TestRobustIntegerInputs(BaseTester):
    @pytest.mark.parametrize(
        "input_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]
    )
    @pytest.mark.parametrize("reduction", ["none", None, "sum", "mean"])
    def test_integer_values(self, device, input_dtype, reduction, loss_fn, loss_module):
        if reduction is None and loss_fn is losses.welsch_loss:
            pytest.skip("Welsch accepts only string reductions")
        values = {
            torch.uint8: ([0, 16, 255, 32], [255, 0, 0, 32]),
            torch.int8: ([0, 16, 127, -128], [127, -128, 0, 16]),
            torch.int16: ([0, 256, 32767, -32768], [32767, -32768, 0, 256]),
            torch.int32: ([0, 65536, -65536, 16], [65536, 0, 65536, 16]),
            torch.int64: ([0, 2**32, -(2**32), 16], [2**32, 0, 2**32, 16]),
            torch.bool: ([False, True, False, True], [True, False, False, True]),
        }
        left, right = values[input_dtype]
        img1 = torch.tensor(left, device=device, dtype=input_dtype)
        img2 = torch.tensor(right, device=device, dtype=input_dtype)
        # Evaluate the defining penalties independently on converted pixel values.
        residuals = [x - y for x, y in zip(img1.float().cpu().tolist(), img2.float().cpu().tolist())]
        if loss_fn is losses.charbonnier_loss:
            expected_values = [math.sqrt(1 + r * r) - 1 for r in residuals]
        elif loss_fn is losses.cauchy_loss:
            expected_values = [math.log1p(r * r / 2) for r in residuals]
        elif loss_fn is losses.geman_mcclure_loss:
            expected_values = [2 * r * r / (r * r + 4) for r in residuals]
        else:
            expected_values = [-math.expm1(-r * r / 2) for r in residuals]
        expected = torch.tensor(expected_values, device=device, dtype=torch.float32)
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()

        actual = loss_fn(img1, img2, reduction)
        assert actual.dtype == torch.float32
        assert actual.device == device
        self.assert_close(actual, expected, rtol=1e-6, atol=0)
        self.assert_close(loss_fn(img2, img1, reduction), actual, rtol=0, atol=0)
        self.assert_close(loss_module(reduction)(img1, img2), actual, rtol=0, atol=0)

    @pytest.mark.parametrize("integer_first", [True, False])
    def test_mixed_float_values_and_gradients(self, device, dtype, integer_first, loss_fn, loss_module):
        pixels = torch.tensor([0, 16, 255], device=device, dtype=torch.uint8)
        image = torch.tensor([0.25, 15.5, 253.0], device=device, dtype=dtype, requires_grad=True)
        result_dtype = torch.promote_types(dtype, torch.float32)
        reference = image.detach().to(result_dtype).requires_grad_(True)
        expected = loss_fn(pixels.float(), reference)
        expected_grad = torch.autograd.grad(expected.sum(), reference)[0].to(dtype)
        actual = loss_fn(pixels, image) if integer_first else loss_fn(image, pixels)
        actual_grad = torch.autograd.grad(actual.sum(), image)[0]
        assert actual.dtype == result_dtype
        self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_traced_integer_input(self, device, loss_fn, loss_module):
        img1 = torch.tensor([0, 16, 255], device=device, dtype=torch.uint8)
        img2 = torch.zeros_like(img1)
        traced = torch.jit.trace(loss_module(), (img1, img2))
        expected = loss_fn(img1.float(), img2.float())
        self.assert_close(traced(img1, img2), expected, rtol=0, atol=0)
