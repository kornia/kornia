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


class TestWelschLoss(BaseTester):
    def test_smoke(self, device, dtype):
        img1 = torch.rand(2, 3, 2, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 2, device=device, dtype=dtype)

        criterion = kornia.losses.WelschLoss()

        assert criterion(img1, img2) is not None

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.WelschLoss(reduction="none")(img, img)
        assert actual.shape == shape

        actual = kornia.losses.WelschLoss(reduction="mean")(img, img)
        assert actual.shape == ()

    def test_gradcheck(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)

        self.gradcheck(kornia.losses.welsch_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.welsch_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.welsch_loss
        op_module = kornia.losses.WelschLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_small_residual(self, device, dtype, reduction):
        # Exact powers of two from #5600. Reference from the original formula:
        # with decimal.localcontext() as ctx:
        #     ctx.prec = 80
        #     expected = Decimal(1) - (-Decimal(residual) ** 2 / 2).exp()
        residual, expected_value = {
            torch.float16: (2**-6, 0.0001220628622225587251301834),
            torch.bfloat16: (2**-4, 0.001951218892524527289957341),
            torch.float32: (2**-12, 2.980232194360610706156728e-8),
            torch.float64: (2**-30, 4.336808689942017735089416e-19),
        }[dtype]
        img1 = torch.tensor([-residual, 0.0, residual], device=device, dtype=dtype, requires_grad=True)
        img2 = torch.zeros_like(img1, requires_grad=True)
        expected = torch.tensor([expected_value, 0.0, expected_value], device=device, dtype=dtype)
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()

        actual = kornia.losses.welsch_loss(img1, img2, reduction)
        assert actual.dtype == dtype
        assert actual.device == device
        assert (actual > 0).any()
        # Absolute tolerance would hide the original all-zero output.
        self.assert_close(actual, expected, rtol=4 * torch.finfo(dtype).eps, atol=0)
        self.assert_close(
            kornia.losses.WelschLoss(reduction)(img1, img2), expected, rtol=4 * torch.finfo(dtype).eps, atol=0
        )
        self.assert_close(actual, kornia.losses.welsch_loss(img2, img1, reduction), rtol=0, atol=0)

        grad1, grad2 = torch.autograd.grad(actual.sum(), (img1, img2))
        expected_grad = img1.detach() * (-0.5 * img1.detach().square()).exp()
        if reduction == "mean":
            expected_grad = expected_grad / img1.numel()
        assert torch.isfinite(grad1).all()
        assert torch.isfinite(grad2).all()
        assert grad1[0] < 0
        assert grad1[2] > 0
        self.assert_close(grad1, expected_grad, rtol=4 * torch.finfo(dtype).eps, atol=0)
        self.assert_close(grad2, -expected_grad, rtol=4 * torch.finfo(dtype).eps, atol=0)

    def test_residual_values_and_gradients(self, device, dtype):
        # Values generated with the Decimal snippet in test_small_residual.
        img1 = torch.tensor([0.0, 0.125, 0.5, 1.0, 3.0, 6.0, 10.0], device=device, dtype=dtype, requires_grad=True)
        img2 = torch.zeros_like(img1, requires_grad=True)
        expected = torch.tensor(
            [
                0.0,
                0.00778206173975648789406,
                0.117503097415404597135,
                0.393469340287366576396,
                0.988891003461757693504,
                0.999999984770020255287,
                1.0,
            ],
            device=device,
            dtype=dtype,
        )
        actual = kornia.losses.welsch_loss(img1, img2)
        # MPS float32 has about 2.52e-6 relative error at residual 0.125 (PR #5604 CI).
        rtol = 3e-6 if device.type == "mps" and dtype == torch.float32 else 4 * torch.finfo(dtype).eps
        self.assert_close(actual, expected, rtol=rtol, atol=0)
        self.assert_close(actual, kornia.losses.welsch_loss(img2, img1), rtol=0, atol=0)
        grad1, grad2 = torch.autograd.grad(actual.sum(), (img1, img2))
        expected_grad = img1.detach() * (-0.5 * img1.detach().square()).exp()
        assert torch.isfinite(grad1).all()
        assert torch.isfinite(grad2).all()
        self.assert_close(grad1, expected_grad, rtol=4 * torch.finfo(dtype).eps, atol=0)
        self.assert_close(grad2, -expected_grad, rtol=4 * torch.finfo(dtype).eps, atol=0)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.welsch_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.welsch_loss(img1, img2, reduction=reduction)

        if reduction == "mean":
            expected = torch.tensor(0.39346934028, device=device, dtype=dtype)
        elif reduction == "sum":
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.39346934028).sum()

        self.assert_close(actual, expected)

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        from kornia.core.exceptions import BaseError

        with pytest.raises(BaseError) as execinfo:
            kornia.losses.welsch_loss(img, img, reduction="test")
        assert "Given type of reduction is not supported. Got: test" in str(execinfo.value)

        # Check if both are tensors
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.welsch_loss(1.0, img)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.welsch_loss(img, 1.0)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        # Check if same shape
        from kornia.core.exceptions import ShapeError

        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.losses.welsch_loss(img, img_b)
        assert "Shape mismatch" in str(errinfo.value)
