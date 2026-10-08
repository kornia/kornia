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


class TestCharbonnierLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none", None])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_smoke(self, device, dtype, reduction, shape):
        img1 = torch.rand(shape, device=device, dtype=dtype)
        img2 = torch.rand(shape, device=device, dtype=dtype)

        assert kornia.losses.charbonnier_loss(img1, img2, reduction) is not None

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        from kornia.core.exceptions import BaseError

        with pytest.raises(BaseError) as execinfo:
            kornia.losses.charbonnier_loss(img, img, reduction="test")
        assert "Given type of reduction is not supported. Got: test" in str(execinfo.value)

        # Check if both are tensors
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.charbonnier_loss(1.0, img)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.charbonnier_loss(img, 1.0)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        # Check if same shape
        from kornia.core.exceptions import ShapeError

        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.losses.charbonnier_loss(img, img_b)
        assert "Shape mismatch" in str(errinfo.value)

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img, img, reduction="none")
        assert actual.shape == shape

        actual = kornia.losses.charbonnier_loss(img, img, reduction="sum")
        assert actual.shape == ()

        actual = kornia.losses.charbonnier_loss(img, img, reduction="mean")
        assert actual.shape == ()

    def test_gradcheck(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)

        self.gradcheck(kornia.losses.charbonnier_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_module = kornia.losses.CharbonnierLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none", None])
    def test_small_residual(self, device, dtype, reduction):
        # These powers of two are exact in the input dtype. The reference uses the original formula:
        # with decimal.localcontext() as ctx:
        #     ctx.prec = 80
        #     expected = (Decimal(1) + Decimal(residual) ** 2).sqrt() - Decimal(1)
        residual, expected_value = {
            torch.float16: (2**-6, 0.000122062862828759023779),
            torch.bfloat16: (2**-6, 0.000122062862828759023779),
            torch.float32: (2**-16, 1.16415321820158550875879e-10),
            torch.float64: (2**-30, 4.33680868994201773508942e-19),
        }[dtype]
        img1 = torch.tensor([-residual, 0.0, residual], device=device, dtype=dtype)
        img2 = torch.zeros_like(img1)
        expected = torch.tensor([expected_value, 0.0, expected_value], device=device, dtype=dtype)
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()

        # An absolute tolerance would hide the failure: the old implementation returned all zeros.
        self.assert_close(
            kornia.losses.charbonnier_loss(img1, img2, reduction),
            expected,
            rtol=4 * torch.finfo(dtype).eps,
            atol=0,
        )
        self.assert_close(
            kornia.losses.CharbonnierLoss(reduction)(img1, img2),
            expected,
            rtol=4 * torch.finfo(dtype).eps,
            atol=0,
        )

    def test_moderate_residual(self, device, dtype):
        # Below |r| = 1 the original formula loses digits long before it returns 0: at r = 0.0625 it is off by about
        # 16 eps (relative) in float32, 168 in float64 and 128 in bfloat16, and at r = 0.25 by 16 in float16. Residuals
        # up to 0.4375 also catch the threshold of the rationalized form moved from r**2 = 1 to below 0.19. These
        # residuals and their squares are exact in every dtype; the references are generated as in test_small_residual.
        residual = torch.tensor([0.4375, 0.375, 0.3125, 0.25, 0.125, 0.09375, 0.0625], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                9.151557478581129039506523e-2,
                6.800046816469139598395604e-2,
                4.769091339001313302785597e-2,
                3.077640640441513745535246e-2,
                7.782218537318706545826654e-3,
                4.384917499262331490555988e-3,
                1.951221367587335304459672e-3,
            ],
            device=device,
            dtype=dtype,
        )
        actual = kornia.losses.charbonnier_loss(residual, torch.zeros_like(residual))
        self.assert_close(actual, expected, rtol=4 * torch.finfo(dtype).eps, atol=0)

    def test_zero_gradient(self, device, dtype):
        img1 = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        img2 = torch.zeros_like(img1, requires_grad=True)
        loss = kornia.losses.charbonnier_loss(img1, img2, reduction="sum")
        grad1, grad2 = torch.autograd.grad(loss, (img1, img2))
        self.assert_close(grad1, torch.zeros_like(img1))
        self.assert_close(grad2, torch.zeros_like(img2))

    def test_large_residual(self, device, dtype):
        # Squaring this finite residual overflows the dtype the loss is computed in; float16 residuals are squared in
        # float32 and do not overflow. The unused rationalized branch must neither change the original loss nor
        # contaminate the gradient with inf / inf.
        overflow_residual = 2.0 * torch.finfo(dtype).max ** 0.5
        img1 = torch.tensor([1.0, 3.0, overflow_residual], device=device, dtype=dtype, requires_grad=True)
        img2 = torch.zeros_like(img1)
        actual = kornia.losses.charbonnier_loss(img1, img2)
        compute = img1.float() if dtype in (torch.float16, torch.bfloat16) else img1
        expected = ((compute.square() + 1.0).sqrt() - 1.0).to(dtype)
        self.assert_close(actual, expected, rtol=4 * torch.finfo(dtype).eps, atol=0)
        # detect_anomaly also rejects a nan inside the backward that the outer torch.where would discard.
        with torch.autograd.detect_anomaly():
            grad_actual = torch.autograd.grad(actual.sum(), img1, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected.sum(), img1)[0]
        self.assert_close(grad_actual, grad_expected)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.charbonnier_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img1, img2, reduction=reduction)

        if reduction == "mean":
            expected = torch.tensor(0.41421356237, device=device, dtype=dtype)
        elif reduction == "sum":
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.41421356237).sum()

        self.assert_close(actual, expected)


class TestConventionsRobustLosses(BaseTester):
    @pytest.mark.parametrize(
        "loss_fn, module, expected",
        [
            pytest.param(
                kornia.losses.charbonnier_loss,
                kornia.losses.CharbonnierLoss,
                [0.0, 0.1180339887498949, 1.2360679774997898],
                id="charbonnier-alpha_1",
            ),
            pytest.param(
                kornia.losses.cauchy_loss,
                kornia.losses.CauchyLoss,
                [0.0, 0.11778303565638346, 1.0986122886681098],
                id="cauchy-alpha_0",
            ),
            pytest.param(
                kornia.losses.geman_mcclure_loss,
                kornia.losses.GemanMcclureLoss,
                [0.0, 0.11764705882352944, 1.0],
                id="geman_mcclure-alpha_-2",
            ),
            pytest.param(
                kornia.losses.welsch_loss,
                kornia.losses.WelschLoss,
                [0.0, 0.1175030974154046, 0.8646647167633873],
                id="welsch-alpha_-inf",
            ),
        ],
    )
    def test_convention_robust_loss_is_barron_loss_at_unit_scale(self, loss_fn, module, expected, device, dtype):
        # Each loss is Barron's general loss rho(img1 - img2, alpha, c) at the fixed scale c = 1: Charbonnier alpha = 1,
        # Cauchy alpha = 0, Geman-McClure alpha = -2, Welsch alpha = -inf. It is even in the residual, so symmetric in
        # (img1, img2), and the default reduction 'none' keeps the input shape.
        # Snippet used to generate expected (robust_loss_pytorch 0.0.2, git jonbarron/robust_loss_pytorch@0c25c59):
        #   general.lossfun(torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64), torch.tensor(alpha), torch.tensor(1.0))
        img2 = torch.tensor([[0.25, -1.0, 1.5], [0.75, 0.125, -0.5]], device=device, dtype=dtype)
        residual = torch.tensor([[0.0, 0.5, -2.0], [-0.5, 2.0, 0.0]], device=device, dtype=dtype)
        img1 = img2 + residual
        values = torch.tensor(expected, device=device, dtype=dtype)
        expected_map = torch.stack([values, values[[1, 2, 0]]])
        out = loss_fn(img1, img2)
        assert out.shape == (2, 3)
        self.assert_close(out, expected_map)
        self.assert_close(loss_fn(img2, img1), expected_map)
        self.assert_close(module()(img1, img2), expected_map)
