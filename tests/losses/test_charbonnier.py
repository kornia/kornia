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
