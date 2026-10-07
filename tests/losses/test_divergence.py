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
from kornia.core.exceptions import BaseError, ShapeError

from testing.base import BaseTester


class TestDivergenceLoss(BaseTester):
    @pytest.mark.parametrize(
        "pred,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
        ],
    )
    def test_js_div_loss_2d(self, device, dtype, pred, target, expected):
        actual = kornia.losses.js_div_loss_2d(pred.to(device, dtype), target.to(device, dtype))
        expected = torch.tensor(expected).to(device, dtype)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "pred,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_kl_div_loss_2d(self, device, dtype, pred, target, expected):
        actual = kornia.losses.kl_div_loss_2d(pred.to(device, dtype), target.to(device, dtype))
        expected = torch.tensor(expected).to(device, dtype)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "pred,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), torch.full((1, 7), 0.0)),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), math.inf)),
        ],
    )
    def test_kl_div_loss_2d_without_reduction(self, device, dtype, pred, target, expected):
        actual = kornia.losses.kl_div_loss_2d(pred.to(device, dtype), target.to(device, dtype), reduction="none")
        self.assert_close(actual, expected.to(device, dtype))

    @pytest.mark.parametrize(
        "pred,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_noncontiguous_kl(self, device, dtype, pred, target, expected):
        pred = pred.to(device, dtype).view(pred.shape[::-1]).transpose(-2, -1)
        target = target.to(device, dtype).view(target.shape[::-1]).transpose(-2, -1)
        actual = kornia.losses.kl_div_loss_2d(pred, target)
        expected = torch.tensor(expected).to(device, dtype)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "pred,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.303251),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.303251),
        ],
    )
    def test_noncontiguous_js(self, device, dtype, pred, target, expected):
        pred = pred.to(device, dtype).view(pred.shape[::-1]).transpose(-2, -1)
        target = target.to(device, dtype).view(target.shape[::-1]).transpose(-2, -1)
        actual = kornia.losses.js_div_loss_2d(pred, target)
        expected = torch.tensor(expected).to(device, dtype)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    def test_reduction_sum(self, device, dtype, loss):
        pred = torch.full((2, 3, 2, 4), 0.125, device=device, dtype=dtype)
        target = torch.zeros((2, 3, 2, 4), device=device, dtype=dtype)
        target[..., 0, 0] = 1.0
        unreduced = loss(pred, target, reduction="none")
        assert unreduced.shape == (2, 3)
        self.assert_close(loss(pred, target, reduction="sum"), unreduced.sum())
        self.assert_close(loss(pred, target, reduction="mean"), unreduced.mean())

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    @pytest.mark.parametrize("reduction", ["batchmean", "MEAN", "avg", None])
    def test_exception_invalid_reduction(self, device, dtype, loss, reduction):
        # An unknown reduction raises as in the sibling losses instead of returning the sum (#5535).
        pred = torch.full((1, 1, 2, 4), 0.125, device=device, dtype=dtype)
        with pytest.raises(NotImplementedError, match="Invalid reduction mode"):
            loss(pred, pred, reduction=reduction)

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    def test_exception_shape(self, device, dtype, loss):
        # pred and target must be 4-D (B, N, H, W) of the same shape; a transposed pred was reinterpreted in the
        # layout of target (#5535).
        target = torch.full((2, 3, 4, 6), 1 / 24, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="pred and target shapes must be the same"):
            loss(target.transpose(-2, -1).contiguous(), target)
        with pytest.raises(BaseError, match="pred and target shapes must be the same"):
            loss(target[:, :2], target)
        with pytest.raises(ShapeError):
            loss(target[0], target[0])

    def test_gradcheck_kl(self, device, dtype):
        dtype = torch.float64
        pred = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        self.gradcheck(kornia.losses.kl_div_loss_2d, (pred, target))

    def test_gradcheck_js(self, device, dtype):
        dtype = torch.float64
        pred = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        self.gradcheck(kornia.losses.js_div_loss_2d, (pred, target))

    def test_dynamo_kl(self, device, dtype, torch_optimizer):
        pred = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        target = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        args = (pred, target)
        op = kornia.losses.kl_div_loss_2d
        op_optimized = torch_optimizer(op)
        self.assert_close(op(*args), op_optimized(*args), rtol=0, atol=1e-5)

    def test_dynamo_js(self, device, dtype, torch_optimizer):
        pred = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        target = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        args = (pred, target)
        op = kornia.losses.js_div_loss_2d
        op_optimized = torch_optimizer(op)
        self.assert_close(op(*args), op_optimized(*args), rtol=0, atol=1e-5)

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    def test_identical_distributions_with_zero_cells(self, device, dtype, loss):
        # 0 * log 0 = 0: a distribution compared with itself has zero divergence and finite gradients even where it
        # has empty cells (#5554).
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        pred, target = p.clone().requires_grad_(), p.clone().requires_grad_()
        actual = loss(pred, target)
        # xlogy(p, p) and p * log(p) can differ by an ulp on some CPUs, so the value is zero up to the default
        # tolerance.
        self.assert_close(actual, torch.zeros_like(actual))
        actual.backward()
        assert torch.isfinite(pred.grad).all() and torch.isfinite(target.grad).all()
        assert torch.equal(pred.grad[p == 0], torch.zeros_like(pred.grad[p == 0]))
        assert torch.equal(target.grad[p == 0], torch.zeros_like(target.grad[p == 0]))

    def test_zero_cell_in_one_input(self, device, dtype):
        # A zero cell in target alone contributes 0; a zero cell in pred where target > 0 is the true KL, +inf (#5554).
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.1], device=device, dtype=dtype).view(1, 1, 2, 3)
        expected = (p[p > 0] * (p[p > 0].log() - q[p > 0].log())).sum()
        self.assert_close(kornia.losses.kl_div_loss_2d(q, p), expected)
        assert kornia.losses.kl_div_loss_2d(p, q).item() == math.inf

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    def test_zero_cell_in_target_alone_has_finite_gradients(self, device, dtype, loss):
        # A zero cell in target alone already contributed 0 to the value, but the gradient with respect to target was
        # NaN there (#5554).
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.1], device=device, dtype=dtype).view(1, 1, 2, 3)
        pred, target = q.clone().requires_grad_(), p.clone().requires_grad_()
        loss(pred, target).backward()
        assert torch.isfinite(pred.grad).all() and torch.isfinite(target.grad).all()

    @pytest.mark.parametrize("loss", [kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d])
    def test_invalid_cells_stay_nan(self, device, dtype, loss):
        # 0 * log 0 = 0 covers only target == 0 with a finite pred >= 0 (#5554): a NaN or a negative entry in either
        # input, or an infinite pred where target is 0, keeps the NaN of its (b, n) slice, and the other slices of the
        # batch are unaffected.
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.1], device=device, dtype=dtype).view(1, 1, 2, 3)
        pred, target = q.repeat(6, 1, 1, 1), p.repeat(6, 1, 1, 1)
        target[1, 0, 0, 0] = float("nan")
        target[2, 0, 0, 0] = -0.25
        pred[3, 0, 1, 0] = float("nan")  # target is 0 in this cell
        pred[4, 0, 1, 0] = -0.1  # target is 0 in this cell
        pred[5, 0, 1, 0] = math.inf  # target is 0 in this cell
        actual = loss(pred, target, reduction="none")
        self.assert_close(actual[0], loss(q, p, reduction="none")[0])
        assert torch.isnan(actual[1:]).all()
