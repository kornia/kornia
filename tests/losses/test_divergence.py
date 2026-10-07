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


class TestConventionsDivergence(BaseTester):
    """Pins for the direction, input form and reductions of the 2-D divergences, and validation (#5535, #5554)."""

    # Two distributions over a 2 x 3 grid with KL(P || Q) != KL(Q || P); every entry is exact in every dtype.
    _P = (12.0 / 16, 2.0 / 16, 1.0 / 16, 0.5 / 16, 0.25 / 16, 0.25 / 16)
    _Q = (2.0 / 16, 2.0 / 16, 2.0 / 16, 2.0 / 16, 4.0 / 16, 4.0 / 16)

    def _distributions(self, device, dtype):
        p = torch.tensor(self._P, device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor(self._Q, device=device, dtype=dtype).view(1, 1, 2, 3)
        return p, q

    @staticmethod
    def _batch(device, dtype):
        # (B, N, H, W) = (2, 3, 4, 6): every (b, n) slice is a different distribution over H x W
        g = torch.Generator().manual_seed(0)
        pred = torch.softmax(torch.randn(2, 3, 24, generator=g, dtype=torch.float64), -1).view(2, 3, 4, 6)
        target = torch.softmax(torch.randn(2, 3, 24, generator=g, dtype=torch.float64), -1).view(2, 3, 4, 6)
        return pred, target

    def test_convention_kl_div_loss_2d_is_kl_of_target_from_pred(self, device, dtype):
        # kl_div_loss_2d(pred, target) = KL(target || pred) = sum target * (log target - log pred) over H x W, the order
        # of torch's F.kl_div(pred.log(), target); scipy.stats.entropy(target.flatten(), pred.flatten()) gives the same
        # number. Inputs are probabilities: the log is taken inside.
        pred, target = self._distributions(device, dtype)
        kl_target_pred = sum(q * math.log(q / p) for p, q in zip(self._P, self._Q))  # 1.4223
        kl_pred_target = sum(p * math.log(p / q) for p, q in zip(self._P, self._Q))  # 1.1705
        assert abs(kl_target_pred - kl_pred_target) > 0.2
        expected = torch.tensor(kl_target_pred, device=device, dtype=dtype)
        self.assert_close(kornia.losses.kl_div_loss_2d(pred, target), expected)
        # A zero in target where pred > 0 adds nothing (0 log 0 = 0): 1/4 ln 2 + 3/4 ln 3 for this dyadic pair.
        full = torch.tensor([0.125, 0.125, 0.25, 0.25, 0.125, 0.125], device=device, dtype=dtype).view(1, 1, 2, 3)
        sparse = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        expected = torch.tensor(0.25 * math.log(2.0) + 0.75 * math.log(3.0), device=device, dtype=dtype)
        self.assert_close(kornia.losses.kl_div_loss_2d(full, sparse), expected)

    def test_convention_kl_div_loss_2d_reduces_over_batch_and_channels(self, device, dtype):
        # Each (b, n) slice is one distribution over H x W: 'none' returns (B, N), the default 'mean' averages over the
        # B N slices (torch's 'batchmean' on the (B N, H W) reshape) and 'sum' adds them.
        pred, target = self._batch(device, dtype)
        expected = (target * (target.log() - pred.log())).sum((-2, -1))
        pred, target, expected = pred.to(device, dtype), target.to(device, dtype), expected.to(device, dtype)
        none = kornia.losses.kl_div_loss_2d(pred, target, reduction="none")
        assert none.shape == (2, 3)
        self.assert_close(none, expected)
        self.assert_close(kornia.losses.kl_div_loss_2d(pred, target), expected.mean())
        # The direct PyTorch port is valid on this strictly positive support.
        torch_loss = torch.nn.functional.kl_div(pred.reshape(6, -1).log(), target.reshape(6, -1), reduction="batchmean")
        self.assert_close(kornia.losses.kl_div_loss_2d(pred, target), torch_loss)
        self.assert_close(kornia.losses.kl_div_loss_2d(pred, target, reduction="sum"), expected.sum())
        # Nothing is normalised: doubling both inputs doubles every slice's value.
        self.assert_close(kornia.losses.kl_div_loss_2d(2 * pred, 2 * target, reduction="none"), 2 * expected)
        # Relabelling check: permuting the channels of both inputs permutes the 'none' output.
        perm = [2, 0, 1]
        self.assert_close(kornia.losses.kl_div_loss_2d(pred[:, perm], target[:, perm], reduction="none"), none[:, perm])

    def test_convention_js_div_loss_2d_is_symmetric_and_at_most_ln_2(self, device, dtype):
        # JS = KL(P || M) / 2 + KL(Q || M) / 2 with M = (P + Q) / 2, in nats: symmetric in its arguments, and ln 2 for
        # distributions with disjoint supports. The default reduction is the mean over the (B, N) slices.
        p, q = self._distributions(device, dtype)
        m = [(a + b) / 2 for a, b in zip(self._P, self._Q)]
        js = 0.5 * sum(a * math.log(a / c) for a, c in zip(self._P, m)) + 0.5 * sum(
            b * math.log(b / c) for b, c in zip(self._Q, m)
        )  # 0.2689
        expected = torch.tensor(js, device=device, dtype=dtype)
        self.assert_close(kornia.losses.js_div_loss_2d(p, q), expected)
        self.assert_close(kornia.losses.js_div_loss_2d(q, p), expected)
        # Disjoint supports attain ln 2, including when both inputs contain an empty cell (#5554).
        left = torch.tensor([0.5, 0.25, 0.25, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        right = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.5], device=device, dtype=dtype).view(1, 1, 2, 3)
        ln2 = torch.tensor(math.log(2.0), device=device, dtype=dtype)
        self.assert_close(kornia.losses.js_div_loss_2d(left, right), ln2)
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        none = kornia.losses.js_div_loss_2d(pred, target, reduction="none")
        assert none.shape == (2, 3)
        self.assert_close(kornia.losses.js_div_loss_2d(pred, target), none.mean())

    def test_convention_div_loss_2d_rejects_unknown_reduction_5535(self, device, dtype):
        """An unknown reduction, torch's 'batchmean' included, raises NotImplementedError (#5535)."""
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        for loss_fn in (kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d):
            total = loss_fn(pred, target, reduction="sum")
            assert (total - loss_fn(pred, target, reduction="mean")).abs() > 0.1
            for reduction in ("batchmean", "MEAN", None):
                with pytest.raises(NotImplementedError, match="Invalid reduction mode"):
                    loss_fn(pred, target, reduction=reduction)

    def test_convention_kl_div_loss_2d_rejects_a_transposed_pred_5535(self, device, dtype):
        """A pred with H and W swapped is rejected instead of reinterpreted in target's layout (#5535)."""
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        transposed = pred.transpose(-2, -1).contiguous()
        assert transposed.shape == (2, 3, 6, 4)
        with pytest.raises(BaseError, match="pred and target shapes must be the same"):
            kornia.losses.kl_div_loss_2d(transposed, target)

    def test_convention_div_loss_2d_empty_cells_contribute_zero_5554(self, device, dtype):
        """A cell that is zero in both pred and target contributes zero (#5554)."""
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor([0.0, 0.5, 0.25, 0.25, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        full = torch.tensor([0.125, 0.125, 0.25, 0.25, 0.125, 0.125], device=device, dtype=dtype).view(1, 1, 2, 3)
        self.assert_close(kornia.losses.kl_div_loss_2d(p, p), p.new_zeros(()))
        self.assert_close(kornia.losses.js_div_loss_2d(p, p), p.new_zeros(()))
        m = (p + q) / 2
        expected = 0.5 * (
            (p[p > 0] * (p[p > 0] / m[p > 0]).log()).sum() + (q[q > 0] * (q[q > 0] / m[q > 0]).log()).sum()
        )
        self.assert_close(kornia.losses.js_div_loss_2d(p, q), expected)
        # The sparse PyTorch port must mask zero-target/finite-pred cells before the logarithm. This pair includes
        # shared zeros, a zero target with positive pred, and unequal positive entries.
        pred, target = full.clone(), p.clone()
        pred[..., 0, 0] = 0.0
        pred[..., 1, 2] += 0.125  # keep pred normalised without changing either positive target cell
        empty = (target == 0) & (pred >= 0) & torch.isfinite(pred)
        torch_loss = torch.nn.functional.kl_div(
            pred.masked_fill(empty, 1.0).reshape(1, -1).log(),
            target.masked_fill(empty, 1.0).reshape(1, -1),
            reduction="batchmean",
        )
        expected_kl = p.new_tensor(0.25 * math.log(2.0) + 0.75 * math.log(3.0))
        self.assert_close(torch_loss, expected_kl)
        self.assert_close(kornia.losses.kl_div_loss_2d(pred, target), torch_loss)
        batch = torch.cat([p, full])
        none = kornia.losses.kl_div_loss_2d(batch, batch, reduction="none")
        self.assert_close(none, torch.zeros_like(none))
        self.assert_close(kornia.losses.kl_div_loss_2d(batch, batch), p.new_zeros(()))
