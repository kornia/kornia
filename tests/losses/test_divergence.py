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


class TestConventionsDivergence(BaseTester):
    """Pins for the direction, input form and reductions of the 2-D divergences, and their warts (#5535, #5554)."""

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
        # of torch's F.kl_div(pred.log(), target); scipy.stats.entropy(target, pred) gives the same number. Inputs are
        # probabilities: the log is taken inside.
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
        # disjoint supports that together cover the grid (a cell empty in both inputs gives NaN, #5554)
        left = torch.tensor([0.5, 0.25, 0.25, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        right = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.5], device=device, dtype=dtype).view(1, 1, 2, 3)
        ln2 = torch.tensor(math.log(2.0), device=device, dtype=dtype)
        self.assert_close(kornia.losses.js_div_loss_2d(left, right), ln2)
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        none = kornia.losses.js_div_loss_2d(pred, target, reduction="none")
        assert none.shape == (2, 3)
        self.assert_close(kornia.losses.js_div_loss_2d(pred, target), none.mean())

    def test_wart_div_loss_2d_unknown_reduction_returns_the_sum_5535(self, device, dtype):
        """An unknown reduction, torch's 'batchmean' included, silently returns the 'sum' value (#5535)."""
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        for loss_fn in (kornia.losses.kl_div_loss_2d, kornia.losses.js_div_loss_2d):
            total = loss_fn(pred, target, reduction="sum")
            assert (total - loss_fn(pred, target, reduction="mean")).abs() > 0.1
            for reduction in ("batchmean", "MEAN", None):
                self.assert_close(loss_fn(pred, target, reduction=reduction), total, rtol=0.0, atol=0.0)

    def test_wart_kl_div_loss_2d_reinterprets_a_transposed_pred_5535(self, device, dtype):
        """The shapes are not validated: a pred with H and W swapped is read in target's layout (#5535)."""
        pred, target = self._batch(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)
        transposed = pred.transpose(-2, -1).contiguous()
        assert transposed.shape == (2, 3, 6, 4)
        actual = kornia.losses.kl_div_loss_2d(transposed, target)
        self.assert_close(actual, kornia.losses.kl_div_loss_2d(transposed.reshape(target.shape), target))

    def test_wart_div_loss_2d_is_nan_for_a_cell_empty_in_both_inputs_5554(self, device, dtype):
        """A cell that is zero in both pred and target gives 0 * log 0 = NaN instead of 0 (#5554)."""
        p = torch.tensor([0.0, 0.25, 0.75, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        q = torch.tensor([0.0, 0.5, 0.25, 0.25, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 2, 3)
        full = torch.tensor([0.125, 0.125, 0.25, 0.25, 0.125, 0.125], device=device, dtype=dtype).view(1, 1, 2, 3)
        assert torch.isnan(kornia.losses.kl_div_loss_2d(p, p))
        assert torch.isnan(kornia.losses.js_div_loss_2d(p, p))
        assert torch.isnan(kornia.losses.js_div_loss_2d(p, q))  # different inputs that share one empty cell
        batch = torch.cat([p, full])  # one poisoned slice: 'none' isolates it, the default 'mean' propagates it
        none = kornia.losses.kl_div_loss_2d(batch, batch, reduction="none")
        assert torch.isnan(none[0, 0])
        assert none[1, 0] == 0
        assert torch.isnan(kornia.losses.kl_div_loss_2d(batch, batch))
