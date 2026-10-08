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


def _lovasz_grad_reference(foreground_sorted):
    """Berman's lovasz_grad in float64, with the counts kept as Python integers (foreground_sorted holds 0. and 1.)."""
    total = int(foreground_sorted.sum().item())
    seen_foreground = seen_background = 0
    jaccard = []
    for value in foreground_sorted.tolist():
        if value:
            seen_foreground += 1
        else:
            seen_background += 1
        jaccard.append(1.0 - (total - seen_foreground) / (total + seen_background))
    out = torch.tensor(jaccard, dtype=torch.float64)
    out[1:] = out[1:] - out[:-1]
    return out


def _lovasz_hinge_reference(logits, labels):
    """kornia's reduction of Berman's per-image Lovasz hinge, evaluated in float64 one sample at a time."""
    logits = logits.double()
    loss = torch.zeros((), dtype=torch.float64)
    for b in range(logits.shape[0]):
        target = labels[b].double().flatten()
        errors = (1.0 - logits[b].flatten() * (2.0 * target - 1.0)).relu()
        errors_sorted, permutation = errors.sort(descending=True)
        loss = loss + errors_sorted.dot(_lovasz_grad_reference(target[permutation])) / logits.shape[0]
    return loss


class TestLovaszHingeLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 1, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 1) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszHingeLoss()
        assert criterion(logits, labels) is not None

    def test_exception(self):
        criterion = kornia.losses.LovaszHingeLoss()

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 2), torch.rand(1, 1, 1))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

    def test_multi_class(self, device, dtype):
        num_classes = 5
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszHingeLoss()
        with pytest.raises(Exception):
            criterion(logits, labels)

    def test_perfect_prediction(self, device, dtype):
        num_classes = 1
        prediction = torch.ones(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszHingeLoss()
        loss = criterion(prediction, labels)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_output_dtype_follows_the_prediction(self, device, dtype):
        logits = torch.randn(2, 1, 4, 5, device=device, dtype=dtype)
        labels = torch.randint(0, 2, (2, 4, 5), device=device)
        assert kornia.losses.lovasz_hinge_loss(logits, labels).dtype == dtype
        assert kornia.losses.LovaszHingeLoss()(logits, labels).dtype == dtype

    def test_default_dtype_does_not_change_the_loss(self, device, dtype):
        logits = torch.randn(2, 1, 4, 5, device=device, dtype=dtype)
        labels = torch.randint(0, 2, (2, 4, 5), device=device)
        expected = kornia.losses.lovasz_hinge_loss(logits, labels)
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32 if original == torch.float64 else torch.float64)
            loss = kornia.losses.lovasz_hinge_loss(logits, labels)
        finally:
            torch.set_default_dtype(original)
        assert loss.dtype == dtype
        self.assert_close(loss, expected)

    def test_float64_matches_a_float64_reference(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 16, 24, device=device, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(0, 2, (2, 16, 24), device=device)
        reference_logits = logits.detach().cpu().clone().requires_grad_()
        expected = _lovasz_hinge_reference(reference_logits, labels.cpu())
        expected_grad = torch.autograd.grad(expected, reference_logits)[0]
        loss = kornia.losses.lovasz_hinge_loss(logits, labels)
        # the Jaccard weights are exact in float64: only the summation order differs from the reference
        self.assert_close(loss, expected.to(device), rtol=1e-12, atol=1e-12)
        self.assert_close(torch.autograd.grad(loss, logits)[0], expected_grad.to(device), rtol=1e-12, atol=1e-15)

    def test_large_foreground_counts(self, device, dtype):
        # More than 65504 foreground pixels exceed float16's finite range, and cumulative counts above 256 (bfloat16)
        # or 2048 (float16) are not exact in half precision.
        logits = torch.zeros((1, 1, 257, 257), device=device, dtype=dtype, requires_grad=True)
        labels = torch.ones((1, 257, 257), device=device, dtype=torch.int64)
        loss = kornia.losses.lovasz_hinge_loss(logits, labels)
        # every error is 1 and the Jaccard weights sum to the last Jaccard index, 1
        self.assert_close(loss, torch.tensor(1.0, device=device, dtype=dtype))
        assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()

    @pytest.mark.parametrize(
        "logits, expected",
        [
            # errors 1 - logit * sign: [-1, 0, 1, -2]; sorted labels [0, 0, 1, 1] weigh them by [1/3, 1/6, 1/4, 1/4]
            ([[2, -1], [0, 3]], 1 / 3),
            # errors [0, 1, 1, 0]: the two errors of 1 take the weights 1/3 and 1/6
            ([[True, False], [False, True]], 1 / 2),
        ],
    )
    def test_integer_logits_return_a_float_loss(self, device, logits, expected):
        logits = torch.tensor([[logits]], device=device)
        labels = torch.tensor([[[1, 0], [0, 1]]], device=device)
        loss = kornia.losses.lovasz_hinge_loss(logits, labels)
        assert loss.dtype == torch.float32
        self.assert_close(loss, torch.tensor(expected, device=device))

    def test_gradcheck(self, device, dtype):
        dtype = torch.float64
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device, dtype=dtype)

        self.gradcheck(kornia.losses.lovasz_hinge_loss, (logits, labels))

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_module = kornia.losses.LovaszHingeLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
