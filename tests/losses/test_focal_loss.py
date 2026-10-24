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

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import kornia
from kornia.core.utils import is_exporting

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

one_hot_module = importlib.import_module("kornia.losses.one_hot")


class TestBinaryFocalLossWithLogits(BaseTester):
    def test_exception_inputs(self, device, dtype):
        from kornia.core.exceptions import DeviceError, ShapeError, TypeCheckError

        pred = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        target = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError):
            kornia.losses.binary_focal_loss_with_logits(None, target)

        with pytest.raises(TypeCheckError):
            kornia.losses.binary_focal_loss_with_logits(pred, [1, 2, 3])

        with pytest.raises(DeviceError):
            kornia.losses.binary_focal_loss_with_logits(pred, target.to("meta"))

        with pytest.raises(DeviceError):
            kornia.losses.binary_focal_loss_with_logits(pred, target, weight=torch.ones(3, device="meta", dtype=dtype))

        with pytest.raises(DeviceError):
            kornia.losses.binary_focal_loss_with_logits(
                pred, target, pos_weight=torch.ones(3, device="meta", dtype=dtype)
            )

        with pytest.raises(ShapeError):
            kornia.losses.binary_focal_loss_with_logits(pred, target[:, :2])

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_value_same_as_torch_bce_loss(self, device, dtype, reduction, ignore_index):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        focal_equivalent_bce_loss = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=None, gamma=0, reduction=reduction, ignore_index=ignore_index
        )
        torch_bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
        self.assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_value_same_as_torch_bce_loss_pos_weight_weight(self, device, dtype, reduction):
        num_classes = 3
        logits = torch.rand(2, num_classes, 2, dtype=dtype, device=device)
        labels = torch.rand(2, num_classes, 2, dtype=dtype, device=device)

        pos_weight = torch.rand(num_classes, 1, dtype=dtype, device=device)
        weight = torch.rand(num_classes, 1, dtype=dtype, device=device)

        focal_equivalent_bce_loss = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=None, gamma=0, reduction=reduction, pos_weight=pos_weight, weight=weight
        )
        torch_bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction=reduction, pos_weight=pos_weight, weight=weight
        )
        self.assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("alpha", [None, 0.2, 0.5])
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=alpha, gamma=gamma, reduction=reduction
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("pos_weight", [None, (1, 2, 5)])
    @pytest.mark.parametrize("weight", [None, (0.2, 0.5, 0.8)])
    def test_shape_pos_weight_weight(self, device, dtype, reduction, expected_shape, pos_weight, weight):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        pos_weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=dtype, device=device)
        weight = None if weight is None else torch.tensor(weight, dtype=dtype, device=device)

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, pos_weight=pos_weight, weight=weight
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_shape_ignore_index(self, device, dtype, reduction, expected_shape, ignore_index):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, ignore_index=ignore_index
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("gamma", [0.5, 2.0])
    def test_dynamo(self, device, dtype, torch_optimizer, gamma):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        op = kornia.losses.binary_focal_loss_with_logits
        op_optimized = torch_optimizer(op)

        args = (0.25, gamma)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        self.assert_close(actual, expected)

    def test_jit_script(self, device, dtype):
        # TorchScript compiles the loss with its weight checks, whose messages format shapes rather than a lambda
        logits = torch.rand(2, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2, device=device, dtype=dtype).round()
        pos_weight = torch.rand(3, device=device, dtype=dtype)
        weight = torch.rand(3, 1, device=device, dtype=dtype)
        op = kornia.losses.binary_focal_loss_with_logits
        op_script = torch.jit.script(op)
        for args in ((0.25, 2.0, "mean", None, None), (0.25, 2.0, "mean", pos_weight, weight)):
            self.assert_close(op_script(logits, labels, *args), op(logits, labels, *args))
        with pytest.raises(Exception, match="pos_weight must hold one value per class of pred"):
            op_script(logits, labels, 0.25, 2.0, "mean", torch.rand(4, device=device, dtype=dtype), None)

    @pytest.mark.parametrize("gamma", [0.5, 2.0])
    def test_gradcheck(self, device, dtype, gamma):
        logits = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        labels = torch.rand(2, 3, 2, device=device, dtype=torch.float64)

        args = (0.25, gamma)
        op = kornia.losses.binary_focal_loss_with_logits
        self.gradcheck(op, (logits, labels, *args))

    def test_gradcheck_ignore_index(self, device, dtype):
        logits = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        labels = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        self.gradcheck(op, (logits, labels, *args), requires_grad=[True, False, False, False])

    def test_module(self, device, dtype):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        op_module = kornia.losses.BinaryFocalLossWithLogits(*args)
        self.assert_close(op_module(logits, labels), op(logits, labels, *args))

    def test_numeric_stability(self, device, dtype):
        logits = torch.tensor([[100.0, -100]], dtype=dtype, device=device)
        labels = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)

        args = (0.25, 2.0)
        actual = kornia.losses.binary_focal_loss_with_logits(logits, labels, *args)
        expected = torch.tensor([[0.0, 0.0]], dtype=dtype, device=device)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75, 2.0])
    @pytest.mark.parametrize("alpha", [None, 0.25])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_numeric_stability_backward(self, device, dtype, gamma, alpha, reduction):
        logits = torch.tensor([[1000.0, -1000.0], [1000.0, -1000.0]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        # The focal objective tends to zero for correct predictions and |logit| for incorrect predictions.
        expected = torch.tensor([[0.0, 0.0], [1000.0, 1000.0]], device=device, dtype=dtype)
        expected_grad = torch.tensor([[0.0, 0.0], [1.0, -1.0]], device=device, dtype=dtype)
        if alpha is not None:
            factors = torch.tensor([[alpha, 1.0 - alpha], [1.0 - alpha, alpha]], device=device, dtype=dtype)
            expected = expected * factors
            expected_grad = expected_grad * factors
        if reduction == "mean":
            expected = expected.mean()
            expected_grad = expected_grad / logits.numel()
        elif reduction == "sum":
            expected = expected.sum()

        actual = kornia.losses.binary_focal_loss_with_logits(logits, labels, alpha, gamma, reduction)
        self.assert_close(actual, expected)
        self.assert_close(torch.autograd.grad(actual.sum(), logits)[0], expected_grad)


class TestFocalLoss(BaseTester):
    def test_exception_inputs(self, device, dtype):
        from kornia.core.exceptions import DeviceError, ShapeError, TypeCheckError

        pred = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        target = torch.randint(0, 3, (2, 4, 4), device=device)

        with pytest.raises(TypeCheckError):
            kornia.losses.focal_loss(None, target, 0.5)

        with pytest.raises(TypeCheckError):
            kornia.losses.focal_loss(pred, [1, 2, 3], 0.5)

        with pytest.raises(DeviceError):
            kornia.losses.focal_loss(pred, target.to("meta"), 0.5)

        with pytest.raises(DeviceError):
            kornia.losses.focal_loss(pred, target, 0.5, weight=torch.ones(3, device="meta", dtype=dtype))

        with pytest.raises(ShapeError):
            kornia.losses.focal_loss(torch.rand(4, device=device, dtype=dtype), target, 0.5)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(2, 4), (2, 4, 3, 5), (0, 4, 3)])
    @pytest.mark.parametrize("ignore_index", [None, -100, 0])
    def test_reduced_matches_dense_value_and_gradient(self, device, dtype, reduction, shape, ignore_index):
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(shape, generator=generator).to(device=device, dtype=dtype).requires_grad_()
        labels = torch.randint(shape[1], (shape[0], *shape[2:]), generator=generator).to(device)
        if labels.numel() and ignore_index is not None:
            labels.reshape(-1)[0] = ignore_index
        if logits.ndim > 3:
            logits = logits.transpose(-1, -2)
            labels = labels.transpose(-1, -2) if labels.ndim > 2 else labels
        # Compare the public dense contract, including empty means and the class-count denominator.
        dense = kornia.losses.focal_loss(logits, labels, 0.25, 0.5, "none", ignore_index=ignore_index)
        expected = dense.mean() if reduction == "mean" else dense.sum()
        actual = kornia.losses.focal_loss(logits, labels, 0.25, 0.5, reduction, ignore_index=ignore_index)
        if logits.numel() == 0 and reduction == "mean":
            assert actual.isnan()
            assert expected.isnan()
        else:
            self.assert_close(actual, expected)
        self.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("alpha,gamma", [(0.25, -1.0), (float("inf"), 2.0), (0.25, float("nan"))])
    def test_reduced_unusual_parameters_match_dense(self, device, dtype, reduction, alpha, gamma):
        logits = torch.tensor([[1000.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([1], device=device)
        dense = kornia.losses.focal_loss(logits, labels, alpha, gamma, "none")
        expected = dense.mean() if reduction == "mean" else dense.sum()
        actual = kornia.losses.focal_loss(logits, labels, alpha, gamma, reduction)
        assert actual.isnan()
        assert expected.isnan()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduced_weight_gradient_and_promotion(self, device, dtype, reduction):
        logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([0, 2], device=device)
        weight = torch.tensor([0.5, 1.5, 2.0], device=device, dtype=torch.float32, requires_grad=True)
        dense = kornia.losses.focal_loss(logits, labels, 0.25, 0.5, "none", weight)
        expected = dense.mean() if reduction == "mean" else dense.sum()
        actual = kornia.losses.focal_loss(logits, labels, 0.25, 0.5, reduction, weight)
        self.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual, (logits, weight))
        expected_grads = torch.autograd.grad(expected, (logits, weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_close(actual_grad, expected_grad)

    @pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True to a scalar")
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("alpha_shape", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_reduced_one_element_tensor_alpha_6026(self, device, dtype, reduction, alpha_shape, requires_grad):
        # The reduced path passed a tensor alpha to new_full, which needs a Python number: a (1,) or (1, 1) alpha, or
        # one that requires grad, raised TypeError there although the dense path accepts them.
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(2, 4, 3, 5, generator=generator).to(device=device, dtype=dtype)
        labels = torch.randint(4, (2, 3, 5), generator=generator).to(device)
        alpha = torch.full(alpha_shape, 0.3, device=device, requires_grad=requires_grad)

        actual = kornia.losses.focal_loss(logits, labels, alpha, 2.0, reduction)

        dense = kornia.losses.focal_loss(logits, labels, alpha, 2.0, "none")
        self.assert_close(actual, dense.mean() if reduction == "mean" else dense.sum())
        self.assert_close(actual, kornia.losses.focal_loss(logits, labels, float(alpha), 2.0, reduction))
        self.assert_close(kornia.losses.FocalLoss(alpha, 2.0, reduction)(logits, labels), actual)

    # The name keeps "compile" and "dynamo" out, so the ordinary jobs run it.
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("alpha_shape", [(), (1,)])
    def test_tensor_alpha_traces_in_one_graph_6026(self, device, dtype, reduction, alpha_shape):
        # A tensor alpha takes the dense path: the reduced path's range check on alpha would be data-dependent.
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(2, 4, 3, 5, generator=generator).to(device=device, dtype=dtype)
        labels = torch.randint(4, (2, 3, 5), generator=generator).to(device)
        alpha = torch.full(alpha_shape, 0.3, device=device)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.losses.focal_loss, backend="eager", fullgraph=True)
        expected = kornia.losses.focal_loss(logits, labels, alpha, 2.0, reduction)
        self.assert_close(compiled(logits, labels, alpha, 2.0, reduction), expected)

    def test_reduced_alpha_does_not_synchronize(self, device, dtype):
        if device.type != "cuda":
            pytest.skip("CUDA synchronization check")
        logits = torch.randn(2, 4, 3, 5, device=device, dtype=dtype, requires_grad=True)
        labels = torch.randint(4, (2, 3, 5), device=device)
        expected = kornia.losses.focal_loss(logits, labels, 0.25, 2.0, "none").mean()
        previous_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            actual = kornia.losses.focal_loss(logits, labels, 0.25, 2.0, "mean")
        finally:
            torch.cuda.set_sync_debug_mode(previous_sync_debug_mode)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("alpha", [0.25, 1])
    def test_reduced_saved_tensor_memory(self, device, dtype, alpha):
        # A reduced objective should not retain dense per-class one-hot and focal-factor intermediates.
        logits = torch.zeros(2, 21, 4, 5, device=device, dtype=dtype, requires_grad=True)
        labels = torch.zeros(2, 4, 5, device=device, dtype=torch.int64)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
            kornia.losses.focal_loss(logits, labels, alpha, reduction="mean")
        assert sum(value.numel() * value.element_size() for value in saved) < 4 * logits.numel() * logits.element_size()

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("alpha", [None, 0.25])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_overflowing_non_target_log_probability_5628(self, device, dtype, reduction, gamma, alpha, weighted):
        # Exact float16/float32 reproducers from #5628; exercise the same overflow in the other dtypes too.
        big = {torch.float16: 40000.0, torch.float32: 3e38}.get(dtype, 0.75 * torch.finfo(dtype).max)
        logits = torch.tensor([[big, -big, 0.0], [1.0, 2.0, 3.0]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([0, 2], device=device)
        assert logits.isfinite().all()
        assert logits.log_softmax(1)[0, 1].isneginf()
        weight = torch.tensor([0.5, 1.5, 2.0], device=device, dtype=dtype) if weighted else None
        # A finite logit gap gives the same saturated probabilities, without overflowing log-softmax.
        control = torch.tensor(
            [[10000.0, -10000.0, 0.0], [1.0, 2.0, 3.0]], device=device, dtype=dtype, requires_grad=True
        )
        expected = kornia.losses.focal_loss(control, labels, alpha, gamma, reduction, weight)
        expected_grad = torch.autograd.grad(expected.sum(), control)[0]
        actual = kornia.losses.focal_loss(logits, labels, alpha, gamma, reduction, weight)
        assert actual.isfinite().all()
        self.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.autograd.grad(actual.sum(), logits)[0]
        assert grad.isfinite().all()
        self.assert_close(grad[0], torch.zeros_like(grad[0]), rtol=0, atol=0)
        self.assert_close(grad[1], expected_grad[1])
        if reduction == "none":
            self.assert_close(actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("alpha", [None, 0.25])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_finite_loss_and_gradient(self, device, dtype, reduction, gamma, alpha, weighted):
        logits = torch.tensor(
            [[[1.0, -0.5], [2.0, 0.3], [3.0, 1.5]], [[-1.0, 0.2], [0.5, -0.7], [0.0, 1.0]]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.tensor([[2, 0], [1, 2]], device=device)
        weight = torch.tensor([0.5, 1.5, 2.0], device=device, dtype=dtype) if weighted else None
        # Original finite-input formula, before masking non-target log probabilities.
        reference = logits.detach().clone().requires_grad_()
        logp = reference.log_softmax(1)
        expected = -(1.0 - logp.exp()).pow(gamma) * logp * F.one_hot(labels, 3).movedim(-1, 1).to(dtype)
        if alpha is not None:
            expected = torch.tensor([1.0 - alpha, alpha, alpha], device=device, dtype=dtype).view(3, 1) * expected
        if weight is not None:
            expected = weight.view(3, 1) * expected
        if reduction == "mean":
            expected = expected.mean()
        elif reduction == "sum":
            expected = expected.sum()
        actual = kornia.losses.focal_loss(logits, labels, alpha, gamma, reduction, weight)
        if not weighted and reduction != "none":
            # Gathering target terms changes the reduction order relative to the dense reference.
            self.assert_close(actual, expected)
        else:
            self.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_close(
            torch.autograd.grad(actual.sum(), logits)[0], torch.autograd.grad(expected.sum(), reference)[0]
        )

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("gamma", [0.0, 2.0])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("spatial", [False, True])
    def test_ignored_extreme_logits(self, device, dtype, reduction, gamma, weighted, spatial):
        extreme = torch.finfo(dtype).max
        logits = torch.tensor([[extreme, -extreme], [0.0, 0.0]], device=device, dtype=dtype)
        labels = torch.tensor([-100, 0], device=device)
        if spatial:
            logits = logits.T.reshape(1, 2, 1, 2)
            labels = labels.reshape(1, 1, 2)
        logits = logits.requires_grad_()
        weight = torch.tensor([0.5, 1.5], device=device, dtype=dtype) if weighted else None
        alpha = 0.25 if weighted else None
        safe_logits = torch.zeros_like(logits, requires_grad=True)
        op = kornia.losses.focal_loss
        expected = op(safe_logits, labels, alpha, gamma, reduction, weight)
        actual = op(logits, labels, alpha, gamma, reduction, weight)
        self.assert_close(actual, expected)
        self.assert_close(
            torch.autograd.grad(actual.sum(), logits)[0], torch.autograd.grad(expected.sum(), safe_logits)[0]
        )
        self.assert_close(kornia.losses.FocalLoss(alpha, gamma, reduction, weight)(logits, labels), expected)

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_all_ignored_extreme_logits(self, device, dtype, ignore_index):
        extreme = torch.finfo(dtype).max
        logits = torch.tensor([[extreme, -extreme]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.full((1,), ignore_index, device=device, dtype=torch.int64)
        loss = kornia.losses.focal_loss(logits, labels, alpha=None, reduction="sum", ignore_index=ignore_index)
        self.assert_close(loss, torch.zeros_like(loss))
        self.assert_close(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits))

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_ignored_non_finite_logits(self, device, dtype, value):
        logits = torch.tensor([[value, 0.0], [0.3, -0.2]], device=device, dtype=dtype, requires_grad=True)
        safe_logits = torch.tensor([[0.0, 0.0], [0.3, -0.2]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([-100, 1], device=device)
        actual = kornia.losses.focal_loss(logits, labels, alpha=0.25, reduction="sum")
        expected = kornia.losses.focal_loss(safe_logits, labels, alpha=0.25, reduction="sum")
        self.assert_close(actual, expected)
        grad = torch.autograd.grad(actual, logits)[0]
        assert (grad[0] == 0).all()
        self.assert_close(grad, torch.autograd.grad(expected, safe_logits)[0])

    @pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    @pytest.mark.parametrize("alpha", [None, 0.25])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_saturated_logits_backward(self, device, dtype, gamma, alpha, reduction):
        # Row 0 is saturated on its target class, row 1 on a wrong class.
        logits = torch.tensor([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.tensor([0, 0], device=device)
        # Every probability rounds to 0 or 1: (1 - p) ** gamma is 1 where p = 0 and multiplies log(p) = 0 where p = 1,
        # so the loss and its gradient reduce to the gamma = 0 ones, which have no singularity.
        cross_entropy_logits = logits.detach().clone().requires_grad_()
        expected = kornia.losses.focal_loss(cross_entropy_logits, labels, alpha, 0.0, reduction)
        expected_grad = torch.autograd.grad(expected.sum(), cross_entropy_logits)[0]
        if reduction == "sum":
            # A correct saturated prediction costs nothing and a wrong one its logit gap.
            self.assert_close(
                expected, torch.tensor(1000.0 * (1.0 - alpha if alpha else 1.0), device=device, dtype=dtype)
            )

        actual = kornia.losses.focal_loss(logits, labels, alpha, gamma, reduction)
        self.assert_close(actual, expected)
        # detect_anomaly also rejects a NaN inside the backward that torch.where would discard, so this pins the safe
        # base of the unselected branch and not only the final gradient.
        with torch.autograd.detect_anomaly():
            grad = torch.autograd.grad(actual.sum(), logits)[0]
        self.assert_close(grad, expected_grad)

    @pytest.mark.parametrize("gamma", [0.5, 2.0])
    def test_dynamo_saturated_logits(self, device, dtype, torch_optimizer, gamma):
        logits = torch.tensor([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]], device=device, dtype=dtype)
        labels = torch.tensor([0, 0], device=device)
        op = kornia.losses.focal_loss
        op_optimized = torch_optimizer(op)
        self.assert_close(op_optimized(logits, labels, 0.25, gamma), op(logits, labels, 0.25, gamma))

    def test_dynamo_ignored_extreme_logits(self, device, dtype, torch_optimizer):
        extreme = torch.finfo(dtype).max
        logits = torch.tensor([[extreme, -extreme]], device=device, dtype=dtype, requires_grad=True)
        labels = torch.full((1,), -100, device=device, dtype=torch.int64)
        op = torch_optimizer(kornia.losses.focal_loss)
        loss = op(logits, labels, None, reduction="sum")
        self.assert_close(loss, torch.zeros_like(loss))
        self.assert_close(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits))

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("alpha", [None, 0.2, 0.5])
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3)), ("mean", ()), ("sum", ())])
    def test_shape_target_with_only_one_dim(self, device, dtype, reduction, expected_shape):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=0.1, gamma=1.5, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("weight", [None, (0.2, 0.5, 0.8)])
    def test_shape_weight(self, device, dtype, reduction, expected_shape, weight):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        weight = None if weight is None else torch.tensor(weight, dtype=dtype, device=device)

        actual_shape = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, weight=weight
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_shape_ignore_index(self, device, dtype, reduction, expected_shape, ignore_index):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        actual_shape = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, ignore_index=ignore_index
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_value_ignore_index(self, device, dtype, ignore_index):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        labels_extra_class = labels.clone()
        labels_extra_class[ignore] = num_classes
        logits_extra_class = torch.cat([logits, logits.new_full((2, 1, 3, 2), float("-inf"))], dim=1)

        expected_values = kornia.losses.focal_loss(
            logits_extra_class, labels_extra_class, alpha=0.8, gamma=0.5, reduction="none"
        )[:, :-1, ...]

        actual_values = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction="none", ignore_index=ignore_index
        )

        self.assert_close(actual_values, expected_values)

    def test_value_non_target_classes_are_zero(self, device, dtype):
        # The target is an exact one-hot, so the per-class values of a pixel are 0 off its class.
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        values = kornia.losses.focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction="none")

        off_class = F.one_hot(labels, num_classes).movedim(-1, 1) == 0
        self.assert_close(values[off_class], torch.zeros_like(values[off_class]), rtol=0, atol=0)
        assert (values[~off_class] > 0).all()

    @pytest.mark.parametrize("reduction, weighted", [("mean", False), ("sum", False), ("mean", True), ("none", False)])
    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_exception_labels_outside_the_classes(self, device, dtype, reduction, weighted, ignore_index):
        # Unweighted 'mean' and 'sum' gather the target class, the others build a one-hot target: both raise
        # one_hot's errors. The gather used to raise its own "index ... is out of bounds".
        if device.type != "cpu":
            pytest.skip("an out-of-range index fails a device-side assert off the CPU, which ends the CUDA context")
        num_classes = 3
        logits = torch.rand(2, num_classes, 4, device=device, dtype=dtype)
        weight = torch.ones(num_classes, device=device, dtype=dtype) if weighted else None
        for value, message in ((num_classes, "smaller than num_classes"), (-3, "non-negative")):
            labels = torch.randint(num_classes, (2, 4), device=device)
            labels[0, 1] = value
            with pytest.raises(RuntimeError, match=f"Class values must be {message}"):
                kornia.losses.focal_loss(logits, labels, 0.25, 2.0, reduction, weight, ignore_index)

    # The name keeps "compile" and "dynamo" out, so the ordinary jobs run it.
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduced_label_check_traces_in_one_graph(self, device, dtype, reduction):
        # The label check of the gather path is captured with the loss, in one graph, and raises when it runs.
        if device.type != "cpu":
            pytest.skip("the label check runs on the CPU only")
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        logits = torch.rand(2, 3, 4, device=device, dtype=dtype)
        labels = torch.randint(3, (2, 4), device=device)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.losses.focal_loss, backend=backend, fullgraph=True)
        expected = kornia.losses.focal_loss(logits, labels, 0.25, 2.0, reduction)
        self.assert_close(compiled(logits, labels, 0.25, 2.0, reduction), expected)
        assert len(graphs) == 1
        nodes = graphs[0].graph.nodes
        assert sum(node.target is torch._assert_async for node in nodes) == 2
        # Compiled from torch 2.9 on, the gather reads clamped labels, in bounds in whatever order Inductor runs the
        # asserts. They stay unclamped where a torch.compile capture also reports an export (torch 2.9.1).
        index = next(node for node in nodes if node.target == "gather").args[2]
        while index.target == "unsqueeze":
            index = index.args[0]
        assert (index.target == "clamp") == (
            one_hot_module._INDUCTOR_KEEPS_ASSERT_ASYNC and not _compile_reports_export()
        )
        labels[0, 1] = 3
        with pytest.raises(RuntimeError, match="Class values must be smaller than num_classes"):
            compiled(logits, labels, 0.25, 2.0, reduction)
        assert len(graphs) == 1

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dynamo(self, device, dtype, torch_optimizer, reduction):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        op = kornia.losses.focal_loss
        op_optimized = torch_optimizer(op)

        args = (0.25, 2.0, reduction)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        self.assert_close(actual, expected)

    def test_jit_script(self, device, dtype):
        # TorchScript compiles the loss with its class and weight checks, whose messages format shapes
        num_classes = 3
        logits = torch.rand(2, num_classes, 4, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 4), device=device)
        weight = torch.rand(num_classes, device=device, dtype=dtype)
        op = kornia.losses.focal_loss
        op_script = torch.jit.script(op)
        for w in (None, weight):
            self.assert_close(op_script(logits, labels, 0.25, 2.0, "mean", w), op(logits, labels, 0.25, 2.0, "mean", w))
        with pytest.raises(Exception, match="weight must hold one value per class of pred"):
            op_script(logits, labels, 0.25, 2.0, "mean", torch.rand(num_classes + 1, device=device, dtype=dtype))

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(num_classes, (2, 3, 2), device=device).long()
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100

        self.gradcheck(
            kornia.losses.focal_loss, (logits, labels, 0.25, 2.0), dtypes=[torch.float64, torch.int64, None, None]
        )

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        args = (0.25, 2.0)
        op = kornia.losses.focal_loss
        op_module = kornia.losses.FocalLoss(*args)
        self.assert_close(op_module(logits, labels), op(logits, labels, *args))


def _compile_reports_export() -> bool:
    # Some releases (2.5.1 and 2.9.1 checked) also report a torch.compile capture as an export.
    probe = torch.compile(lambda x: x + float(is_exporting()), backend="eager", fullgraph=True)
    return bool(probe(torch.zeros(())) > 0)


# Run in a fresh interpreter: an index check that fails inside an Inductor CPU kernel aborts the whole process.
_COMPILED_INVALID_LABEL = """
import sys, warnings
warnings.filterwarnings("ignore")
import torch, kornia
reduction, ignore_index, value, requires_grad = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
generator = torch.Generator().manual_seed(0)
pred = torch.randn(2, 21, 128, 128, generator=generator).requires_grad_(requires_grad)
target = torch.randint(0, 21, (2, 128, 128), generator=generator)
target[0, 5, 7] = value
loss = torch.compile(kornia.losses.focal_loss, fullgraph=True)
try:
    out = loss(pred, target, 0.25, 2.0, reduction, None, None if ignore_index == "None" else int(ignore_index))
    if requires_grad:
        out.backward()
except RuntimeError as error:
    print("RuntimeError:", error)
else:
    print("no error")
"""


@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
class TestFocalLossCompiledLabels(BaseTester):
    @pytest.mark.parametrize(
        "reduction, ignore_index, requires_grad", [("mean", -100, False), ("mean", -100, True), ("sum", None, False)]
    )
    @pytest.mark.parametrize("value, message", [(21, "smaller than num_classes"), (-3, "non-negative")])
    def test_dynamo_invalid_label_raises(
        self, reduction, ignore_index, requires_grad, value, message, device, dtype, optimizer_backend
    ):
        # Compiled, the target-class gather of 'mean' and 'sum' terminated the process (exit code 134, "index out of
        # bounds") where eager raises: for a label past the classes on torch 2.5.1, 2.9.1 and 2.14, and for a
        # negative one on 2.14; 2.5.1 and 2.9.1 returned a loss for it without gradients. The check does not depend
        # on is_exporting(), which a torch.compile capture also reports on 2.5.1 and 2.9.1, so 2.9.1 runs it too.
        if device.type != "cpu" or dtype != torch.float32 or optimizer_backend != "inductor":
            pytest.skip("one fresh-process inductor compile per case, on the CPU in float32")
        from torch._inductor.decomposition import decompositions

        if torch.ops.aten._assert_async.msg in decompositions:
            pytest.skip("this torch release's Inductor compiles torch._assert_async to nothing (before torch 2.9)")
        # The checkout under test, which "-c" imports first from the working directory.
        root = Path(kornia.__file__).resolve().parents[1]
        result = subprocess.run(  # noqa: S603 - this interpreter and a literal program; no shell or external input.
            [
                sys.executable,
                "-c",
                _COMPILED_INVALID_LABEL,
                reduction,
                str(ignore_index),
                str(value),
                str(int(requires_grad)),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.startswith("RuntimeError:"), result.stdout
        assert f"Class values must be {message}" in result.stdout


class TestConventionsFocalLoss(BaseTester):
    """Pins for the class weighting, reductions, ignored labels and known defect of the focal losses."""

    @staticmethod
    def _logits_and_labels(device, dtype, scale=1.0):
        # two 4 x 5 images with different class balances
        g = torch.Generator().manual_seed(0)
        logits = (scale * torch.randn(2, 3, 4, 5, generator=g)).to(device=device, dtype=dtype)
        labels = torch.randint(0, 3, (2, 4, 5), generator=g).to(device)
        return logits, labels

    def test_convention_focal_loss_alpha_weights_class_0_by_one_minus_alpha(self, device, dtype):
        # alpha_t is 1 - alpha for class 0 (background) and alpha for classes 1 .. C-1, as in Detectron's softmax
        # focal loss: a relabelling that keeps class 0 in place leaves the loss unchanged, one that moves class 0
        # changes it, and alpha=None is invariant under both
        logits, labels = self._logits_and_labels(device, dtype, scale=2.0)
        focal = kornia.losses.focal_loss
        target = labels[:, None]
        alpha_t = torch.where(target == 0, 0.75, 0.25).to(dtype)
        self.assert_close(
            focal(logits, labels, 0.25).gather(1, target), alpha_t * focal(logits, labels, None).gather(1, target)
        )
        for order, moves_class_0 in (([0, 2, 1], False), ([1, 2, 0], True)):
            perm = torch.tensor(order, device=device)  # class c becomes class perm[c]
            relabelled = (logits[:, perm.argsort()], perm[labels])
            self.assert_close(focal(*relabelled, None, reduction="sum"), focal(logits, labels, None, reduction="sum"))
            moved = focal(*relabelled, 0.25, reduction="sum")
            kept = focal(logits, labels, 0.25, reduction="sum")
            if moves_class_0:
                assert (moved - kept).abs() > 0.05 * kept.abs()
            else:
                self.assert_close(moved, kept)

    def test_convention_focal_loss_default_reduction_is_none_in_the_input_layout(self, device, dtype):
        # the default reduction is 'none': a (B, C, H, W) map whose target slice holds (1 - p_t)^gamma (-log p_t)
        logits, labels = self._logits_and_labels(device, dtype)
        out = kornia.losses.focal_loss(logits, labels, None)
        assert out.shape == (2, 3, 4, 5)
        log_p_t = logits.cpu().double().log_softmax(1).gather(1, labels.cpu()[:, None])
        expected = (1 - log_p_t.exp()) ** 2 * -log_p_t
        self.assert_close(out.gather(1, labels[:, None]), expected.to(device=device, dtype=dtype))
        transposed = kornia.losses.focal_loss(logits.transpose(-2, -1), labels.transpose(-2, -1), None)
        self.assert_close(transposed, out.transpose(-2, -1))

    def test_convention_focal_loss_mean_divides_by_the_class_axis_too(self, device, dtype):
        # 'mean' averages every element of the (B, C, *) output, so with gamma = 0 and alpha = None it is the per-pixel
        # cross entropy divided by C, where torch's cross_entropy 'mean' averages over the pixels only
        logits, labels = self._logits_and_labels(device, dtype)
        cross_entropy = -logits.cpu().double().log_softmax(1).gather(1, labels.cpu()[:, None])
        mean = kornia.losses.focal_loss(logits, labels, None, gamma=0.0, reduction="mean")
        self.assert_close(mean, (cross_entropy.mean() / 3).to(device=device, dtype=dtype))
        total = kornia.losses.focal_loss(logits, labels, None, gamma=0.0, reduction="sum")
        self.assert_close(total, cross_entropy.sum().to(device=device, dtype=dtype))

    def test_convention_focal_loss_does_not_validate_alpha_and_gamma(self, device, dtype):
        # alpha outside [0, 1] and a negative gamma are accepted and give a finite loss: kornia checks neither
        logits, labels = self._logits_and_labels(device, dtype, scale=0.5)
        for alpha, gamma in ((2.0, 2.0), (0.25, -1.0)):
            assert kornia.losses.focal_loss(logits, labels, alpha, gamma, "mean").isfinite()

    def test_convention_focal_loss_weight_is_not_normalised_in_the_mean(self, device, dtype):
        # weight (C,) scales each class's slice and 'mean' still divides by the element count, where torch's
        # cross_entropy(weight=) 'mean' divides by the summed weights of the target labels
        logits, labels = self._logits_and_labels(device, dtype)
        weight = torch.tensor([0.2, 1.0, 4.0], device=device, dtype=dtype)
        w_t = torch.tensor([0.2, 1.0, 4.0], dtype=torch.float64)[labels.cpu()[:, None]]
        weighted = -w_t * logits.cpu().double().log_softmax(1).gather(1, labels.cpu()[:, None])
        mean = kornia.losses.focal_loss(logits, labels, None, gamma=0.0, reduction="mean", weight=weight)
        self.assert_close(mean, (weighted.sum() / (3 * labels.numel())).to(device=device, dtype=dtype))
        assert (mean.cpu().double() * 3 - weighted.sum() / w_t.sum()).abs() > 0.05 * weighted.sum() / w_t.sum()

    def test_convention_focal_losses_ignored_labels_are_zero_weighted_in_the_mean(self, device, dtype):
        # A label equal to ignore_index (-100) contributes 0 but its elements stay in the 'mean' denominator; torch's
        # cross_entropy(ignore_index=) and binary_cross_entropy over the valid entries drop them from the mean instead
        logits, labels = self._logits_and_labels(device, dtype)
        labels[0, 0, :4] = -100  # 4 ignored pixels in image 0, 3 in image 1
        labels[1, 3, 2:] = -100
        ignored = labels.cpu() == -100
        safe = labels.cpu().clamp_min(0)[:, None]
        out = kornia.losses.focal_loss(logits, labels, None, gamma=0.0)
        assert (out.cpu().movedim(1, -1)[ignored] == 0).all()
        cross_entropy = -logits.cpu().double().log_softmax(1).gather(1, safe)[:, 0][~ignored]
        mean = kornia.losses.focal_loss(logits, labels, None, gamma=0.0, reduction="mean")
        self.assert_close(mean, (cross_entropy.sum() / out.numel()).to(device=device, dtype=dtype))
        assert (mean.cpu().double() * 3 - cross_entropy.mean()).abs() > 0.05 * cross_entropy.mean()
        # binary: a target entry equal to -100 is zero-weighted the same way
        target = (labels[:, None] > 0).to(dtype).expand(-1, 2, -1, -1).clone()
        target[:, 0][ignored.to(device)] = -100
        out = kornia.losses.binary_focal_loss_with_logits(logits[:, :2], target, None, gamma=0.0)
        assert (out[target == -100] == 0).all()
        target = target.cpu().double()
        bce = F.binary_cross_entropy_with_logits(logits[:, :2].cpu().double(), target, reduction="none")
        mean = kornia.losses.binary_focal_loss_with_logits(logits[:, :2], target.to(device, dtype), None, 0.0, "mean")
        self.assert_close(mean, (bce[target != -100].sum() / target.numel()).to(device=device, dtype=dtype))

    def test_convention_focal_loss_two_class_target_slice_is_binary_focal_loss(self, device, dtype):
        # With C = 2, logits [z0, z1] and class 1 the positive class, the target slice of focal_loss is
        # binary_focal_loss_with_logits(z1 - z0): alpha weights class 1 in both
        logits, labels = self._logits_and_labels(device, dtype, scale=2.0)
        labels = labels.clamp_max(1)
        multiclass = kornia.losses.focal_loss(logits[:, :2], labels, 0.25).gather(1, labels[:, None])
        binary = kornia.losses.binary_focal_loss_with_logits(logits[:, 1:2] - logits[:, :1], labels[:, None].to(dtype))
        self.assert_close(multiclass, binary)

    def test_convention_binary_focal_loss_matches_sigmoid_focal_loss_on_hard_targets(self, device, dtype):
        # alpha weights the positive term and 1 - alpha the negative one, and the function defaults to alpha = 0.25,
        # gamma = 2, reduction 'none'. Snippet used to generate expected (torchvision 0.29.0, torch 2.14.0, float64):
        #   torchvision.ops.sigmoid_focal_loss(z, t, alpha=0.25, gamma=2.0, reduction="none")
        z = torch.tensor(
            [[[[-2.0, -0.5, 0.0], [0.75, 1.5, 3.0]], [[2.5, -1.25, 0.25], [-3.0, 1.0, -0.75]]]],
            device=device,
            dtype=dtype,
        )
        t = torch.tensor(
            [[[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]], device=device, dtype=dtype
        )
        expected = torch.tensor(
            [
                [
                    [[0.41252, 0.0506801, 0.0433217], [0.393315, 0.00167571, 2.73208e-05]],
                    [[1.65185, 0.00937088, 0.0276004], [0.69157, 0.526401, 0.131105]],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(kornia.losses.binary_focal_loss_with_logits(z, t), expected)

    def test_convention_binary_focal_loss_fractional_target_mixes_the_two_terms(self, device, dtype):
        # A target t in (0, 1) weights the positive term by t and the negative term by 1 - t:
        #   t alpha (1 - p)^gamma (-log p) + (1 - t) (1 - alpha) p^gamma (-log(1 - p)).
        # torchvision's sigmoid_focal_loss applies alpha_t (1 - p_t)^gamma to the whole BCE instead; at t = 0.3 with the
        # defaults alpha = 0.25, gamma = 2 it gives 0.0611245, 0.1039721, 0.2952082 for z = -1, 0, 1.5 (torchvision
        # 0.29.0, torch 2.14.0, float64); the two agree at t in {0, 1}. Snippet used to generate those values:
        #   torchvision.ops.sigmoid_focal_loss(z, torch.full_like(z, 0.3), alpha=0.25, gamma=2.0, reduction="none")
        z = torch.tensor([[-1.0, 0.0, 1.5]], dtype=torch.float64)
        p = z.sigmoid()
        expected = 0.3 * 0.25 * (1 - p) ** 2 * -p.log() + 0.7 * 0.75 * p**2 * -(1 - p).log()
        actual = kornia.losses.binary_focal_loss_with_logits(
            z.to(device, dtype), torch.full_like(z, 0.3).to(device, dtype)
        )
        self.assert_close(actual, expected.to(device, dtype))
        assert (actual[0, 2].cpu().double() - 0.2952082).abs() > 0.1

    def test_convention_binary_focal_loss_pos_weight_runs_along_the_channel_axis(self, device, dtype):
        # pos_weight (C,) scales channel c's positive term: with gamma = 0 and alpha = None the loss is
        # binary_cross_entropy_with_logits(pos_weight=pos_weight.view(C, 1, 1)); torch broadcasts a (C,) pos_weight
        # along the last axis instead
        logits, labels = self._logits_and_labels(device, dtype)
        logits = logits[:, :2]
        target = torch.stack([labels == 1, labels == 2], 1).to(dtype)
        pos_weight = torch.tensor([3.0, 0.5], device=device, dtype=dtype)
        actual = kornia.losses.binary_focal_loss_with_logits(logits, target, None, 0.0, pos_weight=pos_weight)
        expected = F.binary_cross_entropy_with_logits(
            logits.cpu().double(),
            target.cpu().double(),
            pos_weight=torch.tensor([3.0, 0.5]).double().view(2, 1, 1),
            reduction="none",
        )
        self.assert_close(actual, expected.to(device=device, dtype=dtype))

    def test_convention_focal_loss_overflowing_non_target_log_probability_5628(self, device, dtype):
        # A logit gap beyond the range of the dtype overflows a non-target log-probability to -inf; its slice is still
        # 0 (#5628). The pixel is classified with probability 1, so its loss and its logits' gradient are 0, as for
        # F.cross_entropy, and the other pixel keeps the value it has on its own.
        big = 0.75 * torch.finfo(dtype).max
        logits = torch.tensor([[[[big, 1.0]], [[-big, -0.5]], [[0.0, 0.3]]]], device=device, dtype=dtype)
        logits.requires_grad_()
        labels = torch.zeros(1, 1, 2, device=device, dtype=torch.long)
        assert logits.log_softmax(1)[0, 1, 0, 0].isneginf()
        out = kornia.losses.focal_loss(logits, labels, None)
        self.assert_close(out[..., 0], torch.zeros_like(out[..., 0]), rtol=0, atol=0)
        rest = logits[..., 1:].detach().clone().requires_grad_()
        rest_total = kornia.losses.focal_loss(rest, labels[..., 1:], None, reduction="sum")
        self.assert_close(out[..., 1:], kornia.losses.focal_loss(rest, labels[..., 1:], None))
        total = kornia.losses.focal_loss(logits, labels, None, reduction="sum")
        self.assert_close(total, rest_total)
        (grad,) = torch.autograd.grad(total, logits)
        self.assert_close(grad[..., 0], torch.zeros_like(grad[..., 0]), rtol=0, atol=0)
        self.assert_close(grad[..., 1:], torch.autograd.grad(rest_total, rest)[0])
