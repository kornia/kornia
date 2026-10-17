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
import warnings
from typing import Optional

import pytest
import torch

from kornia.losses import FocalLoss, focal_loss

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

focal_module = importlib.import_module("kornia.losses.focal")

# 2 x 21 x 128 x 128 logits reach the CPU path's size threshold; the 21 classes are those of the flagship benchmark.
LARGE = (2, 21, 128, 128)


def _require_native_cpu(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
        pytest.skip("the CPU target log-probability path supports native CPU float32 and float64 only")


def _inputs(shape, device, dtype, ignore_index: Optional[int] = -100, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = (3 * torch.randn(shape, generator=generator)).to(device=device, dtype=dtype)
    labels = torch.randint(shape[1], (shape[0], *shape[2:]), generator=generator)
    if ignore_index is not None:
        labels[torch.rand(labels.shape, generator=generator) < 0.2] = ignore_index
    return logits, labels.to(device)


def _value_and_grad(logits, labels, *args, **kwargs):
    logits = logits.detach().clone().requires_grad_()
    loss = focal_loss(logits, labels, *args, **kwargs)
    (grad,) = torch.autograd.grad(loss, logits)
    return loss, grad


def _reference(monkeypatch, logits, labels, *args, **kwargs):
    """Value and gradient of the log_softmax path: masked log_softmax, then gather."""
    with monkeypatch.context() as patch:
        patch.setattr(focal_module, "_focal_loss_cpu_eligible", lambda pred: False)
        return _value_and_grad(logits, labels, *args, **kwargs)


def _count_calls(monkeypatch) -> list:
    calls = []
    apply = focal_module._TargetLogSoftmax.apply

    def counting(*args):
        calls.append(args[0].shape)
        return apply(*args)

    monkeypatch.setattr(focal_module._TargetLogSoftmax, "apply", counting)
    return calls


class TestFocalLossCpu(BaseTester):
    def assert_close_scaled(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        """``assert_close`` relative to the largest finite entry of ``expected``.

        A mean over a large batch makes gradients of about 1 / (B H W), far below the absolute tolerance, so both
        sides are divided by that entry first; the dtype tolerances then bound the error relative to it.
        """
        finite = expected[expected.isfinite()]
        scale = finite.abs().max() if finite.numel() else expected.new_ones(())
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        self.assert_close(actual / scale, expected / scale)

    def assert_close_with_nan(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        assert torch.equal(actual.isnan(), expected.isnan())
        self.assert_close_scaled(actual.masked_fill(actual.isnan(), 0), expected.masked_fill(expected.isnan(), 0))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_large_reduced_input_takes_the_cpu_path(self, monkeypatch, reduction, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(LARGE, device, dtype)
        _value_and_grad(logits, labels, 0.25, 2.0, reduction)
        assert calls == [LARGE]

    @pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
    def test_without_recorded_gradients_takes_the_cpu_path(self, monkeypatch, mode, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(LARGE, device, dtype)
        logits.requires_grad_()
        with torch.no_grad() if mode == "no_grad" else torch.inference_mode():
            focal_loss(logits, labels, 0.25, 2.0, "mean")
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reduction": "none"},
            {"reduction": "mean", "weight": "ones"},
            {"reduction": "mean", "alpha": "tensor"},
            {"reduction": "mean", "gamma": -1.0},
            {"reduction": "mean", "shape": (2, 21, 64, 64)},
        ],
        ids=["none", "weight", "tensor-alpha", "negative-gamma", "small"],
    )
    def test_other_inputs_keep_log_softmax(self, monkeypatch, kwargs, device, dtype):
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(kwargs.get("shape", LARGE), device, dtype)
        alpha = torch.tensor(0.25, device=device) if kwargs.get("alpha") == "tensor" else 0.25
        weight = torch.ones(logits.shape[1], device=device, dtype=dtype) if kwargs.get("weight") else None
        focal_loss(logits, labels, alpha, kwargs.get("gamma", 2.0), kwargs["reduction"], weight)
        assert calls == []

    def test_captures_and_transforms_keep_log_softmax(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(LARGE, device, dtype)
        torch.jit.trace(lambda x: focal_loss(x, labels, 0.25, 2.0, "mean"), (logits,), check_trace=False)
        torch.func.grad(lambda x: focal_loss(x, labels, 0.25, 2.0, "mean"))(logits)
        torch.vmap(lambda x: focal_loss(x[None], labels[:1], 0.25, 2.0, "mean"))(logits)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            focal_loss(logits, labels, 0.25, 2.0, "mean")
        assert calls == []

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("alpha", [None, 0.25])
    @pytest.mark.parametrize("ignore_index", [None, -100, 255])
    def test_matches_log_softmax(self, monkeypatch, reduction, gamma, alpha, ignore_index, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index)
        kwargs = {"ignore_index": ignore_index}
        expected, expected_grad = _reference(monkeypatch, logits, labels, alpha, gamma, reduction, **kwargs)
        actual, grad = _value_and_grad(logits, labels, alpha, gamma, reduction, **kwargs)
        self.assert_close(actual, expected)
        self.assert_close_scaled(grad, expected_grad)

    @pytest.mark.parametrize("shape", [(5, 4), (5, 4, 3, 5), (5, 3, 2, 3, 4), (0, 4, 3), (5, 4, 0)])
    def test_shapes_and_uneven_chunks_match_log_softmax(self, monkeypatch, shape, device, dtype):
        _require_native_cpu(device, dtype)
        # Two samples per chunk: five samples leave a last chunk of one.
        sample_bytes = max(1, torch.Size(shape[1:]).numel()) * torch.empty((), dtype=dtype).element_size()
        monkeypatch.setattr(focal_module, "_CPU_CHUNK_BYTES", 2 * sample_bytes)
        monkeypatch.setattr(focal_module, "_CPU_MIN_NUMEL", 0)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(shape, device, dtype)
        for reduction in ("mean", "sum"):
            expected, expected_grad = _reference(monkeypatch, logits, labels, 0.25, 2.0, reduction)
            actual, grad = _value_and_grad(logits, labels, 0.25, 2.0, reduction)
            self.assert_close_with_nan(actual, expected)
            self.assert_close_scaled(grad, expected_grad)
        assert len(calls) == 2

    @pytest.mark.parametrize("layout", ["transposed", "channels_last", "sliced"])
    def test_strided_logits_match_log_softmax(self, monkeypatch, layout, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        if layout == "transposed":
            logits, labels = logits.transpose(-1, -2), labels.transpose(-1, -2)
        elif layout == "channels_last":
            logits = logits.contiguous(memory_format=torch.channels_last)
        else:
            logits = torch.cat([logits, logits], dim=1)[:, ::2]
        expected, expected_grad = _reference(monkeypatch, logits, labels, 0.25, 2.0, "mean")
        actual, grad = _value_and_grad(logits, labels, 0.25, 2.0, "mean")
        self.assert_close(actual, expected)
        self.assert_close_scaled(grad, expected_grad)
        # Contiguous, as the log_softmax path's gather backward makes it.
        assert grad.stride() == expected_grad.stride()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("ignore_index", [None, -100])
    def test_non_finite_logits_match_log_softmax(self, monkeypatch, reduction, ignore_index, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index=None)
        big = 0.75 * torch.finfo(dtype).max
        # (pixel row, class, value): +inf, -inf and NaN logits, a pixel of -inf only, and the #5628 overflow of a
        # finite non-target logit gap, each on a pixel that is labelled and, with an ignore index, on one that is not.
        cases = [(0, 3, float("inf")), (1, 5, float("-inf")), (2, 7, float("nan")), (4, 0, big), (4, 1, -big)]
        for row, cls, value in cases:
            logits[0, cls, row, :2] = value
        logits[0, :, 3, :2] = float("-inf")
        if ignore_index is not None:
            labels[0, :, 1] = ignore_index
        for reduction_logits in (logits, logits.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)):
            expected, expected_grad = _reference(monkeypatch, reduction_logits, labels, 0.25, 2.0, reduction)
            actual, grad = _value_and_grad(reduction_logits, labels, 0.25, 2.0, reduction)
            self.assert_close_with_nan(actual, expected)
            self.assert_close_with_nan(grad, expected_grad)
            assert torch.equal(grad.isinf(), expected_grad.isinf())
            if ignore_index is not None:
                # A position that is not labelled gets an exact zero gradient, whatever its logits.
                assert (grad[0, :, :, 1] == 0).all()

    def test_ignored_non_finite_logits_have_zero_gradient_and_loss(self, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index=None)
        labels[1] = -100
        safe = logits.clone()
        logits[1, :, :, ::3] = float("inf")
        logits[1, :, :, 1::3] = float("nan")
        logits[1, :, :, 2::3] = float("-inf")
        expected, expected_grad = _value_and_grad(safe, labels, 0.25, 2.0, "sum")
        actual, grad = _value_and_grad(logits, labels, 0.25, 2.0, "sum")
        self.assert_close(actual, expected)
        assert (grad[1] == 0).all()
        self.assert_close_scaled(grad, expected_grad)

    def test_chunks_and_functional_form_are_bitwise_equal(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs((5, 21, 128, 128), device, dtype)
        expected, expected_grad = _value_and_grad(logits, labels, 0.25, 2.0, "mean")
        assert focal_module._batch_chunks(logits)[0][0].shape[0] < 5
        # One chunk for the whole batch, and the out-of-place form torch.compile and functorch trace.
        monkeypatch.setattr(focal_module, "_CPU_CHUNK_BYTES", 1 << 40)
        actual, grad = _value_and_grad(logits, labels, 0.25, 2.0, "mean")
        assert torch.equal(actual, expected)
        assert torch.equal(grad, expected_grad)
        monkeypatch.setattr(focal_module, "is_compiling", lambda: True)
        actual, grad = _value_and_grad(logits, labels, 0.25, 2.0, "mean")
        assert torch.equal(actual, expected)
        assert torch.equal(grad, expected_grad)

    def test_gradcheck(self, monkeypatch, device):
        _require_native_cpu(device, torch.float64)
        monkeypatch.setattr(focal_module, "_CPU_MIN_NUMEL", 0)
        monkeypatch.setattr(focal_module, "_CPU_CHUNK_BYTES", 1)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs((3, 4, 3, 2), device, torch.float64)
        for gamma in (0.5, 2.0):
            self.gradcheck(
                focal_loss, (logits, labels, 0.25, gamma, "mean"), dtypes=[torch.float64, torch.int64, None, None, None]
            )
        assert calls

    def test_gradgradcheck(self, monkeypatch, device):
        # A double backward differentiates the backward: it recomputes the softmax from the logits.
        _require_native_cpu(device, torch.float64)
        monkeypatch.setattr(focal_module, "_CPU_MIN_NUMEL", 0)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs((3, 4, 3, 2), device, torch.float64)
        logits.requires_grad_()
        assert torch.autograd.gradgradcheck(lambda x: focal_loss(x, labels, 0.25, 2.0, "sum"), (logits,))
        assert calls

    def test_second_derivatives_match_log_softmax(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        direction = torch.randn_like(logits)

        def hessian_vector_product():
            x = logits.detach().clone().requires_grad_()
            (grad,) = torch.autograd.grad(focal_loss(x, labels, 0.25, 2.0, "mean"), x, create_graph=True)
            return grad, torch.autograd.grad((grad * direction).sum(), x)[0]

        actual = hessian_vector_product()
        with monkeypatch.context() as patch:
            patch.setattr(focal_module, "_focal_loss_cpu_eligible", lambda pred: False)
            expected = hessian_vector_product()
        self.assert_close_scaled(actual[0], expected[0])
        self.assert_close_scaled(actual[1], expected[1])

    def test_forward_mode_ad_matches_log_softmax(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        tangent = torch.randn_like(logits)

        def jvp():
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(logits, tangent)
                return torch.autograd.forward_ad.unpack_dual(focal_loss(dual, labels, 0.25, 2.0, "mean"))

        calls = _count_calls(monkeypatch)
        actual = jvp()
        assert calls
        with monkeypatch.context() as patch:
            patch.setattr(focal_module, "_focal_loss_cpu_eligible", lambda pred: False)
            expected = jvp()
        self.assert_close(actual.primal, expected.primal)
        self.assert_close_scaled(actual.tangent, expected.tangent)

    def test_retain_graph_second_backward(self, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        logits.requires_grad_()
        loss = focal_loss(logits, labels, 0.25, 2.0, "mean")
        (first,) = torch.autograd.grad(loss, logits, retain_graph=True)
        (second,) = torch.autograd.grad(loss, logits)
        assert torch.equal(first, second)

    def test_module_and_backward_to_leaf(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        expected, expected_grad = _reference(monkeypatch, logits, labels, 0.25, 2.0, "mean")
        leaf = logits.clone().requires_grad_()
        loss = FocalLoss(0.25, 2.0, "mean")(leaf, labels)
        loss.backward()
        self.assert_close(loss.detach(), expected)
        self.assert_close_scaled(leaf.grad, expected_grad)

    def test_vmap_has_no_user_warning(self, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda x, y: focal_loss(x[None], y[None], 0.25, 2.0, "mean"))(logits, labels)
        assert not [w for w in recorded if issubclass(w.category, UserWarning)]
        expected = torch.stack([focal_loss(x[None], y[None], 0.25, 2.0, "mean") for x, y in zip(logits, labels)])
        self.assert_close(actual, expected)

    def test_scripted_loss_matches(self, device, dtype):
        # TorchScript compiles focal_loss without the custom function: its branch is a static `is_scripting` if.
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            scripted = torch.jit.script(focal_loss)
        self.assert_close(scripted(logits, labels, 0.25, 2.0, "mean"), focal_loss(logits, labels, 0.25, 2.0, "mean"))

    def test_dynamo_on_an_eligible_input(self, device, dtype, torch_optimizer):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        expected, expected_grad = _value_and_grad(logits, labels, 0.25, 2.0, "mean")
        compiled = torch_optimizer(focal_loss)
        x = logits.detach().clone().requires_grad_()
        actual = compiled(x, labels, 0.25, 2.0, "mean")
        (grad,) = torch.autograd.grad(actual, x)
        self.assert_close(actual, expected)
        self.assert_close_scaled(grad, expected_grad)


class _Exportable(torch.nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, labels, 0.25, 2.0, "mean")


@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
class TestFocalLossCpuExport(BaseTester):
    # torch.export keeps log_softmax: the CPU path's size check would become a guard of the exported program, which
    # then fails at run time for logits on the other side of the threshold although its dynamic shapes allow them.
    @pytest.mark.parametrize(
        "example, other", [((2, 21, 128, 128), (3, 21, 8, 12)), ((2, 21, 8, 8), (3, 21, 160, 170))]
    )
    def test_dynamic_sizes_run_on_both_sides_of_the_threshold(self, example, other, device, dtype):
        _require_native_cpu(device, dtype)
        dim = torch.export.Dim
        batch, height, width = dim("B", min=1, max=64), dim("H", min=2, max=4096), dim("W", min=2, max=4096)
        dynamic = {"logits": {0: batch, 2: height, 3: width}, "labels": {0: batch, 1: height, 2: width}}
        program = torch.export.export(_Exportable(), _inputs(example, device, dtype), dynamic_shapes=dynamic)
        assert any(node.target == torch.ops.aten.log_softmax.int for node in program.graph.nodes)
        exported = program.module()
        for shape in (example, other):
            logits, labels = _inputs(shape, device, dtype)
            self.assert_close(exported(logits, labels), focal_loss(logits, labels, 0.25, 2.0, "mean"))
