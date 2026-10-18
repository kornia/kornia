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

from kornia.core._compat import torch_version_ge
from kornia.losses import DiceLoss, dice_loss

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

dice_module = importlib.import_module("kornia.losses.dice")

# 2 x 21 x 128 x 128 logits are past the CPU path's size threshold; the 21 classes are those of the flagship benchmark.
LARGE = (2, 21, 128, 128)


def _require_native_cpu(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
        pytest.skip("the CPU one-hot-free Dice path supports native CPU float32 and float64 only")


def _inputs(shape, device, dtype, ignore_index: Optional[int] = -100, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = (3 * torch.randn(shape, generator=generator)).to(device=device, dtype=dtype)
    # Classes 0 .. C - 2 only: the last class is absent from the target, as macro averaging must notice.
    labels = torch.randint(max(1, shape[1] - 1), (shape[0], *shape[2:]), generator=generator)
    if ignore_index is not None:
        labels[torch.rand(labels.shape, generator=generator) < 0.2] = ignore_index
    return logits, labels.to(device)


def _value_and_grads(logits, labels, *args, weight=None, **kwargs):
    logits = logits.detach().clone().requires_grad_()
    inputs = [logits]
    if weight is not None:
        weight = weight.detach().clone().requires_grad_()
        inputs.append(weight)
    loss = dice_loss(logits, labels, *args, weight=weight, **kwargs)
    return (loss, *torch.autograd.grad(loss, inputs))


def _reference(monkeypatch, *args, **kwargs):
    """Value and gradients of the one-hot path: a one-hot target and full-size products."""
    with monkeypatch.context() as patch:
        patch.setattr(dice_module, "_dice_loss_cpu_eligible", lambda pred, weight: False)
        return _value_and_grads(*args, **kwargs)


def _count_calls(monkeypatch) -> list:
    calls = []
    cpu_loss = dice_module._dice_loss_cpu

    def counting(pred, *args):
        calls.append(pred.shape)
        return cpu_loss(pred, *args)

    monkeypatch.setattr(dice_module, "_dice_loss_cpu", counting)
    return calls


class TestDiceLossCpu(BaseTester):
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

    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_large_input_takes_the_cpu_path(self, monkeypatch, average, weighted, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(LARGE, device, dtype)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype) if weighted else None
        _value_and_grads(logits, labels, average, weight=weight)
        with torch.no_grad():
            dice_loss(logits.requires_grad_(), labels, average, weight=weight)
        assert calls == [LARGE, LARGE]

    @pytest.mark.parametrize(
        "case", ["small", "float32-pred-float64-weight", "half", "trace", "autocast"], ids=lambda case: case
    )
    def test_other_inputs_keep_the_one_hot_target(self, monkeypatch, case, device, dtype):
        calls = _count_calls(monkeypatch)
        if case == "half":
            dtype = torch.bfloat16
        logits, labels = _inputs((1, 21, 32, 32) if case == "small" else LARGE, device, dtype)
        weight = None
        if case == "float32-pred-float64-weight":
            logits, weight = logits.float(), torch.ones(LARGE[1], device=device, dtype=torch.float64)
        if case == "trace":
            torch.jit.trace(lambda x: dice_loss(x, labels), (logits,), check_trace=False)
        elif case == "autocast":
            with torch.autocast(device.type, dtype=torch.bfloat16):
                dice_loss(logits, labels)
        else:
            dice_loss(logits, labels, weight=weight)
        assert calls == []

    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("ignore_index", [None, -100, 255])
    @pytest.mark.parametrize("eps", [1e-8, 0.0])
    def test_matches_one_hot_target(self, monkeypatch, average, weighted, ignore_index, eps, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype) if weighted else None
        kwargs = {"weight": weight, "ignore_index": ignore_index}
        expected = _reference(monkeypatch, logits, labels, average, eps, **kwargs)
        actual = _value_and_grads(logits, labels, average, eps, **kwargs)
        for actual_value, expected_value in zip(actual, expected):
            self.assert_close_scaled(actual_value, expected_value)

    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_edge_samples_match_one_hot_target(self, monkeypatch, average, weighted, device, dtype):
        # Sample 0 is fully ignored, sample 1 holds class 0 only, whose weight is 0 when weighted; sample 2 is normal.
        _require_native_cpu(device, dtype)
        monkeypatch.setattr(dice_module, "_CPU_MIN_NUMEL", 0)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs((3, 4, 5, 6), device, dtype)
        labels[0] = -100
        labels[1] = 0
        weight = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype) if weighted else None
        for eps in (1e-8, 0.0):
            expected = _reference(monkeypatch, logits, labels, average, eps, weight=weight)
            actual = _value_and_grads(logits, labels, average, eps, weight=weight)
            for actual_value, expected_value in zip(actual, expected):
                self.assert_close_scaled(actual_value, expected_value)
        assert calls

    @pytest.mark.parametrize("shape", [(3, 1, 4, 5), (0, 4, 5, 6), (3, 4, 0, 6), (1, 2, 1, 1)])
    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_shapes_match_one_hot_target(self, monkeypatch, shape, average, device, dtype):
        _require_native_cpu(device, dtype)
        monkeypatch.setattr(dice_module, "_CPU_MIN_NUMEL", 0)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs(shape, device, dtype, ignore_index=None)
        expected = _reference(monkeypatch, logits, labels, average)
        actual = _value_and_grads(logits, labels, average)
        self.assert_close_with_nan(actual[0], expected[0])
        self.assert_close_scaled(actual[1], expected[1])
        assert calls

    @pytest.mark.parametrize("layout", ["transposed", "channels_last", "sliced"])
    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_strided_inputs_match_one_hot_target(self, monkeypatch, layout, average, weighted, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        if layout == "transposed":
            logits, labels = logits.transpose(-1, -2), labels.transpose(-1, -2)
        elif layout == "channels_last":
            logits = logits.contiguous(memory_format=torch.channels_last)
        else:
            logits = torch.cat([logits, logits], dim=1)[:, ::2]
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype) if weighted else None
        expected = _reference(monkeypatch, logits, labels, average, weight=weight)
        actual = _value_and_grads(logits, labels, average, weight=weight)
        for actual_value, expected_value in zip(actual, expected):
            self.assert_close_scaled(actual_value, expected_value)
        # The gradient keeps the one-hot path's layout.
        assert actual[1].stride() == expected[1].stride()

    @pytest.mark.parametrize("average", ["micro", "macro"])
    @pytest.mark.parametrize("ignore_index", [None, -100])
    def test_non_finite_logits_match_one_hot_target(self, monkeypatch, average, ignore_index, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index=None)
        # +inf, -inf and NaN logits on a labelled pixel of sample 0; with an ignore index, the same on an ignored one.
        logits[0, 3, 0, 0] = float("inf")
        logits[0, 5, 0, 1] = float("-inf")
        logits[0, 7, 0, 2] = float("nan")
        logits[0, :, 0, 3] = float("-inf")
        if ignore_index is not None:
            logits[0, 3, 1, 0] = float("inf")
            logits[0, 7, 1, 2] = float("nan")
            labels[0, 1] = ignore_index
        expected = _reference(monkeypatch, logits, labels, average, ignore_index=ignore_index)
        actual = _value_and_grads(logits, labels, average, ignore_index=ignore_index)
        self.assert_close_with_nan(actual[0], expected[0])
        self.assert_close_with_nan(actual[1], expected[1])
        # Sample 1 has finite logits: its gradient is unaffected.
        assert actual[1][1].isfinite().all()

    def test_fused_selection_is_bitwise_the_gather(self, device, dtype):
        # The one-hot comparison that torch.compile fuses selects the same values and gradients as the gather.
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index=None)
        logits[0, :, 0, :4] = float("nan")
        upstream = torch.randn(labels.shape, generator=torch.Generator().manual_seed(1)).to(device=device, dtype=dtype)
        results = []
        for fused in (False, True):
            x = logits.clone().requires_grad_()
            selected = dice_module._target_probability(x.softmax(1), labels, fused)
            results.append((selected, *torch.autograd.grad((selected * upstream).nansum(), x)))
        for actual_value, expected_value in zip(results[1], results[0]):
            assert torch.equal(actual_value.isnan(), expected_value.isnan())
            assert torch.equal(actual_value.nan_to_num(), expected_value.nan_to_num())

    @pytest.mark.parametrize("weighted", [False, True])
    def test_fused_micro_loss_is_bitwise_the_gathered_one(self, monkeypatch, weighted, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype) if weighted else None
        expected = _value_and_grads(logits, labels, "micro", weight=weight)
        monkeypatch.setattr(dice_module, "is_compiling", lambda: True)
        actual = _value_and_grads(logits, labels, "micro", weight=weight)
        for actual_value, expected_value in zip(actual, expected):
            assert torch.equal(actual_value, expected_value)

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_gradients_are_deterministic(self, average, device, dtype):
        # As on the one-hot path, repeated calls give the same bits, also for the weight's gradient.
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype)
        first = _value_and_grads(logits, labels, average, weight=weight)
        for _ in range(3):
            for actual_value, expected_value in zip(_value_and_grads(logits, labels, average, weight=weight), first):
                assert torch.equal(actual_value, expected_value)

    def test_invalid_labels_raise(self, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype, ignore_index=None)
        with pytest.raises(Exception, match=r"labels must be of dtype torch\.int64"):
            dice_loss(logits, labels.float())
        weight = torch.ones(LARGE[1], device=device, dtype=dtype)
        for value in (LARGE[1], -3):
            bad = labels.clone()
            bad[0, 0, 0] = value
            for average in ("micro", "macro"):
                for w in (None, weight):
                    with pytest.raises(RuntimeError):
                        dice_loss(logits, bad, average, weight=w)

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_gradcheck(self, monkeypatch, average, device):
        _require_native_cpu(device, torch.float64)
        monkeypatch.setattr(dice_module, "_CPU_MIN_NUMEL", 0)
        calls = _count_calls(monkeypatch)
        logits, labels = _inputs((2, 4, 3, 5), device, torch.float64)
        weight = torch.tensor([0.5, 1.0, 2.0, 1.5], device=device, dtype=torch.float64)
        self.gradcheck(
            lambda x, w: dice_loss(x, labels, average, weight=w), (logits, weight), dtypes=[torch.float64] * 2
        )
        logits.requires_grad_()
        assert torch.autograd.gradgradcheck(lambda x: dice_loss(x, labels, average), (logits,))
        assert calls

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_forward_mode_ad_matches_one_hot_target(self, monkeypatch, average, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        tangent = torch.randn_like(logits)

        def jvp():
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(logits, tangent)
                return torch.autograd.forward_ad.unpack_dual(dice_loss(dual, labels, average))

        calls = _count_calls(monkeypatch)
        actual = jvp()
        assert calls
        with monkeypatch.context() as patch:
            patch.setattr(dice_module, "_dice_loss_cpu_eligible", lambda pred, weight: False)
            expected = jvp()
        self.assert_close(actual.primal, expected.primal)
        self.assert_close_scaled(actual.tangent, expected.tangent)

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_vmap_has_no_user_warning(self, monkeypatch, average, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda x, y: dice_loss(x[None], y[None], average))(logits, labels)
        assert not [w for w in recorded if issubclass(w.category, UserWarning)]
        expected = torch.stack([dice_loss(x[None], y[None], average) for x, y in zip(logits, labels)])
        self.assert_close(actual, expected)

    def test_module_and_backward_to_leaf(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        logits, labels = _inputs(LARGE, device, dtype)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype)
        expected = _reference(monkeypatch, logits, labels, "macro", weight=weight)
        leaf = logits.clone().requires_grad_()
        loss = DiceLoss("macro", weight=weight)(leaf, labels)
        loss.backward()
        self.assert_close(loss.detach(), expected[0])
        self.assert_close_scaled(leaf.grad, expected[1])

    @pytest.mark.parametrize("average", ["micro", "macro"])
    def test_dynamo_on_an_eligible_input(self, average, device, dtype, torch_optimizer, optimizer_backend):
        _require_native_cpu(device, dtype)
        if optimizer_backend == "jit":
            pytest.skip("dice_loss is not scriptable: its average check builds a set")
        if dtype == torch.float64 and not torch_version_ge(2, 6):
            # On torch 2.5.1 a compiled call keeps the one-hot target (is_exporting() is is_compiling() there), and
            # Inductor cannot compile its float64 backward at this size: "fmadd" has no double-vector overload.
            pytest.skip("torch 2.5.1's Inductor fails on float64 dice_loss backward, with or without this path")
        logits, labels = _inputs(LARGE, device, dtype)
        weight = torch.linspace(0.5, 1.5, LARGE[1], device=device, dtype=dtype)
        expected = _value_and_grads(logits, labels, average, weight=weight)
        compiled = torch_optimizer(dice_loss)
        x = logits.detach().clone().requires_grad_()
        actual = compiled(x, labels, average, 1e-8, weight)
        (grad,) = torch.autograd.grad(actual, x)
        self.assert_close(actual, expected[0])
        self.assert_close_scaled(grad, expected[1])


class _Exportable(torch.nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return dice_loss(logits, labels, "macro")


@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
class TestDiceLossCpuExport(BaseTester):
    # torch.export keeps the one-hot target: the CPU path's size check would become a guard of the exported
    # program, which then fails at run time for logits on the other side of the threshold although its dynamic shapes
    # allow them.
    @pytest.mark.parametrize(
        "example, other", [((2, 21, 128, 128), (3, 21, 8, 12)), ((2, 21, 8, 8), (3, 21, 160, 170))]
    )
    def test_dynamic_sizes_run_on_both_sides_of_the_threshold(self, example, other, device, dtype):
        _require_native_cpu(device, dtype)
        dim = torch.export.Dim
        batch, height, width = dim("B", min=1, max=64), dim("H", min=2, max=4096), dim("W", min=2, max=4096)
        dynamic = {"logits": {0: batch, 2: height, 3: width}, "labels": {0: batch, 1: height, 2: width}}
        program = torch.export.export(_Exportable(), _inputs(example, device, dtype), dynamic_shapes=dynamic)
        assert any(node.target == torch.ops.aten.one_hot.default for node in program.graph.nodes)
        exported = program.module()
        for shape in (example, other):
            logits, labels = _inputs(shape, device, dtype)
            self.assert_close(exported(logits, labels), dice_loss(logits, labels, "macro"))
