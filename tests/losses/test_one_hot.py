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
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

import kornia
from kornia.core.exceptions import BaseError
from kornia.losses.one_hot import _INDUCTOR_KEEPS_ASSERT_ASYNC, _check_class_labels, one_hot

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, assert_close, dynamo_is_available


class TestOneHot:
    def test_smoke(self, device, dtype):
        num_classes = 4
        labels = torch.zeros(2, 2, 1, dtype=torch.int64, device=device)
        labels[0, 0, 0] = 0
        labels[0, 1, 0] = 1
        labels[1, 0, 0] = 2
        labels[1, 1, 0] = 3

        # convert labels to one hot tensor
        one_hot_tensor = one_hot(labels, num_classes, device, dtype)

        assert_close(one_hot_tensor[0, labels[0, 0, 0], 0, 0].item(), 1.0)
        assert_close(one_hot_tensor[0, labels[0, 1, 0], 1, 0].item(), 1.0)
        assert_close(one_hot_tensor[1, labels[1, 0, 0], 0, 0].item(), 1.0)
        assert_close(one_hot_tensor[1, labels[1, 1, 0], 1, 0].item(), 1.0)

    def test_exact_values(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)
        expected = F.one_hot(labels, 3).movedim(-1, 1).to(dtype)

        one_hot_tensor = one_hot(labels, 3, device, dtype)

        assert one_hot_tensor.dtype == dtype
        assert_close(one_hot_tensor, expected, rtol=0, atol=0)
        assert_close(one_hot_tensor.sum(1), torch.ones_like(expected[:, 0]), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.int64, torch.uint8, torch.bool])
    def test_integer_dtypes(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)

        one_hot_tensor = one_hot(labels, 3, device, dtype)

        assert one_hot_tensor.dtype == dtype
        assert torch.equal(one_hot_tensor, F.one_hot(labels, 3).movedim(-1, 1).to(dtype))

    def test_eps_smoothing(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)
        exact = F.one_hot(labels, 3).movedim(-1, 1).to(dtype)

        one_hot_tensor = one_hot(labels, 3, device, dtype, eps=0.125)

        assert_close(one_hot_tensor, torch.where(exact == 1, 1.0, 0.125).to(dtype))

    def test_exception_eps_with_integer_dtype(self, device):
        labels = torch.tensor([[0, 1]], device=device)

        with pytest.raises(BaseError, match="floating point dtype"):
            one_hot(labels, 2, device, torch.int64, eps=1e-6)

    @pytest.mark.parametrize("eps", [0.0, 0.125])
    def test_jit_script(self, device, dtype, eps):
        # The replacement of ``kornia.utils.one_hot``, which scripted in 0.8.2, has to script too (#5901).
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)
        scripted = torch.jit.script(one_hot)
        for shape in [(2, 3), (2,), (1, 2, 3)]:
            lbl = labels.flatten()[: math.prod(shape)].reshape(shape)
            got, expected = scripted(lbl, 3, device, dtype, eps), one_hot(lbl, 3, device, dtype, eps)
            assert (got.dtype, got.shape) == (expected.dtype, expected.shape)
            assert torch.equal(got, expected)

    def test_jit_script_integer_dtype(self, device):
        labels = torch.tensor([[0, 2], [1, 0]], device=device)
        assert torch.equal(
            torch.jit.script(one_hot)(labels, 3, device, torch.uint8), one_hot(labels, 3, device, torch.uint8)
        )

    def test_jit_script_rejects_eps_with_an_integer_dtype(self, device):
        # the scripted path checks the dtype through a tensor, since TorchScript has no ``dtype.is_floating_point``
        labels = torch.tensor([[0, 2], [1, 0]], device=device)
        with pytest.raises(torch.jit.Error, match="A non-zero eps needs a floating point dtype"):
            torch.jit.script(one_hot)(labels, 3, device, torch.int64, 0.125)


# Run in a fresh interpreter: an index check that fails inside an Inductor CPU kernel aborts the whole process. The
# target-class focal loss of masked logits, as a custom log-softmax computes it: torch 2.14's Inductor fuses the gather
# into a parallel loop that runs before the label asserts, so unclamped labels terminated the process here.
_COMPILED_GATHER_BEFORE_THE_ASSERTS = """
import sys, warnings
warnings.filterwarnings("ignore")
import torch
from kornia.losses.one_hot import _check_class_labels
value = int(sys.argv[1])

def loss(pred, target):
    valid = (target != -100).unsqueeze(1)
    target = _check_class_labels(torch.where(target != -100, target, 0), pred.shape[1]).unsqueeze(1)
    x = torch.where(valid, pred, 0.0)
    x_max = x.amax(1, keepdim=True)
    log_prob = (x.gather(1, target) - x_max) - (x - x_max).exp().sum(1, keepdim=True).log()
    base = 1.0 - log_prob.exp()
    weight = torch.where(base == 0, base.pow(2.0), base.masked_fill(base == 0, 1.0).pow(2.0))
    term = -weight * log_prob.masked_fill(~valid, 0.0)
    term = torch.where(target == 0, term.new_full((), 0.75), term.new_full((), 0.25)) * term
    return term.mean() / pred.shape[1]

generator = torch.Generator().manual_seed(0)
pred = torch.randn(2, 21, 128, 128, generator=generator)
target = torch.randint(0, 21, (2, 128, 128), generator=generator)
target[0, 5, 7] = value
try:
    torch.compile(loss, fullgraph=True)(pred, target)
except RuntimeError as error:
    print("RuntimeError:", error)
else:
    print("no error")
"""


class TestCheckClassLabels(BaseTester):
    def test_valid_labels_pass_unchanged(self, device, dtype):
        labels = torch.tensor([[0, 2, 1], [2, 2, 0]], device=device)
        assert _check_class_labels(labels, 3) is labels
        empty = torch.zeros(0, 3, device=device, dtype=torch.int64)
        assert _check_class_labels(empty, 3) is empty

    def test_exception(self, device, dtype):
        if device.type != "cpu":
            pytest.skip("only CPU labels are checked")
        for value, message in ((3, "smaller than num_classes"), (-1, "non-negative")):
            labels = torch.tensor([[0, 2, 1], [2, value, 0]], device=device)
            with pytest.raises(RuntimeError, match=f"Class values must be {message}"):
                _check_class_labels(labels, 3)

    @pytest.mark.device_agnostic
    def test_other_devices_and_vmap_skip_the_check(self, device, dtype):
        # A meta tensor stands for any device but the CPU on every leg, and vmap has no batching rule for the assert.
        def asserts(labels):
            graph = make_fx(lambda x: _check_class_labels(x, 3))(labels).graph
            return sum(node.target is torch.ops.aten._assert_async.msg for node in graph.nodes)

        assert asserts(torch.zeros(2, 3, dtype=torch.int64)) == 2
        labels = torch.full((2, 3), 7, device="meta", dtype=torch.int64)
        assert asserts(labels) == 0
        assert _check_class_labels(labels, 3) is labels
        bad = torch.tensor([[0, 3], [1, 2]])
        assert torch.equal(torch.func.vmap(lambda x: _check_class_labels(x, 3))(bad), bad)

    @pytest.mark.device_agnostic
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_version_gate_matches_inductor(self, device, dtype):
        # Compiled labels are clamped only where Inductor keeps the asserts; up to torch 2.8 it decomposes them away.
        from torch._inductor.decomposition import decompositions

        assert _INDUCTOR_KEEPS_ASSERT_ASYNC == (torch.ops.aten._assert_async.msg not in decompositions)

    @pytest.mark.device_agnostic
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export_keeps_the_labels_unclamped(self, device, dtype):
        # ONNX translates an exported graph without its asserts, where clamped labels would turn the failing index op
        # into a silent wrong value. The exported program itself keeps the asserts.
        class _Gather(torch.nn.Module):
            def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                return x.gather(1, _check_class_labels(labels, x.shape[1]).unsqueeze(1))

        x = torch.rand(2, 3, 4)
        labels = torch.randint(3, (2, 4))
        exported = torch.export.export(_Gather(), (x, labels))
        targets = [str(node.target) for node in exported.graph.nodes]
        assert not any("clamp" in target for target in targets)
        assert sum("_assert_async" in target for target in targets) == 2
        labels[0, 1] = 3
        with pytest.raises(RuntimeError, match="Class values must be smaller than num_classes"):
            exported.module()(x, labels)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize("value, message", [(21, "smaller than num_classes"), (-3, "non-negative")])
    def test_dynamo_gather_fused_before_the_asserts_raises(self, value, message, device, dtype, optimizer_backend):
        if device.type != "cpu" or dtype != torch.float32 or optimizer_backend != "inductor":
            pytest.skip("one fresh-process inductor compile per case, on the CPU in float32")
        from torch._inductor.decomposition import decompositions

        if torch.ops.aten._assert_async.msg in decompositions:
            pytest.skip("this torch release's Inductor compiles torch._assert_async to nothing (before torch 2.9)")
        # The checkout under test, which "-c" imports first from the working directory.
        root = Path(kornia.__file__).resolve().parents[1]
        result = subprocess.run(  # noqa: S603 - this interpreter and a literal program; no shell or external input.
            [sys.executable, "-c", _COMPILED_GATHER_BEFORE_THE_ASSERTS, str(value)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.startswith("RuntimeError:"), result.stdout
        assert f"Class values must be {message}" in result.stdout


class TestConventionsOneHot(BaseTester):
    """Pins for the layout of :func:`kornia.losses.one_hot`."""

    def test_convention_one_hot_class_axis_is_dim_1(self, device, dtype):
        # int64 labels (N, *) -> (N, C, *): the class axis is dim 1 where F.one_hot puts it last, and with the default
        # eps=0 the result is F.one_hot(labels, C).movedim(-1, 1).to(device, dtype)
        labels = torch.tensor([[[0, 2, 1, 3, 3], [1, 0, 0, 2, 0], [3, 3, 1, 0, 2]]], device=device)
        labels = torch.cat([labels, labels.flip(-1)])  # (2, 3, 5)
        out = one_hot(labels, 4, device, dtype)
        assert out.shape == (2, 4, 3, 5)
        assert torch.equal(out, F.one_hot(labels, 4).movedim(-1, 1).to(dtype))
        assert torch.equal(one_hot(labels.transpose(1, 2), 4, device, dtype), out.transpose(2, 3))
        assert one_hot(labels[:, 0, 0], 4, device, dtype).shape == (2, 4)
        assert one_hot(labels.cpu(), 4, torch.device("meta"), dtype).device.type == "meta"
