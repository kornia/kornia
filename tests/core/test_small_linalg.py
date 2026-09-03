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

"""Tests for the closed-form small-matrix kernels in ``kornia.core._small_linalg``.

These are **raw kernels**: exact shape and a real floating dtype are caller preconditions,
not runtime checks, and behavior on anything else is unspecified. So nothing here asserts
that a kernel rejects an integer or complex tensor, and nothing pins what
``_adjugate_3x3`` does when handed a 4x4 (it silently uses the leading 3x3 block) or a 2x2
(it raises an incidental ``IndexError``). Those are recorded as known warts in the design
note, deliberately not as behavior. The size guard that *is* contractual belongs to the
dispatcher ``_adjugate_closed_form`` and is tested in ``test_helpers.py``.
"""

import io
import re

import pytest
import torch

from kornia.core._compat import torch_version_ge, torch_version_lt
from kornia.core._small_linalg import (
    _adjugate_2x2,
    _adjugate_3x3,
    _adjugate_4x4,
    _inverse_3x3_cross,
    _inverse_3x3_scalar,
)

from testing.base import BaseTester

ADJUGATE = {2: _adjugate_2x2, 3: _adjugate_3x3, 4: _adjugate_4x4}

# Op-type substrings that mean a linalg decomposition reached the graph. The point of these
# kernels is that an exported graph is basic arithmetic and nothing else, so this is what a
# regression would look like.
#
# Note what does the real work in the ONNX tests below: a MISSING lowering raises rather than
# emitting a node -- measured on torch 2.9.1, ``torch.linalg.inv`` gives
# ``UnsupportedOperatorError`` from the legacy exporter and ``ConversionError`` from the dynamo
# one -- so the export call succeeding is itself the gate. This check covers the other case: a
# future decomposition that exports *successfully* through something heavier than arithmetic.
# Matched as a substring rather than a prefix, because the dynamo exporter names a fallback node
# ``aten_linalg_inv``, which a ``startswith("linalg")`` test would miss.
# Long markers are unambiguous as bare substrings. The short ones are anchored to a name
# boundary, or they swallow basic arithmetic: "qr" matches ``Sqrt`` and "lu" matches the whole
# ``Relu``/``Selu``/``Elu``/``Celu``/``PRelu``/``LeakyRelu``/``ThresholdedRelu`` family -- all of
# them exactly the category this check exists to allow.
_LINALG_OP_RE = re.compile(
    r"(?:linalg|inverse|matmul|gemm|einsum)|(?:^|_)(?:det|lu|svd|qr|cholesky|eig)(?:$|_)", re.IGNORECASE
)


def _assert_basic_arithmetic_only(ops):
    offenders = sorted(op for op in ops if _LINALG_OP_RE.search(op))
    assert not offenders, f"linalg-shaped ops in the exported graph: {offenders} (full set: {sorted(ops)})"


def _export_kwargs(use_dynamo_exporter: bool) -> dict:
    """Export kwargs that are valid on every torch kornia declares support for.

    ``dynamo=`` does not exist before torch 2.4 -- measured: ``TypeError: export() got an
    unexpected keyword argument 'dynamo'`` on 2.0.1 and 2.2.2, present in 2.4.1. Passing it
    unconditionally would be a ``TypeError`` on an older pin rather than a skip. Gated at 2.5
    to match ``tests/filters/test_gaussian.py::test_onnx_export_legacy``, which is conservative
    by one release and costs nothing.
    """
    kwargs = {"opset_version": 18}
    if torch_version_ge(2, 5, 0):
        kwargs["dynamo"] = use_dynamo_exporter
    return kwargs


def _require_exporter(use_dynamo_exporter: bool) -> None:
    """Skip unless this build can actually run the requested ONNX exporter.

    ``dynamo=`` first exists in torch 2.5, but there it still routes through the experimental
    ``_compat.export_compat`` shim and needs ``onnxscript`` -- which the 2.5.1 CI legs do not
    install, so an unguarded call is a hard failure rather than a skip there. Same reasoning and
    same guard as ``tests/filters/test_gaussian.py::test_onnx_export_modern``.
    """
    pytest.importorskip("onnx")
    if use_dynamo_exporter:
        if torch_version_lt(2, 6, 0):
            pytest.skip("the dynamo ONNX exporter is only non-experimental from PyTorch 2.6")
        pytest.importorskip("onnxscript")


def _well_conditioned(n, device, dtype, batch=()):
    """A deterministic, well-conditioned matrix family.

    Built without touching the global RNG: a seeded ``torch.rand`` would still mutate it, and
    a fresh draw per run turns a tolerance assertion into a coin flip.

    Each batch entry gets a distinct scale. A bare ``expand`` would make every slice identical,
    and then a kernel that broadcast element 0 across the batch -- an indexing slip in the
    trailing-dim arithmetic -- would pass the batched cases. Scaling leaves the condition number
    untouched, so the tolerance argument above still holds.
    """
    ramp = torch.arange(1, n * n + 1, device=device, dtype=dtype).reshape(n, n) * 0.1
    x = (torch.eye(n, device=device, dtype=dtype) + ramp).expand(*batch, n, n).clone()
    if batch:
        idx = torch.arange(x.numel() // (n * n), device=device, dtype=dtype).reshape(*batch, 1, 1)
        x = x * (1 + idx * 0.05)
    return x


class TestAdjugateKernels(BaseTester):
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_smoke(self, device, dtype, n):
        adj, det = ADJUGATE[n](_well_conditioned(n, device, dtype))
        assert adj.shape == (n, n)
        assert det.shape == ()

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
    def test_cardinality(self, device, dtype, n, batch):
        adj, det = ADJUGATE[n](_well_conditioned(n, device, dtype, batch))
        assert adj.shape == (*batch, n, n)
        assert det.shape == batch

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_matches_linalg_inv_on_well_conditioned_input(self, device, dtype, n):
        # A smoke-test invariant on a controlled family, evaluated in a stated dtype -- not an
        # accuracy oracle. linalg.inv has no half kernel, so the reference is computed in
        # float32 and compared back.
        x = _well_conditioned(n, device, dtype, (2,))
        ref_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        adj, det = ADJUGATE[n](x)
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        self.assert_close(
            adj / det[..., None, None],
            torch.linalg.inv(x.to(ref_dtype)).to(dtype),
            atol=tol,
            rtol=tol,
        )
        self.assert_close(det, torch.linalg.det(x.to(ref_dtype)).to(dtype), atol=tol, rtol=tol)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_adjugate_identity_is_exact_on_integral_float64(self, device, n):
        # A @ adj(A) == det(A) * I holds identically. Small integers held in float64 make every
        # intermediate exactly representable, so this is an equality, not a tolerance -- and it
        # is the only check here that exercises the float64 path against something other than a
        # same-precision torch.linalg call, which would be no reference at all.
        if device.type == "mps":
            pytest.skip("MPS has no float64")
        values = torch.arange(1, n * n + 1, device=device, dtype=torch.float64).reshape(n, n)
        x = values + torch.eye(n, device=device, dtype=torch.float64) * (n * n)
        adj, det = ADJUGATE[n](x)
        expected = det * torch.eye(n, device=device, dtype=torch.float64)
        assert torch.equal(x @ adj, expected)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_gradcheck(self, device, n):
        x = _well_conditioned(n, device, torch.float64)
        self.gradcheck(lambda t: ADJUGATE[n](t)[0], (x,))
        self.gradcheck(lambda t: ADJUGATE[n](t)[1], (x,))

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_dynamo(self, device, dtype, n, torch_optimizer):
        x = _well_conditioned(n, device, dtype, (2,))
        op = ADJUGATE[n]
        op_optimized = torch_optimizer(op)
        expected_adj, expected_det = op(x)
        actual_adj, actual_det = op_optimized(x)
        self.assert_close(actual_adj, expected_adj)
        self.assert_close(actual_det, expected_det)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_scripts(self, device, dtype, n):
        x = _well_conditioned(n, device, dtype, (2,))
        scripted = torch.jit.script(ADJUGATE[n])
        expected_adj, expected_det = ADJUGATE[n](x)
        actual_adj, actual_det = scripted(x)
        self.assert_close(actual_adj, expected_adj)
        self.assert_close(actual_det, expected_det)

    @pytest.mark.device_agnostic
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("use_dynamo_exporter", [False, True], ids=["torchscript", "torchexport"])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_onnx_export_uses_basic_arithmetic(self, device, n, use_dynamo_exporter):
        # Capability matrix: the scalar kernels are the ONNX-safe path, and that is the whole
        # reason they exist. Assert the graph carries no linalg decomposition, on both exporters.
        _require_exporter(use_dynamo_exporter)
        onnx = pytest.importorskip("onnx")
        x = _well_conditioned(n, device, torch.float32, (2,))
        buffer = io.BytesIO()
        torch.onnx.export(_AdjugateModule(n), (x,), buffer, **_export_kwargs(use_dynamo_exporter))
        ops = {node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node}
        _assert_basic_arithmetic_only(ops)


class _AdjugateModule(torch.nn.Module):
    """Wrapper so the kernels can be handed to ``torch.onnx.export``."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj, det = ADJUGATE[self.n](x)
        return adj / det[..., None, None]


class TestInverse3x3Kernels(BaseTester):
    def test_smoke(self, device, dtype):
        x = _well_conditioned(3, device, dtype)
        assert _inverse_3x3_cross(x).shape == (3, 3)
        assert _inverse_3x3_scalar(x).shape == (3, 3)

    @pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
    def test_cardinality(self, device, dtype, batch):
        x = _well_conditioned(3, device, dtype, batch)
        assert _inverse_3x3_cross(x).shape == (*batch, 3, 3)
        assert _inverse_3x3_scalar(x).shape == (*batch, 3, 3)

    def test_cross_and_scalar_kernels_agree(self, device, dtype):
        # The mode split exists so that export avoids ``aten::linalg_inv``, not so that the two
        # paths compute different answers. They are different orderings of the same algebra, so
        # they agree to rounding rather than bit-for-bit.
        x = _well_conditioned(3, device, dtype, (2,))
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        self.assert_close(_inverse_3x3_cross(x), _inverse_3x3_scalar(x), atol=tol, rtol=tol)

    @pytest.mark.parametrize("kernel", [_inverse_3x3_cross, _inverse_3x3_scalar])
    def test_matches_linalg_inv_on_well_conditioned_input(self, device, dtype, kernel):
        x = _well_conditioned(3, device, dtype, (2,))
        ref_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        self.assert_close(kernel(x), torch.linalg.inv(x.to(ref_dtype)).to(dtype), atol=tol, rtol=tol)

    @pytest.mark.parametrize("kernel", [_inverse_3x3_cross, _inverse_3x3_scalar])
    def test_gradcheck(self, device, kernel):
        self.gradcheck(kernel, (_well_conditioned(3, device, torch.float64),))

    @pytest.mark.parametrize("kernel", [_inverse_3x3_cross, _inverse_3x3_scalar])
    def test_dynamo(self, device, dtype, kernel, torch_optimizer):
        x = _well_conditioned(3, device, dtype, (2,))
        self.assert_close(torch_optimizer(kernel)(x), kernel(x))

    @pytest.mark.parametrize("kernel", [_inverse_3x3_cross, _inverse_3x3_scalar])
    def test_scripts(self, device, dtype, kernel):
        x = _well_conditioned(3, device, dtype, (2,))
        self.assert_close(torch.jit.script(kernel)(x), kernel(x))

    @pytest.mark.device_agnostic
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("use_dynamo_exporter", [False, True], ids=["torchscript", "torchexport"])
    def test_scalar_kernel_exports_to_onnx(self, device, use_dynamo_exporter):
        # Contractual: the scalar kernel is the path the dispatcher takes under export, so its
        # graph must contain no linalg decomposition. Both exporters.
        _require_exporter(use_dynamo_exporter)
        onnx = pytest.importorskip("onnx")
        x = _well_conditioned(3, device, torch.float32, (2,))
        buffer = io.BytesIO()
        torch.onnx.export(_ScalarInverseModule(), (x,), buffer, **_export_kwargs(use_dynamo_exporter))
        ops = {node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node}
        _assert_basic_arithmetic_only(ops)

    @pytest.mark.device_agnostic
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("use_dynamo_exporter", [False, True], ids=["torchscript", "torchexport"])
    def test_cross_kernel_onnx_export_is_informational(self, device, use_dynamo_exporter):
        # NOT a gate, deliberately -- but the measurement behind it is now stronger than it was.
        # ``torch.linalg.cross`` lowers on every torch version CI runs: 2.14.0 and 2.9.1 on both
        # exporters, and 2.5.1 on the legacy one (2.5.1's dynamo path needs onnxscript, which
        # those CI legs do not install, so it is out of reach rather than broken). Nothing below
        # 2.5.1 is measured, and nothing below it is supported either.
        # It stays a skip rather than a hard assertion for two reasons: the dispatcher never takes
        # this kernel under export, so nothing in kornia depends on it lowering; and a future torch
        # that stops lowering ``cross`` should surface as a visible skip here rather than as a red
        # test for something kornia does not rely on.
        _require_exporter(use_dynamo_exporter)
        onnx = pytest.importorskip("onnx")
        x = _well_conditioned(3, device, torch.float32, (2,))
        buffer = io.BytesIO()
        try:
            torch.onnx.export(_CrossInverseModule(), (x,), buffer, **_export_kwargs(use_dynamo_exporter))
        except TypeError:
            # An API misuse is not a lowering failure. Re-raised rather than skipped, or a wrong
            # kwarg would masquerade forever as "cross does not lower here".
            raise
        except Exception as err:
            pytest.skip(f"torch.linalg.cross does not lower on torch {torch.__version__}: {err}")
        ops = {node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node}
        _assert_basic_arithmetic_only(ops)


class _ScalarInverseModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _inverse_3x3_scalar(x)


class _CrossInverseModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _inverse_3x3_cross(x)
