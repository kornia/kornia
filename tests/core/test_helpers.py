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

from kornia.core.exceptions import DeviceError
from kornia.core.utils import (
    _adjugate_closed_form,
    _extract_device_dtype,
    _torch_histc_cast,
    _torch_inverse_cast,
    _torch_solve_cast,
    _torch_svd_cast,
    is_exporting,
    register_module_state,
    safe_inverse_with_mask,
    safe_solve_with_mask,
)

from testing.base import assert_close


@pytest.mark.parametrize(
    "tensor_list,out_device,out_dtype,will_throw_error",
    [
        ([], torch.device("cpu"), torch.get_default_dtype(), False),
        ([None, None], torch.device("cpu"), torch.get_default_dtype(), False),
        ([torch.tensor(0, device="cpu", dtype=torch.float16), None], torch.device("cpu"), torch.float16, False),
        ([torch.tensor(0, device="cpu", dtype=torch.float32), None], torch.device("cpu"), torch.float32, False),
        ([torch.tensor(0, device="cpu", dtype=torch.float64), None], torch.device("cpu"), torch.float64, False),
        ([torch.tensor(0, device="cpu", dtype=torch.float16)] * 2, torch.device("cpu"), torch.float16, False),
        ([torch.tensor(0, device="cpu", dtype=torch.float32)] * 2, torch.device("cpu"), torch.float32, False),
        ([torch.tensor(0, device="cpu", dtype=torch.float64)] * 2, torch.device("cpu"), torch.float64, False),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float16), torch.tensor(0, device="cpu", dtype=torch.float64)],
            None,
            None,
            True,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float32), torch.tensor(0, device="cpu", dtype=torch.float64)],
            None,
            None,
            True,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float16), torch.tensor(0, device="cpu", dtype=torch.float32)],
            None,
            None,
            True,
        ),
    ],
)
def test_extract_device_dtype(tensor_list, out_device, out_dtype, will_throw_error):
    if will_throw_error:
        with pytest.raises(DeviceError):
            _extract_device_dtype(tensor_list)
    else:
        device, dtype = _extract_device_dtype(tensor_list)
        assert device == out_device
        assert dtype == out_dtype


class TestInverseCast:
    @pytest.mark.parametrize("input_shape", [(4, 4), (1, 3, 4, 4), (2, 4, 5, 5)])
    def test_smoke(self, device, dtype, input_shape):
        x = torch.rand(input_shape, device=device, dtype=dtype)
        y = _torch_inverse_cast(x)
        assert y.shape == x.shape

    def test_values(self, device, dtype):
        x = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device=device, dtype=dtype)

        y_expected = torch.tensor([[0.6, -0.7], [-0.2, 0.4]], device=device, dtype=dtype)

        y = _torch_inverse_cast(x)

        assert_close(y, y_expected)

    def test_jit(self, device, dtype):
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        op = _torch_inverse_cast
        op_jit = torch.jit.script(op)
        assert_close(op(x), op_jit(x))

    def test_not_invertible(self, device, dtype):
        with pytest.raises(RuntimeError):
            x = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype)
            _ = _torch_inverse_cast(x)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_closed_form_matches_linalg_inv(self, device, dtype, n):
        # The adjugate formula is what graph capture (trace / dynamo ONNX export) uses in place of
        # ``aten::linalg_inv``; it has to agree with the eager path on well-conditioned input.
        torch.manual_seed(0)
        x = torch.eye(n, device=device, dtype=dtype).expand(2, 3, n, n).clone()
        x.add_(torch.rand_like(x), alpha=0.5)
        adj, det = _adjugate_closed_form(x)
        assert adj.shape == x.shape
        assert det.shape == x.shape[:-2]
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        x_ref = x.to(torch.float32) if dtype in (torch.float16, torch.bfloat16) else x
        assert_close(adj / det[..., None, None], torch.linalg.inv(x_ref).to(dtype), atol=tol, rtol=tol)
        assert_close(det, torch.linalg.det(x_ref).to(dtype), atol=tol, rtol=tol)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_trace_has_no_linalg_inv(self, device, dtype, n):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("tracing under half precision is not a supported surface")
        x = torch.eye(n, device=device, dtype=dtype).expand(2, n, n).clone()
        x.add_(torch.rand_like(x), alpha=0.5)
        traced = torch.jit.trace(_torch_inverse_cast, x)
        assert "linalg_inv" not in str(traced.graph)
        assert_close(traced(x), _torch_inverse_cast(x))

    def test_closed_form_rejects_other_shapes(self, device, dtype):
        with pytest.raises(NotImplementedError):
            _adjugate_closed_form(torch.eye(5, device=device, dtype=dtype))


class TestExportHelpers:
    def test_is_exporting_eager(self):
        assert is_exporting() is False

    def test_is_exporting_scripted(self):
        # The guard is called from TorchScript-compiled functions (matching, calibration); it
        # must compile and evaluate to False there rather than being an unused stub that raises.
        assert torch.jit.script(is_exporting)() is False

    def test_is_exporting_falls_back_to_is_compiling(self, monkeypatch):
        # torch < 2.6 has no ``torch.compiler.is_exporting``; inside a Dynamo trace the guard must
        # still be true, as it is on newer torch where Dynamo folds the flag to True for
        # ``torch.compile`` as well.
        from kornia.core import utils

        monkeypatch.setattr(utils, "_torch_is_exporting", None)
        assert is_exporting() is False
        seen = []

        def fn(x):
            seen.append(is_exporting())
            return x + 1

        torch.compile(fn, backend="eager")(torch.zeros(2))
        assert seen == [True]

    def test_register_module_state_wraps_leaf(self, device, dtype):
        m = torch.nn.Module()
        x = torch.rand(3, device=device, dtype=dtype)
        register_module_state(m, "x", x)
        assert isinstance(m.x, torch.nn.Parameter)
        assert dict(m.named_parameters()).keys() == {"x"}
        p = torch.nn.Parameter(x.clone())
        register_module_state(m, "p", p)
        assert m.p is p

    def test_register_module_state_keeps_history(self, device, dtype):
        # A tensor with a grad_fn must not be re-rooted as a leaf, or gradients stop at it; it is
        # a buffer instead so ``.to()`` and ``state_dict()`` still reach it.
        m = torch.nn.Module()
        v = torch.rand(3, device=device, dtype=dtype, requires_grad=True)
        register_module_state(m, "y", v * 2)
        assert not isinstance(m.y, torch.nn.Parameter)
        assert dict(m.named_buffers()).keys() == {"y"}
        assert list(m.state_dict()) == ["y"]
        other = torch.float64 if dtype != torch.float64 else torch.float32
        moved = m.to(other)
        assert moved.y.dtype == other
        moved.y.sum().backward()
        assert_close(v.grad, torch.full_like(v, 2.0))


class TestHistcCast:
    def test_smoke(self, device, dtype):
        x = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        y_expected = torch.tensor([0.0, 2.0, 1.0, 0.0], device=device, dtype=dtype)

        y = _torch_histc_cast(x, bins=4, min=0, max=3)

        assert_close(y, y_expected)


class TestSvdCast:
    def test_smoke(self, device, dtype):
        a = torch.randn(5, 3, 3, device=device, dtype=dtype)
        u, s, v = _torch_svd_cast(a)

        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-3
        assert_close(a, u @ torch.diag_embed(s) @ v.transpose(-2, -1), atol=tol_val, rtol=tol_val)


class TestSolveCast:
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)
        # Exercise a reproducible, well-conditioned system instead of letting a random draw
        # decide whether the fixed residual bound is meaningful on a given backend.
        A = torch.eye(4).to(dtype).expand(2, 3, 1, 4, 4).clone()
        A.add_(torch.randn_like(A), alpha=0.05)
        B = torch.randn(2, 3, 1, 4, 6, dtype=dtype).to(device)
        A = A.to(device)

        X = _torch_solve_cast(A, B)
        error = torch.dist(B, A.matmul(X)) / B.norm().clamp_min(torch.finfo(dtype).eps)
        tol_val: float = max(1e-4, torch.finfo(dtype).eps)
        assert_close(error, torch.zeros_like(error), atol=tol_val, rtol=tol_val)


class TestSolveWithMask:
    def test_smoke(self, device, dtype):
        torch.manual_seed(0)  # issue kornia#2027
        A = torch.randn(2, 3, 1, 4, 4, device=device, dtype=dtype)
        B = torch.randn(2, 3, 1, 4, 6, device=device, dtype=dtype)

        X, _, mask = safe_solve_with_mask(B, A)
        X2 = _torch_solve_cast(A, B)
        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-4
        if mask.sum() > 0:
            assert_close(X[mask], X2[mask], atol=tol_val, rtol=tol_val)

    @pytest.mark.skipif(
        (int(torch.__version__.split(".")[0]) == 1) and (int(torch.__version__.split(".")[1]) < 10),
        reason="<1.10.0 not supporting",
    )
    def test_all_bad(self, device, dtype):
        A = torch.ones(10, 3, 3, device=device, dtype=dtype)
        B = torch.ones(10, 3, device=device, dtype=dtype)

        _X, _, mask = safe_solve_with_mask(B, A)
        assert torch.equal(mask, torch.zeros_like(mask))


class TestInverseWithMask:
    def test_smoke(self, device, dtype):
        x = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device=device, dtype=dtype)

        y_expected = torch.tensor([[0.6, -0.7], [-0.2, 0.4]], device=device, dtype=dtype)

        y, mask = safe_inverse_with_mask(x)

        assert_close(y, y_expected)
        assert torch.equal(mask, torch.ones_like(mask))

    def test_all_bad(self, device, dtype):
        A = torch.ones(10, 3, 3, device=device, dtype=dtype)
        _X, mask = safe_inverse_with_mask(A)
        assert torch.equal(mask, torch.zeros_like(mask))
