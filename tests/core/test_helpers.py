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

from kornia.core.exceptions import DeviceError, TypeCheckError
from kornia.core.utils import (
    _adjugate_closed_form,
    _extract_device_dtype,
    _inverse_3x3_closed_form,
    _torch_histc_cast,
    _torch_inverse_cast,
    _torch_linalg_svdvals,
    _torch_solve_cast,
    _torch_svd_cast,
    is_autocast_enabled,
    is_exporting,
    is_mps_tensor_safe,
    register_module_state,
    safe_inverse_with_mask,
    safe_solve_with_mask,
)

from testing.base import BaseTester, assert_close


@pytest.mark.parametrize(
    "tensor_list,out_device,out_dtype,error",
    [
        ([], torch.device("cpu"), torch.get_default_dtype(), None),
        ([None, None], torch.device("cpu"), torch.get_default_dtype(), None),
        ([torch.tensor(0, device="cpu", dtype=torch.float16), None], torch.device("cpu"), torch.float16, None),
        ([torch.tensor(0, device="cpu", dtype=torch.float32), None], torch.device("cpu"), torch.float32, None),
        ([torch.tensor(0, device="cpu", dtype=torch.float64), None], torch.device("cpu"), torch.float64, None),
        ([torch.tensor(0, device="cpu", dtype=torch.float16)] * 2, torch.device("cpu"), torch.float16, None),
        ([torch.tensor(0, device="cpu", dtype=torch.float32)] * 2, torch.device("cpu"), torch.float32, None),
        ([torch.tensor(0, device="cpu", dtype=torch.float64)] * 2, torch.device("cpu"), torch.float64, None),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float16), torch.tensor(0, device="cpu", dtype=torch.float64)],
            None,
            None,
            TypeCheckError,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float32), torch.tensor(0, device="cpu", dtype=torch.float64)],
            None,
            None,
            TypeCheckError,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float16), torch.tensor(0, device="cpu", dtype=torch.float32)],
            None,
            None,
            TypeCheckError,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float32), torch.tensor(0, device="meta", dtype=torch.float32)],
            None,
            None,
            DeviceError,
        ),
        (
            [torch.tensor(0, device="cpu", dtype=torch.float32), torch.tensor(0, device="meta", dtype=torch.float64)],
            None,
            None,
            DeviceError,
        ),
        # A device mismatch anywhere in the list wins over a dtype mismatch, whichever comes first.
        (
            [
                torch.tensor(0, device="cpu", dtype=torch.float32),
                torch.tensor(0, device="cpu", dtype=torch.float64),
                torch.tensor(0, device="meta", dtype=torch.float32),
            ],
            None,
            None,
            DeviceError,
        ),
        (
            [
                torch.tensor(0, device="meta", dtype=torch.float32),
                torch.tensor(0, device="cpu", dtype=torch.float64),
                torch.tensor(0, device="cpu", dtype=torch.float32),
            ],
            None,
            None,
            DeviceError,
        ),
    ],
)
def test_extract_device_dtype(tensor_list, out_device, out_dtype, error):
    if error is not None:
        with pytest.raises(error) as excinfo:
            _extract_device_dtype(tensor_list)
        assert type(excinfo.value) is error
    else:
        device, dtype = _extract_device_dtype(tensor_list)
        assert device == out_device
        assert dtype == out_dtype


class TestExtractDeviceDtype(BaseTester):
    def test_dtype_only_mismatch_raises_type_check_error(self, device):
        # Same device, two dtypes: a dtype problem, reported as one (#5199).
        a = torch.zeros(1, device=device, dtype=torch.float32)
        b = torch.zeros(1, device=device, dtype=torch.float16)
        with pytest.raises(TypeError) as excinfo:
            _extract_device_dtype([a, None, b])
        err = excinfo.value
        assert type(err) is TypeCheckError
        assert err.expected_type == torch.float32
        assert err.actual_type == torch.float16
        assert "expected torch.float32, got torch.float16" in str(err)

    def test_device_mismatch_raises_device_error(self, device):
        # A device mismatch stays a DeviceError, also when the dtypes differ too, and also when a same-device pair
        # with two dtypes comes before it in the list.
        a = torch.zeros(1, device=device, dtype=torch.float32)
        other_dtype = torch.zeros(1, device=device, dtype=torch.float16)
        for tensors in (
            [a, torch.zeros(1, device="meta", dtype=torch.float32)],
            [a, torch.zeros(1, device="meta", dtype=torch.float16)],
            [a, other_dtype, torch.zeros(1, device="meta", dtype=torch.float32)],
        ):
            with pytest.raises(DeviceError) as excinfo:
                _extract_device_dtype(tensors)
            assert excinfo.value.actual_devices == [a.device, torch.device("meta")]
            assert excinfo.value.expected_device == a.device
            assert str(excinfo.value) == f"Passed tensors are not on the same device: expected {a.device}, got meta."


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
        x = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype)
        with pytest.raises(RuntimeError):
            _torch_inverse_cast(x)

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

        try:
            torch.compile(fn, backend="eager")(torch.zeros(2))
        except RuntimeError as e:  # e.g. "Dynamo is not supported on Python 3.13+" on torch 2.5
            pytest.skip(f"no Dynamo here: {e}")
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
        other = torch.float16 if dtype == torch.float32 else torch.float32  # float64 is unavailable on MPS
        moved = m.to(other)
        assert moved.y.dtype == other
        moved.y.sum().backward()
        assert_close(v.grad, torch.full_like(v, 2.0))


class TestHistcCast(BaseTester):
    def test_smoke(self, device, dtype):
        # The counts come back in the dtype they are computed in -- float32, or float64 for a float64 input -- and
        # not in the input's dtype: a count is not an image value and half precision cannot hold it (#5196).
        x = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        count_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        y_expected = torch.tensor([0.0, 2.0, 1.0, 0.0], device=device, dtype=count_dtype)

        y = _torch_histc_cast(x, bins=4, min=0, max=3)

        self.assert_close(y, y_expected)

    def test_counts_above_the_half_precision_integer_range_are_exact(self, device, dtype):
        # #5196: float16 holds integers exactly only up to 2048 and rounds them to inf from 65520, bfloat16 only up
        # to 256. With the counts cast back to the input dtype, 70000 equal values came back as inf in float16 and as
        # 70144 in bfloat16.
        x = torch.zeros(70001, device=device, dtype=dtype)
        x[0] = 1.0

        y = _torch_histc_cast(x, bins=2, min=0, max=1)

        assert y.tolist() == [70000.0, 1.0]
        assert y.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


class TestInverse3x3ClosedForm(BaseTester):
    @pytest.mark.parametrize("capture", [False, True], ids=["eager", "capture"])
    def test_half_input_is_inverted_in_float32(self, device, dtype, capture, monkeypatch):
        # #5197: inverted in float32 and rounded once to the half dtype, the result is within half an ulp-ish of the
        # float64 inverse of the rounded matrix (0.47 eps measured on 256 matrices, float16 and bfloat16); the
        # adjugate computed in the half dtype is 1.9 to 2.5 eps off. Pins the bfloat16 promotion as well, which no
        # other CI leg exercises (the cross kernel it avoids is missing only on MPS with torch 2.5.1).
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("the float32 promotion applies to half dtypes only")
        if capture:
            monkeypatch.setattr("kornia.core.utils._is_tracing_or_exporting", lambda: True)
        generator = torch.Generator().manual_seed(0)
        eye = torch.eye(3, dtype=torch.float64)
        matrix = torch.randn(256, 3, 3, generator=generator, dtype=torch.float64) + 3 * eye
        matrix = matrix.to(device=device, dtype=dtype)
        expected = torch.linalg.inv(matrix.cpu().double())

        inverse = _inverse_3x3_closed_form(matrix)

        assert inverse.dtype == dtype
        error = (inverse.cpu().double() - expected).abs() / expected.abs().amax((-2, -1), keepdim=True)
        assert error.max() <= torch.finfo(dtype).eps

    @pytest.mark.parametrize("capture", [False, True], ids=["eager", "capture"])
    @pytest.mark.parametrize("side", [3000, 11600])
    def test_pixel_normalization_matrix_of_a_large_image(self, device, dtype, side, capture, monkeypatch):
        # #5197: the pixel-normalization matrix of a `side`-pixel image, [[s, 0, -1], [0, s, -1], [0, 0, 1]] with
        # s = 2 / (side - 1), has determinant s**2: 4.4e-7 at 3000 px, which float16 holds only as a subnormal
        # (4.2e-7, 6 % off), and 3.0e-8 at 11600 px, which rounds to zero in float16 (half its smallest subnormal).
        # Inverted in float16 it lost digits and then turned inf/NaN; a half input is inverted in float32 and cast
        # back, as _torch_inverse_cast does.
        # Oracle: the exact inverse of the dtype-rounded matrix, [[r, 0, r], [0, r, r], [0, 0, 1]] with r = 1 / s, in
        # float64 (not torch.linalg.inv, whose float64 result on torch 2.5.1 is 2e-13 off at an exact zero). The
        # result is off it by the final rounding to `dtype` (half an ulp) plus the few roundings of the adjugate
        # formula. `capture` takes the scalar kernel that tracing and export use, which promotes the same way.
        if capture:
            monkeypatch.setattr("kornia.core.utils._is_tracing_or_exporting", lambda: True)
        s = 2.0 / (side - 1)
        matrix = torch.tensor([[[s, 0.0, -1.0], [0.0, s, -1.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
        matrix = matrix.to(device=device, dtype=dtype)

        inverse = _inverse_3x3_closed_form(matrix)

        assert inverse.dtype == dtype
        r = 1.0 / matrix[0, 0, 0].item()
        expected = torch.tensor([[[r, 0.0, r], [0.0, r, r], [0.0, 0.0, 1.0]]], dtype=torch.float64)
        eps = torch.finfo(dtype).eps
        self.assert_close(inverse.cpu().double(), expected, rtol=4 * eps, atol=0.0)


class TestSvdCast:
    def test_smoke(self, device, dtype):
        a = torch.randn(5, 3, 3, device=device, dtype=dtype)
        u, s, v = _torch_svd_cast(a)

        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-3
        assert_close(a, u @ torch.diag_embed(s) @ v.transpose(-2, -1), atol=tol_val, rtol=tol_val)

    def test_batch_above_mps_ceiling(self, device, dtype):
        # torch 2.14's MPS SVD raises at 8192 input elements or more (#4201), far below the hypothesis
        # count RANSAC's batched minimal solvers use; 1000 * 3 * 3 = 9000 elements clears it.
        torch.manual_seed(0)
        a = torch.randn(1000, 3, 3, device=device, dtype=dtype)
        u, s, v = _torch_svd_cast(a)

        assert u.device == a.device
        assert s.device == a.device
        assert v.device == a.device
        # Both half dtypes round three factors back before the reconstruction contracts them.
        tol_val: float = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert_close(a, u @ torch.diag_embed(s) @ v.transpose(-2, -1), atol=tol_val, rtol=tol_val)

    def test_svdvals_at_mps_ceiling(self, device, dtype, monkeypatch):
        # svdvals shares the 8192-element ceiling with svd (#4201), and feeds
        # solve_pnp_dlt. 512 4x4 matrices is exactly the boundary.
        torch.manual_seed(0)
        a = torch.randn(512, 4, 4, device=device, dtype=dtype)
        assert a.numel() == 8192

        seen = []
        real = torch.linalg.svdvals

        def spy(x, *args, **kwargs):
            seen.append(x.device.type)
            return real(x, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, "svdvals", spy)
        s = _torch_linalg_svdvals(a)

        assert s.device == a.device
        assert s.shape == (512, 4)
        if a.device.type == "mps":
            assert seen == ["cpu"], "an 8192-element MPS batch must take the CPU fallback"
        assert torch.isfinite(s).all()

    def test_batch_at_mps_ceiling(self, device, dtype, monkeypatch):
        # 8192 is the first failing size rather than the last working one (#4201), so a batch that
        # holds exactly 8192 elements has to take the CPU fallback too. 512 * 4 * 4 is the measured
        # boundary case. Spying on the decomposition itself keeps the check meaningful off MPS,
        # where it asserts the fallback stays out of the way.
        torch.manual_seed(0)
        a = torch.randn(512, 4, 4, device=device, dtype=dtype)
        assert a.numel() == 8192

        seen = []
        real_svd = torch.linalg.svd

        def spy_svd(x, *args, **kwargs):
            seen.append(x.device.type)
            return real_svd(x, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, "svd", spy_svd)
        u, s, v = _torch_svd_cast(a)

        expected = "cpu" if is_mps_tensor_safe(a) else a.device.type
        assert seen == [expected]
        assert u.device == a.device
        assert s.device == a.device
        assert v.device == a.device
        tol_val: float = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert_close(a, u @ torch.diag_embed(s) @ v.transpose(-2, -1), atol=tol_val, rtol=tol_val)

    def test_gradient_above_mps_ceiling(self, device, dtype):
        # Decomposing the batch elsewhere must not detach it from the graph.
        torch.manual_seed(0)
        a = torch.randn(1000, 3, 3, device=device, dtype=dtype, requires_grad=True)
        _, s, _ = _torch_svd_cast(a)

        s.sum().backward()
        assert a.grad is not None
        assert a.grad.device == a.device
        assert not a.grad.isnan().any()


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


def test_is_autocast_enabled_cpu():
    assert not is_autocast_enabled()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert is_autocast_enabled()

    assert not is_autocast_enabled()


def test_is_autocast_enabled_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    assert not is_autocast_enabled()

    with torch.autocast("mps", dtype=torch.float16):
        assert torch.is_autocast_enabled("mps")
        assert is_autocast_enabled()

    assert not is_autocast_enabled()


def test_is_autocast_enabled_xpu():
    # XPU autocast can be entered without an XPU device, and ``torch.is_autocast_enabled()`` without a device type
    # never reports it, so this case pins the per-device-type query on every CI leg (#5198).
    assert not is_autocast_enabled()

    with torch.autocast("xpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("xpu")
        assert is_autocast_enabled()

    assert not is_autocast_enabled()
