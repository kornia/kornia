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

from kornia.core.exceptions import BaseError, DeviceError, TypeCheckError
from kornia.core.utils import (
    _adjugate_closed_form,
    _closed_form_inverse,
    _det_perm_closed_form,
    _extract_device_dtype,
    _inverse_3x3_closed_form,
    _is_singular,
    _torch_det,
    _torch_histc_cast,
    _torch_inverse_cast,
    _torch_linalg_lu_factor_ex,
    _torch_linalg_lu_solve,
    _torch_linalg_qr,
    _torch_linalg_solve_ex,
    _torch_linalg_svdvals,
    _torch_lu_unpack,
    _torch_solve_cast,
    _torch_svd_cast,
    batched_forward,
    is_autocast_enabled,
    is_exporting,
    is_mps_tensor_safe,
    register_module_state,
    safe_inverse_with_mask,
    safe_solve_with_mask,
)

from testing.base import BaseTester, assert_close


def _issue_5476_batch(device, dtype):
    """Identity, a 3x3 with equal first and last rows, one with a doubled row, zeros and a translation homography."""
    repeated = torch.tensor([[-956.0, -761.0, -67.0], [760.0, -37.0, -621.0], [-956.0, -761.0, -67.0]])
    doubled = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])
    translation = torch.tensor([[1.0, 0.0, 4096.0], [0.0, 1.0, -2048.0], [0.0, 0.0, 1.0]])
    return torch.stack([torch.eye(3), repeated, doubled, torch.zeros(3, 3), translation]).to(device, dtype)


def _scales(dtype):
    """Powers of two keep the scaling exact, the decimal ones, kept out of half precision, perturb it by a rounding."""
    if dtype in (torch.float16, torch.bfloat16):
        return [2.0**-10, 1.0, 2.0**10]
    return [1e-3, 1.0, 1e3, 2.0**-20, 2.0**20]


def _regular_at_extreme_scales(device, dtype):
    """Well-conditioned 4x4 matrices scaled so that the products of four entries overflow or underflow the dtype
    while the inverse fits: the identity and a dense matrix at ``big`` and ``1 / big``, the dense matrix with two
    columns, then two rows, scaled by ``small``, the identity at the largest finite value of the dtype, where ``log2``
    rounds up to the exponent above, and a block matrix whose column factor ``tiny / huge`` and the row factor ``low``
    of its other block add up to an exponent beyond the dtype, by which the zeros of its inverse are scaled back,
    while no entry of its row- or column-scaled copy is subnormal (MPS flushes those to zero). Powers of two keep the
    scaling exact."""
    big = 2.0**40 if dtype == torch.float32 else 2.0**300
    small = 2.0**-80 if dtype == torch.float32 else 2.0**-600
    huge, tiny, low = (2.0**80, 2.0**-40, 2.0**-20) if dtype == torch.float32 else (2.0**800, 2.0**-200, 2.0**-100)
    A = torch.tensor(
        [[4.0, 1.0, 2.0, 1.0], [1.0, 5.0, 1.0, 2.0], [2.0, 1.0, 6.0, 1.0], [1.0, 2.0, 1.0, 7.0]],
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(4, device=device, dtype=dtype)
    D = torch.diag(torch.tensor([1.0, 1.0, small, small], device=device, dtype=dtype))
    compound = eye * low
    compound[:2, :2] = torch.tensor([[huge, tiny], [huge, -tiny]], device=device, dtype=dtype)
    return torch.stack([eye * big, eye / big, A * big, A / big, A @ D, D @ A, eye * torch.finfo(dtype).max, compound])


def _regular_with_a_vanishing_column(n, device, dtype):
    """``[[2 ** 100, 2 ** -60], [2 ** 100, -2 ** -60]]`` (``2 ** 1000`` and ``2 ** -600`` in float64) in the corner of
    the identity of order ``n``, and its inverse, ``[[1 / (2 huge), 1 / (2 huge)], [1 / (2 tiny), -1 / (2 tiny)]]`` in
    the corner. Regular, with a finite inverse and a finite ``adj / det``, while the second column is ``2 ** -160`` of
    the first in both rows and would underflow in a row scaling that is carried out. ``torch.linalg.inv`` of torch
    2.5.1 returns ``[[1 / huge, 0], [1 / tiny, -1 / tiny]]`` for it, off by 1 in ``A @ inv``; torch 2.14 is exact."""
    huge, tiny = (2.0**100, 2.0**-60) if dtype == torch.float32 else (2.0**1000, 2.0**-600)
    A = torch.eye(n, device=device, dtype=dtype)
    A[:2, :2] = torch.tensor([[huge, tiny], [huge, -tiny]], device=device, dtype=dtype)
    inverse = torch.eye(n, device=device, dtype=dtype)
    inverse[:2, :2] = torch.tensor([[0.5 / huge, 0.5 / huge], [0.5 / tiny, -0.5 / tiny]], device=device, dtype=dtype)
    return A, inverse


# One row is the sum of two others, or a multiple of the other, in small integers: exactly singular, and every
# product of the closed-form determinant is exact.
_DEPENDENT_ROWS = {
    2: [[3.0, -7.0], [-6.0, 14.0]],
    3: [[2.0, -3.0, 5.0], [7.0, 1.0, -4.0], [9.0, -2.0, 1.0]],
    4: [[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0], [-9.0, 1.0, 2.0, -3.0], [-3.0, 5.0, -2.0, 9.0]],
}


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

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_scaled_closed_form_is_adj_over_det_bit_for_bit(self, device, dtype, n):
        # The closed-form inverse balances rows and columns by powers of two before the adjugate (#5507). A power
        # of two changes no rounding step, so wherever the adjugate and the determinant are normal numbers the
        # result is the same to the bit. The last matrix has a second column that is ``2 ** -160`` of the first in
        # every row: a row scaling that is carried out would flush it, the exponent arithmetic does not.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("the capture path computes in float32 or float64")
        torch.manual_seed(0)
        x = torch.randn(64, n, n, device=device, dtype=dtype)
        x = x * torch.logspace(-3, 3, 64, dtype=dtype).to(device)[:, None, None]  # ``logspace`` has no MPS kernel
        x[::2] *= torch.logspace(-2, 2, n, dtype=dtype).to(device)[None, :, None]
        x = torch.cat([x, _regular_with_a_vanishing_column(n, device, dtype)[0][None]])
        adj, det = _adjugate_closed_form(x)
        assert torch.isfinite(adj).all()
        assert torch.isfinite(det).all()
        assert torch.equal(_closed_form_inverse(x), adj / det[..., None, None])

    def test_trace_keeps_the_inverse_finite_at_an_extreme_scale_5507(self, device, dtype):
        # Under capture ``1e13 * I`` of order 4 had ``adj inf`` and ``det inf`` in float32 and came back as NaN,
        # while eager returned ``1e-13 * I``. The scaled adjugate keeps the traced inverse finite and eager-close.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("tracing under half precision is not a supported surface")
        # The last matrix is regular with a column ``2 ** -160`` of the other in every row; ``_is_singular`` reads it
        # as singular in eager mode and under capture alike, so it is tested on this mask-free path. It is read
        # against its exact inverse, which the powers of two reach to the bit, not against eager, whose LU loses the
        # column on torch 2.5.1.
        vanishing, vanishing_inverse = _regular_with_a_vanishing_column(4, device, dtype)
        A = torch.cat([_regular_at_extreme_scales(device, dtype), vanishing[None]])
        traced = torch.jit.trace(_torch_inverse_cast, A, check_trace=False)
        inverse = traced(A)
        assert torch.isfinite(inverse).all()
        rtol = 1e-5 if dtype == torch.float32 else 1e-12
        assert_close(inverse[:-1], _torch_inverse_cast(A[:-1]), atol=0.0, rtol=rtol)
        assert torch.equal(inverse[-1], vanishing_inverse)

    @pytest.mark.parametrize("n", [3, 4])
    def test_trace_scales_the_inverse_back_beyond_the_largest_power_of_two_5507(self, device, dtype, n):
        # The entry (1, 2) of the inverse of this matrix is ``-2 ** 45`` in float32, but its row factor ``2 ** -20``
        # and the column factor ``2 ** -126`` of the ``2 ** -26`` entries scale it back by ``2 ** 146``, beyond the
        # largest power of two of the dtype (``2 ** 1122`` in float64). One power of two capped at that largest
        # one returns ``-2 ** 26``. Every entry and the exact inverse are normal powers of two.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("tracing under half precision is not a supported surface")
        h, low, r = (100, 26, 20) if dtype == torch.float32 else (800, 222, 100)
        A = torch.eye(n, device=device, dtype=dtype)
        A[:3, :3] = torch.tensor(
            [[2.0**h, 2.0**-low, 1.0], [2.0**h, -(2.0**-low), 0.0], [0.0, 0.0, 2.0**-r]], device=device, dtype=dtype
        )
        expected = torch.eye(n, device=device, dtype=dtype)
        expected[:3, :3] = torch.tensor(
            [
                [2.0 ** -(h + 1), 2.0 ** -(h + 1), -(2.0 ** (r - h - 1))],
                [2.0 ** (low - 1), -(2.0 ** (low - 1)), -(2.0 ** (low + r - 1))],
                [0.0, 0.0, 2.0**r],
            ],
            device=device,
            dtype=dtype,
        )
        traced = torch.jit.trace(_torch_inverse_cast, A[None], check_trace=False)
        assert torch.equal(traced(A[None])[0], expected)


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
    def test_rejects_non_tensor_A(self):
        # A was unchecked while its sibling safe_inverse_with_mask checks A (#5201).
        with pytest.raises(TypeCheckError, match=r"A must be torch.Tensor"):
            _torch_solve_cast([[1.0]], torch.ones(2, 2))

    def test_rejects_non_tensor_B(self):
        with pytest.raises(TypeCheckError, match=r"B must be torch.Tensor"):
            _torch_solve_cast(torch.eye(2), [[1.0]])

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
    def test_rejects_non_tensor_A(self):
        with pytest.raises(TypeCheckError, match=r"A must be torch.Tensor"):
            safe_solve_with_mask(torch.ones(2, 3), [[1.0]])

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

    @pytest.mark.parametrize("masking", ["mul", "where"])
    def test_singular_matrix_keeps_shared_gradient_finite(self, masking, device, dtype):
        # kornia#5194: one singular matrix in the batch made the gradient of a parameter shared by all
        # of them NaN, however the caller masked the output.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the reference gradient needs full precision")
        M = torch.stack([torch.eye(3), torch.zeros(3, 3), 2 * torch.eye(3)]).to(device, dtype)
        B = torch.ones(3, 3, device=device, dtype=dtype)

        def grad(M: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            W = torch.eye(3, device=device, dtype=dtype) + 0.1 * torch.arange(9.0, device=device, dtype=dtype).view(
                3, 3
            )
            W.requires_grad_()
            X, _, mask = safe_solve_with_mask(B, W @ M)
            m = mask[:, None, None]
            loss = (X * m).sum() if masking == "mul" else torch.where(m, X, torch.zeros_like(X)).sum()
            loss.backward()
            return W.grad

        _, _, mask = safe_solve_with_mask(B, M)
        assert mask.tolist() == [True, False, True]
        # The singular system contributes nothing, so the gradient is the one of the valid systems alone.
        assert_close(grad(M, B), grad(M[[0, 2]], B[[0, 2]]))

    def test_singular_matrix_solves_identity(self, device, dtype):
        A = torch.stack([2 * torch.eye(3), torch.ones(3, 3)]).to(device, dtype)
        B = torch.arange(6.0, device=device, dtype=dtype).view(2, 3)
        X, _, mask = safe_solve_with_mask(B, A)
        assert mask.tolist() == [True, False]
        assert_close(X[0, :, 0], B[0] / 2)
        assert torch.equal(X[1, :, 0], B[1])

    def test_overflowing_solution_is_invalid(self, device):
        # The solution of (1e-5 * I) x = 1 is 1e5, which float16 cannot hold although the float32 solve can.
        A = torch.eye(2, device=device, dtype=torch.float16)[None] * 1e-5
        B = torch.ones(1, 2, 1, device=device, dtype=torch.float16)
        X, _, mask = safe_solve_with_mask(B, A)
        assert mask.tolist() == [False]
        assert torch.equal(X, B)

    def test_singular_verdict_is_the_one_of_safe_inverse_with_mask_5476(self, device, dtype):
        # The 3x3 with two equal rows has a float32 LU pivot that is zero on some torch versions and tiny on
        # others; the verdict is kornia's rule, the same as the inverse's, on every one of them.
        A = _issue_5476_batch(device, dtype)
        B = torch.ones(5, 3, device=device, dtype=dtype)
        X, _, mask = safe_solve_with_mask(B, A)
        assert mask.tolist() == [True, False, False, False, True]
        assert torch.equal(mask, safe_inverse_with_mask(A)[1])
        assert_close(X[0, :, 0], B[0])
        assert_close(X[4, :, 0], torch.tensor([1.0 - 4096.0, 1.0 + 2048.0, 1.0], device=device, dtype=dtype))

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_dependent_rows_are_singular_at_every_scale(self, device, dtype, n):
        A = torch.tensor(_DEPENDENT_ROWS[n], device=device, dtype=dtype)
        scales = _scales(dtype)
        A = A * torch.tensor(scales, device=device, dtype=dtype)[:, None, None]
        B = torch.ones(len(scales), n, device=device, dtype=dtype)
        _, _, mask = safe_solve_with_mask(B, A)
        assert not mask.any()


class TestBatchedForwardBatchSize:
    def test_rejects_zero_batch_size(self):
        # batch_size=0 divided by zero inside the micro-batch loop (#5201).
        model = torch.nn.Linear(2, 3)
        data = torch.rand(5, 2)
        with pytest.raises(BaseError, match="batch_size must be positive, got 0"):
            batched_forward(model, data, torch.device("cpu"), batch_size=0)

    def test_rejects_negative_batch_size(self):
        model = torch.nn.Linear(2, 3)
        data = torch.rand(5, 2)
        with pytest.raises(BaseError, match="batch_size must be positive, got -1"):
            batched_forward(model, data, torch.device("cpu"), batch_size=-1)

    def test_valid_batch_sizes_keep_the_output(self):
        model = torch.nn.Linear(2, 3)
        data = torch.rand(5, 2)
        expected = model(data)
        # 1 is the smallest valid size; 5 == len(data) takes the single-call path
        for bs in (1, 2, 5, 128):
            assert_close(batched_forward(model, data, torch.device("cpu"), batch_size=bs), expected)


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

    @pytest.mark.parametrize("masking", ["mul", "where"])
    def test_singular_matrix_keeps_shared_gradient_finite(self, masking, device, dtype):
        # kornia#5194: one singular matrix in the batch made the gradient of a parameter shared by all
        # of them NaN, however the caller masked the output.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the reference gradient needs full precision")
        M = torch.stack([torch.eye(3), torch.zeros(3, 3), 2 * torch.eye(3)]).to(device, dtype)

        def grad(M: torch.Tensor) -> torch.Tensor:
            W = torch.eye(3, device=device, dtype=dtype) + 0.1 * torch.arange(9.0, device=device, dtype=dtype).view(
                3, 3
            )
            W.requires_grad_()
            inverse, mask = safe_inverse_with_mask(W @ M)
            m = mask[:, None, None]
            loss = (inverse * m).sum() if masking == "mul" else torch.where(m, inverse, torch.zeros_like(inverse)).sum()
            loss.backward()
            return W.grad

        _, mask = safe_inverse_with_mask(M)
        assert mask.tolist() == [True, False, True]
        # The singular matrix contributes nothing, so the gradient is the one of the valid matrices alone.
        assert_close(grad(M), grad(M[[0, 2]]))

    def test_singular_matrix_is_identity_in_eager_and_trace(self, device, dtype):
        A = torch.stack([2 * torch.eye(3), torch.zeros(3, 3)]).to(device, dtype)
        expected = torch.stack([0.5 * torch.eye(3), torch.eye(3)]).to(device, dtype)
        for fn in (safe_inverse_with_mask, torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)):
            inverse, mask = fn(A)
            assert mask.tolist() == [True, False]
            assert_close(inverse, expected)

    def test_overflowing_inverse_is_invalid(self, device):
        # The inverse of 1e-5 * I is 1e5 * I, which float16 cannot hold, in eager mode and under capture.
        A = torch.eye(2, device=device, dtype=torch.float16) * 1e-5
        for fn in (safe_inverse_with_mask, torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)):
            inverse, mask = fn(A)
            assert not mask.item()
            assert torch.equal(inverse, torch.eye(2, device=device, dtype=torch.float16))

    def test_singular_verdict_is_the_same_in_eager_and_trace_5476(self, device, dtype):
        # The 3x3 with two equal rows has a closed-form float32 determinant of -20 against a permanent of
        # 1e9, and a float32 LU pivot that is zero on some torch versions and tiny on others. The verdict
        # is kornia's rule in both modes, so it is the same one everywhere.
        A = _issue_5476_batch(device, dtype)
        expected = [True, False, False, False, True]
        for fn in (safe_inverse_with_mask, torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)):
            inverse, mask = fn(A)
            assert mask.tolist() == expected
            assert_close(inverse[0], A[0])
            assert torch.equal(inverse[1:4], A[0].expand(3, 3, 3))

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_dependent_rows_are_singular_at_every_scale(self, device, dtype, n):
        # Scaling a matrix scales its determinant and the permanent of its absolute values alike, so the
        # verdict does not depend on the units; the inexact decimal scales also perturb the exact
        # dependence by a rounding error, which the rule has to absorb.
        A = torch.tensor(_DEPENDENT_ROWS[n], device=device, dtype=dtype)
        scales = _scales(dtype)
        A = A * torch.tensor(scales, device=device, dtype=dtype)[:, None, None]
        for fn in (safe_inverse_with_mask, torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)):
            inverse, mask = fn(A)
            assert not mask.any()
            assert torch.equal(inverse, torch.eye(n, device=device, dtype=dtype).expand_as(A))

    def test_translation_homography_is_valid_and_inverted_exactly(self, device, dtype):
        # ``det 1`` against ``perm 1``: a rule scaled by the norm of the matrix, ``4096 ** 3`` here, would call
        # the most common matrix kornia inverts singular in float32.
        A = _issue_5476_batch(device, dtype)[4]
        expected = torch.tensor([[1.0, 0.0, -4096.0], [0.0, 1.0, 2048.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        for fn in (safe_inverse_with_mask, torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)):
            inverse, mask = fn(A)
            assert mask.item()
            assert torch.equal(inverse, expected)

    def test_tiny_but_regular_matrix_is_valid(self, device, dtype):
        # diag(1, s) has determinant and permanent s, so it is regular however small s is; only an inverse
        # that does not fit the dtype makes it invalid.
        small = 2.0**-14 if dtype in (torch.float16, torch.bfloat16) else 1e-30
        A = torch.diag(torch.tensor([1.0, small], device=device, dtype=dtype))
        inverse, mask = safe_inverse_with_mask(A)
        assert mask.item()
        assert torch.equal(inverse, torch.diag(torch.tensor([1.0, 1.0 / small], device=device, dtype=dtype)))

    def test_scripts_with_the_eager_verdict_5476(self, device, dtype):
        # The rule has to compile under TorchScript, as the functions did before it.
        A = _issue_5476_batch(device, dtype)
        B = torch.ones(5, 3, device=device, dtype=dtype)
        inverse, mask = torch.jit.script(safe_inverse_with_mask)(A)
        X, _, valid = torch.jit.script(safe_solve_with_mask)(B, A)
        assert mask.tolist() == valid.tolist() == [True, False, False, False, True]
        assert_close(inverse, safe_inverse_with_mask(A)[0])
        assert_close(X, safe_solve_with_mask(B, A)[0])
        if dtype in (torch.float32, torch.float64):
            # the scripted threshold reads the eps and the floor of the dtype, as the eager one does
            eps = torch.finfo(dtype).eps
            edges = torch.tensor([[[1.0, 1.0], [1.0, 1.0 + k * eps]] for k in (24, 64)], device=device, dtype=dtype)
            zero_row = torch.tensor([[1.0, 2.0], [0.0, 0.0]], device=device, dtype=dtype)
            edges = torch.cat([edges, zero_row[None]])
            assert torch.jit.script(_is_singular)(edges).tolist() == _is_singular(edges).tolist() == [True, False, True]

    def test_regular_matrix_is_regular_at_a_scale_whose_products_overflow(self, device, dtype):
        # The terms of a 4x4 determinant are products of four entries, which overflow (or underflow) the dtype
        # long before the inverse does: unscaled, the rule would read ``inf <= inf`` (or ``0 <= 0``) and call a
        # well-conditioned matrix singular. Scaling two rows or two columns alone underflows the products too.
        # Powers of two keep the scaling exact. Half input is decided in float32, where these scales are harmless.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half input is decided in float32")
        A = _regular_at_extreme_scales(device, dtype)
        assert not _is_singular(A).any()
        assert safe_inverse_with_mask(A)[1].all()
        assert safe_solve_with_mask(torch.ones(len(A), 4, device=device, dtype=dtype), A)[2].all()

    def test_regular_matrix_at_an_extreme_scale_is_valid_under_trace_5507(self, device, dtype):
        # Under capture the inverse is the adjugate over the determinant, and their entries overflow or underflow
        # the dtype long before the inverse does: ``1e13 * I`` of order 4 reads ``inf / inf`` in float32 and
        # ``1e-13 * I`` reads ``x / 0``, so the traced mask rejected both matrices that eager accepted. The adjugate
        # is now taken on a row- and column-scaled copy, and the traced verdict and inverse are the eager ones.
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half input is decided in float32")
        A = _regular_at_extreme_scales(device, dtype)
        expected_inverse, expected_mask = safe_inverse_with_mask(A)
        traced = torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)
        inverse, mask = traced(A)
        assert mask.tolist() == expected_mask.tolist() == [True] * len(A)
        assert torch.isfinite(inverse).all()
        assert_close(inverse, expected_inverse, atol=0.0, rtol=1e-5 if dtype == torch.float32 else 1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_nonfinite_matrix_is_invalid_under_trace_5507(self, device, dtype, n, monkeypatch):
        # The traced inverse balances the matrix by exponents read as indices into a table of powers of two. A NaN
        # entry has a NaN exponent, and a NaN cast to int64 is 0 on arm64 but the most negative int64 on x86 and
        # CUDA, out of the table's range; the exponent of a NaN or infinite row or column is 0 instead. The traced
        # mask then rejects the matrix and its row is the identity, as in eager mode.
        from kornia.core import utils

        if dtype not in (torch.float32, torch.float64):
            pytest.skip("half input is decided in float32")
        exponents = []
        times_power_of_two = utils._times_power_of_two

        def spy(x, exponent):
            exponents.append(exponent)
            return times_power_of_two(x, exponent)

        monkeypatch.setattr(utils, "_times_power_of_two", spy)
        eye = torch.eye(n, device=device, dtype=dtype)
        A = torch.stack([eye * 2, eye, eye])
        A[1, 0, 1] = float("nan")
        A[2, 0, 1] = float("inf")
        inverse, mask = torch.jit.trace(safe_inverse_with_mask, (A,), check_trace=False)(A)
        assert mask.tolist() == safe_inverse_with_mask(A)[1].tolist() == [True, False, False]
        assert torch.equal(inverse[1:], eye.expand(2, n, n))
        assert exponents
        assert all(torch.isfinite(e).all() for e in exponents)

    def test_rule_calls_a_zero_row_or_column_singular(self, device, dtype):
        # The rule divides each row and column by its largest magnitude; a zero row or column has to stay
        # zero (determinant and permanent 0, so singular) rather than turn into 0 / 0.
        A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 5.0, 1.0], [2.0, 1.0, 6.0]], device=device, dtype=dtype)
        zero_row, zero_col = A.clone(), A.clone()
        zero_row[1] = 0.0
        zero_col[:, 2] = 0.0
        assert _is_singular(torch.stack([zero_row, zero_col, torch.zeros_like(A)])).all()
        assert not _is_singular(A).item()

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_rule_reads_the_determinant_against_the_permanent(self, device, dtype, n):
        A = torch.tensor(_DEPENDENT_ROWS[n], device=device, dtype=dtype)
        det, perm = _det_perm_closed_form(A)
        assert torch.equal(det, torch.zeros_like(det))
        assert perm.item() > 0
        assert _is_singular(A).item()
        # [[1, 1], [1, 1 + d]] has determinant d, exact for a power of two d, against a permanent of 2 + d:
        # the threshold ``8 * n * eps * perm`` sits at about 32 eps, so 64 eps is regular and 24 eps singular
        # (a threshold without the order ``n``, about 16 eps, would call 24 eps regular)
        eps = torch.finfo(dtype).eps
        edge = torch.tensor([[1.0, 1.0], [1.0, 1.0 + 64 * eps]], device=device, dtype=dtype)
        assert torch.equal(_det_perm_closed_form(edge)[0], torch.tensor(64 * eps, device=device, dtype=dtype))
        assert not _is_singular(edge).item()
        edge[1, 1] = 1.0 + 24 * eps
        assert _is_singular(edge).item()

    def test_rule_rejects_other_shapes(self, device, dtype):
        with pytest.raises(NotImplementedError):
            _is_singular(torch.eye(5, device=device, dtype=dtype))


def test_is_autocast_enabled_cpu():
    assert not is_autocast_enabled()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert is_autocast_enabled()

    assert not is_autocast_enabled()


def test_mps_linalg_helpers_avoid_unsupported_kernels(monkeypatch):
    """MPS fall back to the host instead of calling kernels missing on the torch floor."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    def reject_mps(fn):
        def wrapped(*args, **kwargs):
            if any(torch.is_tensor(arg) and arg.device.type == "mps" for arg in args):
                raise NotImplementedError("MPS kernel unavailable")
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch.linalg, "lu_factor_ex", reject_mps(torch.linalg.lu_factor_ex))
    monkeypatch.setattr(torch.linalg, "lu_solve", reject_mps(torch.linalg.lu_solve))
    monkeypatch.setattr(torch.linalg, "solve", reject_mps(torch.linalg.solve))
    monkeypatch.setattr(torch.linalg, "solve_ex", reject_mps(torch.linalg.solve_ex))
    monkeypatch.setattr(torch.linalg, "svdvals", reject_mps(torch.linalg.svdvals))
    monkeypatch.setattr(torch.linalg, "qr", reject_mps(torch.linalg.qr))
    monkeypatch.setattr(torch, "lu_unpack", reject_mps(torch.lu_unpack))
    monkeypatch.setattr(torch, "det", reject_mps(torch.det))

    device = torch.device("mps")
    A = torch.eye(3, device=device)
    B = torch.ones(1, 3, 1, device=device)

    LU, pivots, info = _torch_linalg_lu_factor_ex(A[None])
    X = _torch_linalg_lu_solve(LU, pivots, B)
    solved, solved_info = _torch_linalg_solve_ex(A[None], B)
    singular_values = _torch_linalg_svdvals(A[None])
    Q, R = _torch_linalg_qr(A[None])
    permutation, _, _ = _torch_lu_unpack(LU, pivots, unpack_data=False)
    determinant = _torch_det(A[None])
    solved_direct = _torch_solve_cast(A[None], B)

    results = (LU, pivots, info, X, solved, solved_info, singular_values, Q, R, permutation, determinant, solved_direct)
    for result in results:
        assert result.device.type == "mps"


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
