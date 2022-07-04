import pytest
import torch

from kornia.testing import assert_close
from kornia.utils import _extract_device_dtype
from kornia.utils.helpers import (
    _torch_histc_cast,
    _torch_inverse_cast,
    _torch_solve_cast,
    _torch_svd_cast,
    safe_inverse_with_mask,
    safe_solve_with_mask,
)


@pytest.mark.parametrize(
    "tensor_list,out_device,out_dtype,will_throw_error",
    [
        ([], torch.device('cpu'), torch.get_default_dtype(), False),
        ([None, None], torch.device('cpu'), torch.get_default_dtype(), False),
        ([torch.tensor(0, device='cpu', dtype=torch.float16), None], torch.device('cpu'), torch.float16, False),
        ([torch.tensor(0, device='cpu', dtype=torch.float32), None], torch.device('cpu'), torch.float32, False),
        ([torch.tensor(0, device='cpu', dtype=torch.float64), None], torch.device('cpu'), torch.float64, False),
        ([torch.tensor(0, device='cpu', dtype=torch.float16)] * 2, torch.device('cpu'), torch.float16, False),
        ([torch.tensor(0, device='cpu', dtype=torch.float32)] * 2, torch.device('cpu'), torch.float32, False),
        ([torch.tensor(0, device='cpu', dtype=torch.float64)] * 2, torch.device('cpu'), torch.float64, False),
        (
            [torch.tensor(0, device='cpu', dtype=torch.float16), torch.tensor(0, device='cpu', dtype=torch.float64)],
            None,
            None,
            True,
        ),
        (
            [torch.tensor(0, device='cpu', dtype=torch.float32), torch.tensor(0, device='cpu', dtype=torch.float64)],
            None,
            None,
            True,
        ),
        (
            [torch.tensor(0, device='cpu', dtype=torch.float16), torch.tensor(0, device='cpu', dtype=torch.float32)],
            None,
            None,
            True,
        ),
    ],
)
def test_extract_device_dtype(tensor_list, out_device, out_dtype, will_throw_error):
    # TODO: include the warning in another way - possibly loggers.
    # Add GPU tests when GPU testing available
    # if torch.cuda.is_available():
    #     import warnings
    #     warnings.warn("Add GPU tests.")

    if will_throw_error:
        with pytest.raises(ValueError):
            _extract_device_dtype(tensor_list)
    else:
        device, dtype = _extract_device_dtype(tensor_list)
        assert device == out_device
        assert dtype == out_dtype


class TestInverseCast:
    @pytest.mark.parametrize("input_shape", [(1, 3, 4, 4), (2, 4, 5, 5)])
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
        A = torch.randn(2, 3, 1, 4, 4, device=device, dtype=dtype)
        B = torch.randn(2, 3, 1, 4, 6, device=device, dtype=dtype)

        X = _torch_solve_cast(A, B)
        error = torch.dist(B, A.matmul(X))

        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-4
        assert_close(error, torch.zeros_like(error), atol=tol_val, rtol=tol_val)


class TestSolveWithMask:
    def test_smoke(self, device, dtype):
        A = torch.randn(2, 3, 1, 4, 4, device=device, dtype=dtype)
        B = torch.randn(2, 3, 1, 4, 6, device=device, dtype=dtype)

        X, _, mask = safe_solve_with_mask(B, A)
        X2 = _torch_solve_cast(A, B)
        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-4
        if mask.sum() > 0:
            assert_close(X[mask], X2[mask], atol=tol_val, rtol=tol_val)

    @pytest.mark.skipif(
        (int(torch.__version__.split('.')[0]) == 1) and (int(torch.__version__.split('.')[1]) < 10),
        reason='<1.10.0 not supporting',
    )
    def test_all_bad(self, device, dtype):
        A = torch.ones(10, 3, 3, device=device, dtype=dtype)
        B = torch.ones(3, 10, device=device, dtype=dtype)

        X, _, mask = safe_solve_with_mask(B, A)
        assert torch.equal(mask, torch.zeros_like(mask))


class TestInverseWithMask:
    def test_smoke(self, device, dtype):
        x = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device=device, dtype=dtype)

        y_expected = torch.tensor([[0.6, -0.7], [-0.2, 0.4]], device=device, dtype=dtype)

        y, mask = safe_inverse_with_mask(x)

        assert_close(y, y_expected)
        assert torch.equal(mask, torch.ones_like(mask))

    @pytest.mark.skipif(
        (int(torch.__version__.split('.')[0]) == 1) and (int(torch.__version__.split('.')[1]) < 9),
        reason='<1.9.0 not supporting',
    )
    def test_all_bad(self, device, dtype):
        A = torch.ones(10, 3, 3, device=device, dtype=dtype)
        X, mask = safe_inverse_with_mask(A)
        assert torch.equal(mask, torch.zeros_like(mask))
