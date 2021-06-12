import pytest
import torch
from torch.testing import assert_allclose

from kornia.utils import _extract_device_dtype
from kornia.utils.helpers import _torch_histc_cast, _torch_inverse_cast, _torch_solve_cast, _torch_svd_cast


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
    # Add GPU tests when GPU testing avaliable
    if torch.cuda.is_available():
        import warnings

        warnings.warn("Add GPU tests.")

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

        assert_allclose(y, y_expected)

    def test_jit(self, device, dtype):
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        op = _torch_inverse_cast
        op_jit = torch.jit.script(op)
        assert_allclose(op(x), op_jit(x))


class TestHistcCast:
    def test_smoke(self, device, dtype):
        x = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        y_expected = torch.tensor([0.0, 2.0, 1.0, 0.0], device=device, dtype=dtype)

        y = _torch_histc_cast(x, bins=4, min=0, max=3)

        assert_allclose(y, y_expected)


class TestSvdCast:
    def test_smoke(self, device, dtype):
        a = torch.randn(5, 3, 3, device=device, dtype=dtype)
        u, s, v = _torch_svd_cast(a)

        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-3
        assert_allclose(a, u @ torch.diag_embed(s) @ v.transpose(-2, -1), atol=tol_val, rtol=tol_val)


class TestSolveCast:
    def test_smoke(self, device, dtype):
        A = torch.randn(2, 3, 1, 4, 4, device=device, dtype=dtype)
        B = torch.randn(2, 3, 1, 4, 6, device=device, dtype=dtype)

        X, LU = _torch_solve_cast(B, A)
        error = torch.dist(B, A.matmul(X))

        tol_val: float = 1e-1 if dtype == torch.float16 else 1e-4
        assert_allclose(error, torch.zeros_like(error), atol=tol_val, rtol=tol_val)
