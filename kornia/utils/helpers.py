import warnings
from functools import partial, wraps
from inspect import isclass, isfunction
from typing import Any, Callable, List, Optional, Tuple

import torch

from kornia.core import Tensor
from kornia.utils._compat import torch_version_geq


def get_cuda_device_if_available(index: int = 0) -> torch.device:
    """Tries to get cuda device, if fail, returns cpu.

    Args:
        index: cuda device index

    Returns:
        torch.device
    """
    try:
        if torch.cuda.is_available():
            dev = torch.device(f'cuda:{index}')
        else:
            dev = torch.device('cpu')
    except BaseException as e:  # noqa: F841
        dev = torch.device('cpu')
    return dev


def _deprecated(func: Callable = None, replace_with: Optional[str] = None):
    if func is None:
        return partial(_deprecated, replace_with=replace_with)

    @wraps(func)
    def wrapper(*args, **kwargs):
        name: str = ""
        if isclass(func):
            name = func.__class__.__name__
        if isfunction(func):
            name = func.__name__
        if replace_with is not None:
            warnings.warn(f"`{name}` is deprecated in favor of `{replace_with}`.", category=DeprecationWarning)
        else:
            warnings.warn(
                f"`{name}` is deprecated and will be removed in the future versions.", category=DeprecationWarning
            )
        return func(*args, **kwargs)

    return wrapper


def _extract_device_dtype(tensor_list: List[Optional[Any]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are torch.Tensor).

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                continue
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError(
                    "Passed values are not in the same device and dtype."
                    f"Got ({device}, {dtype}) and ({_device}, {_dtype})."
                )
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (device, dtype)


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)


def _torch_histc_cast(input: torch.Tensor, bins: int, min: int, max: int) -> torch.Tensor:
    """Helper function to make torch.histc work with other than fp32/64.

    The function torch.histc is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.histc(input.to(dtype), bins, min, max).to(input.dtype)


def _torch_svd_cast(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    out1, out2, out3 = torch.svd(input.to(dtype))

    return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))


# TODO: return only `torch.Tensor` and review all the calls to adjust
def _torch_solve_cast(A: Tensor, B: Tensor) -> Tensor:
    """Helper function to make torch.solve work with other than fp32/64.

    The function torch.solve is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.svd, and cast back to the input dtype.
    """
    dtype: torch.dtype = A.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    out = torch.linalg.solve(A.to(dtype), B.to(dtype))

    return out.to(A.dtype)


def safe_solve_with_mask(B: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Helper function, which avoids crashing because of singular matrix input and outputs the mask of valid
    solution."""
    if not torch_version_geq(1, 10):
        sol = _torch_solve_cast(A, B)
        warnings.warn('PyTorch version < 1.10, solve validness mask maybe not correct', RuntimeWarning)
        return sol, sol, torch.ones(len(A), dtype=torch.bool, device=A.device)
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    if not isinstance(B, torch.Tensor):
        raise AssertionError(f"B must be torch.Tensor. Got: {type(B)}.")
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    A_LU, pivots, info = torch.lu(A.to(dtype), get_infos=True)
    valid_mask: torch.Tensor = info == 0
    X = torch.lu_solve(B.to(dtype), A_LU, pivots)
    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def safe_inverse_with_mask(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Helper function, which avoids crashing because of non-invertable matrix input and outputs the mask of valid
    solution."""
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    if not torch_version_geq(1, 9):
        inv = _torch_inverse_cast(A)
        warnings.warn('PyTorch version < 1.9, inverse validness mask maybe not correct', RuntimeWarning)
        return inv, torch.ones(len(A), dtype=torch.bool, device=A.device)
    if not isinstance(A, torch.Tensor):
        raise AssertionError(f"A must be torch.Tensor. Got: {type(A)}.")
    dtype_original: torch.dtype = A.dtype
    if dtype_original not in (torch.float32, torch.float64):
        dtype = torch.float32
    else:
        dtype = dtype_original
    from torch.linalg import inv_ex  # type: ignore # (not available in 1.8.1)

    inverse, info = inv_ex(A.to(dtype))
    mask = info == 0
    return inverse.to(dtype_original), mask
