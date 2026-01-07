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


import importlib.util
import platform
import sys
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch.linalg import inv_ex

from kornia.core._compat import torch_version_ge
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_TYPE
from kornia.core.exceptions import DeviceError


def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


def is_mps_tensor_safe(x: torch.Tensor) -> bool:
    """Return whether tensor is on MPS device."""
    return "mps" in str(x.device)


def get_cuda_device_if_available(index: int = 0) -> torch.device:
    """Try to get cuda device, if fail, return cpu.

    Args:
        index: cuda device index

    Returns:
        torch.device

    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")

    return torch.device("cpu")


def get_mps_device_if_available() -> torch.device:
    """Try to get mps device, if fail, return cpu.

    Returns:
        torch.device

    """
    dev = "cpu"
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            dev = "mps"
    return torch.device(dev)


def get_cuda_or_mps_device_if_available() -> torch.device:
    """Check OS and platform and run get_cuda_device_if_available or get_mps_device_if_available.

    Returns:
        torch.device

    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return get_mps_device_if_available()
    else:
        return get_cuda_device_if_available()


def _extract_device_dtype(tensor_list: List[Optional[Any]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are torch.Tensor).

    If so, it would return a tuple of (device, dtype).
    Default: (``torch.get_default_device()``, ``torch.get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]

    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                continue
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise DeviceError(
                    f"Passed values are not in the same device and dtype. "
                    f"Got ({device}, {dtype}) and ({_device}, {_dtype}).",
                    actual_devices=[device, _device],
                    expected_device=device,
                )
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (device, dtype)


def _normalize_to_float32_or_float64(dtype: torch.dtype) -> torch.dtype:
    """Normalize dtype to float32 or float64 for operations that require full precision.

    Args:
        dtype: The input dtype to normalize.

    Returns:
        torch.float32 if dtype is not float32 or float64, otherwise returns the original dtype.
    """
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)
    return torch.linalg.inv(input.to(dtype)).to(input.dtype)


def _torch_histc_cast(input: torch.Tensor, bins: int, min: Union[float, bool], max: Union[float, bool]) -> torch.Tensor:
    """Make torch.histc work with other than fp32/64.

    The function torch.histc is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)
    return torch.histc(input.to(dtype), bins, min, max).to(input.dtype)


def _torch_svd_cast(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    dtype = _normalize_to_float32_or_float64(input.dtype)

    out1, out2, out3H = torch.linalg.svd(input.to(dtype))
    # Since kornia requires torch>=2.0.0, we can always use .mH
    out3 = out3H.mH
    return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))


def _torch_linalg_svdvals(input: torch.Tensor) -> torch.Tensor:
    """Make torch.linalg.svdvals work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)

    # Since kornia requires torch>=2.0.0, we can always use torch.linalg.svdvals
    out = torch.linalg.svdvals(input.to(dtype))
    return out.to(input.dtype)


def _torch_solve_cast(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Make torch.solve work with other than fp32/64.

    For stable operation, the input matrices should be cast to fp64, and the output will
    be cast back to the input dtype. However, fp64 is not yet supported on MPS.

    This function is actively used in:
    - kornia.geometry.transform.imgwarp
    - kornia.geometry.transform.thin_plate_spline
    - kornia.geometry.epipolar.essential
    """
    if is_mps_tensor_safe(A):
        dtype = torch.float32
    else:
        dtype = torch.float64

    out = torch.linalg.solve(A.to(dtype), B.to(dtype))

    # cast back to the input dtype
    return out.to(A.dtype)


def safe_solve_with_mask(B: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Solves the system of equations.

    Avoids crashing because of singular matrix input and outputs the mask of valid solution.
    """
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    KORNIA_CHECK_IS_TENSOR(B, "B must be torch.Tensor")
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    # Since kornia requires torch>=2.0.0, we can always use torch.linalg.lu_factor_ex and torch.linalg.lu_solve
    A_LU, pivots, info = torch.linalg.lu_factor_ex(A.to(dtype))

    valid_mask: torch.Tensor = info == 0
    n_dim_B = len(B.shape)
    n_dim_A = len(A.shape)
    if n_dim_A - n_dim_B == 1:
        B = B.unsqueeze(-1)

    X = torch.linalg.lu_solve(A_LU, pivots, B.to(dtype))

    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def safe_inverse_with_mask(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Perform inverse.

    Avoids crashing because of non-invertable matrix input and outputs the mask of valid solution.
    """
    KORNIA_CHECK_IS_TENSOR(A, "A must be torch.Tensor")

    dtype_original = A.dtype
    dtype = _normalize_to_float32_or_float64(dtype_original)

    inverse, info = inv_ex(A.to(dtype))
    mask = info == 0
    return inverse.to(dtype_original), mask


def is_autocast_enabled(both: bool = True) -> bool:
    """Check if torch autocast is enabled.

    Args:
        both: if True will consider autocast region for both types of devices

    Returns:
        Return a Bool,
        will always return False for a torch without support, otherwise will be: if both is True
        `torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')`. If both is False will return just
        `torch.is_autocast_enabled()`.

    """
    # Since kornia requires torch>=2.0.0, autocast is always available
    if both:
        if torch_version_ge(2, 4):
            return torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu")
        else:
            return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()

    return torch.is_autocast_enabled()


def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dictionaries."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {key: dataclass_to_dict(value) for key, value in asdict(obj).items()}
    elif isinstance(obj, list | tuple):
        return type(obj)(dataclass_to_dict(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj


T = TypeVar("T")


def dict_to_dataclass(dict_obj: Dict[str, Any], dataclass_type: Type[T]) -> T:
    """Recursively convert dictionaries to dataclass instances."""
    KORNIA_CHECK_TYPE(dict_obj, dict, "Input conf must be dict")
    KORNIA_CHECK(is_dataclass(dataclass_type), "dataclass_type must be a dataclass")
    field_types: dict[str, Any] = {f.name: f.type for f in fields(dataclass_type)}
    constructor_args = {}
    for key, value in dict_obj.items():
        if key in field_types and is_dataclass(field_types[key]):
            constructor_args[key] = dict_to_dataclass(value, field_types[key])
        else:
            constructor_args[key] = value
    # TODO: remove type ignore when https://github.com/python/mypy/issues/14941 be andressed
    return dataclass_type(**constructor_args)
