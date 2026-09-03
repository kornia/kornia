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
import torch.nn.functional as F
from torch.linalg import inv_ex

from kornia.core._compat import torch_version_ge
from kornia.core._small_linalg import (
    _adjugate_2x2,
    _adjugate_3x3,
    _adjugate_4x4,
    _inverse_3x3_cross,
    _inverse_3x3_scalar,
)
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
        # `torch.empty(0).device` reads the current default device and, unlike
        # `torch.get_default_device()`, is traceable by dynamo — so this helper stays
        # fullgraph-compilable even when a caller can't prove a tensor is in the list.
        device = torch.empty(0).device
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


def _l2_normalize(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """L2-normalise ``input`` along ``dim`` with :func:`torch.nn.functional.normalize`'s default ``eps``.

    ``normalize`` divides by ``norm.clamp_min(eps)``, and the 1e-12 default underflows to zero in
    float16, where an all-zero input therefore normalised to NaN. A float16 input is normalised in
    float32 and cast back. Clamping the norm at the smallest float16 normal instead is safe but not
    neutral: a vector whose norm sits in the subnormal window -- representable, and computed exactly
    because the float16 ``norm`` accumulates in float32 -- came back with a norm of 0.5 rather than
    1. Every other floating dtype carries the 1e-12 default and is unchanged.

    Args:
        input: the tensor to normalise.
        dim: the dimension to normalise along.

    Returns:
        the normalised tensor, in ``input``'s dtype. An all-zero vector normalises to zero with a
        zero gradient: a zero vector has no direction, and the gradient of the ``eps`` clamp there,
        ``1 / eps``, is ~1e12 in float32 and overflows to ``inf`` once cast back to float16. A
        non-zero vector keeps ``normalize``'s value and gradient.
    """
    x = input.float() if input.dtype == torch.float16 else input
    # `amax` rather than a squared norm, so a tiny non-zero vector cannot underflow into the zero branch.
    nonzero = x.abs().amax(dim=dim, keepdim=True) > 0
    out = torch.where(nonzero, F.normalize(x, dim=dim, eps=1e-12), torch.zeros_like(x))
    return out.to(input.dtype)


def _inverse_3x3_closed_form(input: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse for batched 3x3 matrices, dispatching on the execution mode.

    Used as an ONNX-traceable fallback to ``torch.linalg.inv``: the legacy ONNX
    exporter does not lower ``aten::linalg_inv`` (as of opset 17). Computed via
    the adjugate / determinant formula, which is composed entirely of basic
    arithmetic ops that all standard ONNX opsets support.

    The arithmetic lives in :mod:`kornia.core._small_linalg`; this function owns only the
    choice between the two kernels, which is execution-mode policy and therefore stays here.

    Args:
        input: Tensor of shape ``(..., 3, 3)``.

    Returns:
        Tensor of shape ``(..., 3, 3)`` containing the matrix inverse for each
        leading-dim slice. Numerically equivalent to ``torch.linalg.inv`` for
        well-conditioned matrices; behavior on singular matrices is undefined
        (no explicit check, same as ``torch.linalg.inv`` itself).
    """
    if not _is_tracing_or_exporting():
        # Eager: three fused ``cross`` ops beat nine scalar cofactor expressions and four
        # stacks, because kernel launches dominate on matrices this small.
        return _inverse_3x3_cross(input)

    # Under tracing/export (legacy ONNX / jit.trace / dynamo ONNX) stick to the plain scalar
    # adjugate: it lowers to basic arithmetic that every opset supports, whereas ``cross`` may not.
    return _inverse_3x3_scalar(input)


def _is_tracing_or_exporting() -> bool:
    """Whether a graph is being captured by ``torch.jit.trace`` or ``torch.export``/dynamo ONNX export.

    Both capture modes lack ONNX lowerings for the ``linalg`` decompositions (``inv``, ``inv_ex``,
    ``lu_factor``), so callers switch to closed-form arithmetic. Always ``False`` under TorchScript.
    """
    if torch.jit.is_scripting():
        return False
    return torch.jit.is_tracing() or is_exporting()


def _adjugate_closed_form(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjugate and determinant of batched square matrices up to 4x4 in basic arithmetic only.

    Raises:
        NotImplementedError: for shapes other than ``(..., n, n)`` with ``n`` in 2, 3, 4.
    """
    n = input.shape[-1]
    if input.shape[-2] == n and n == 2:
        return _adjugate_2x2(input)
    if input.shape[-2] == n and n == 3:
        return _adjugate_3x3(input)
    if input.shape[-2] == n and n == 4:
        return _adjugate_4x4(input)
    raise NotImplementedError(f"Closed-form inverse only supports 2x2, 3x3 and 4x4 matrices, got {list(input.shape)}")


def _has_closed_form_inverse(input: torch.Tensor) -> bool:
    n = input.shape[-1]
    return input.shape[-2] == n and n in (2, 3, 4)


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.

    Under graph capture (``torch.jit.trace``, legacy ``torch.onnx.export`` and the dynamo
    ``torch.onnx.export(..., dynamo=True)`` / ``torch.export`` path) on 2x2, 3x3 and 4x4
    matrices, falls back to a closed-form adjugate inverse so the resulting graph does not
    include ``aten::linalg_inv``, which neither ONNX exporter lowers. ``torch.jit.is_tracing()``
    is JIT-script-safe (unlike ``torch.onnx.is_in_onnx_export``, which contains an ``import``
    statement).
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)
    if _is_tracing_or_exporting() and _has_closed_form_inverse(input):
        adj, det = _adjugate_closed_form(input.to(dtype))
        return (adj / det[..., None, None]).to(input.dtype)
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

    For numerical stability, fp32 inputs are promoted to fp64 (except on MPS where fp64 is unsupported).
    """
    if is_mps_tensor_safe(input):
        dtype = torch.float32
    elif input.dtype == torch.float32:
        dtype = torch.float64
    else:
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

    if _is_tracing_or_exporting() and _has_closed_form_inverse(A):
        # ``linalg_inv_ex`` has no ONNX lowering; the adjugate form is basic arithmetic, and a
        # zero determinant is exactly the singularity ``inv_ex`` flags through ``info``.
        adj, det = _adjugate_closed_form(A.to(dtype))
        mask = det != 0
        safe_det = torch.where(mask, det, torch.ones_like(det))
        return (adj / safe_det[..., None, None]).to(dtype_original), mask

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


# These helpers moved into ``torch.compiler`` over time; resolve them once at import.
_torch_is_compiling = getattr(torch.compiler, "is_compiling", None) or getattr(torch._dynamo, "is_compiling", None)
_torch_is_exporting = getattr(torch.compiler, "is_exporting", None)


@torch.jit.unused
def is_compiling() -> bool:
    """Whether execution is inside ``torch.compile`` or ``torch.export`` capture.

    Falls back to Torch's older private Dynamo spelling when the public compiler helper is absent.
    """
    return bool(_torch_is_compiling()) if _torch_is_compiling is not None else False


@torch.jit.unused
def _is_exporting_eager() -> bool:
    if _torch_is_exporting is not None:
        return bool(_torch_is_exporting())
    # torch < 2.6 has no export flag. Inside a Dynamo trace the newer releases constant-fold
    # ``torch.compiler.is_exporting`` to ``True`` for ``torch.compile`` as well as for
    # ``torch.export``, so ``is_compiling`` is the fallback with the same semantics.
    return is_compiling()


def is_exporting() -> bool:
    """Whether execution is inside a graph capture by ``torch.export`` or the dynamo ONNX exporter.

    Used to switch to export-safe arithmetic (closed-form inverses, ``sort``-based medians, ...) and
    to skip in-``forward`` side effects (e.g. stashing per-call state on ``self``) that
    ``torch.export`` rejects, without changing the captured output. Inside a Dynamo trace torch
    folds its own flag to ``True`` for ``torch.compile`` too, so the export-safe paths are also
    what a compiled graph contains; on torch < 2.6, which has no export flag, ``is_compiling`` is
    used for the same reason. Always ``False`` inside TorchScript, so the guard is safe to call
    from scripted functions.
    """
    if torch.jit.is_scripting():
        return False
    return _is_exporting_eager()


def register_module_state(module: torch.nn.Module, name: str, x: torch.Tensor) -> None:
    """Store tensor ``x`` on ``module`` as ``name`` so it is optimizable, movable and serializable.

    A leaf tensor (user-provided data or an existing parameter) becomes an ``nn.Parameter``, as
    before. ``nn.Parameter(x)`` would re-root a tensor that already carries a ``grad_fn`` as a new
    leaf, so a group built from ``Se3.exp(v)`` would stop propagating gradients to ``v``; such a
    tensor is registered as a buffer instead, which keeps its history while ``.to()``,
    ``state_dict()`` and ``load_state_dict()`` still reach it under the same key. Under graph
    capture (``torch.jit.trace``, ``torch.compile``, ``torch.export`` and the dynamo ONNX
    exporter) neither a parameter nor a buffer can be created inside the traced region, so the
    tensor is kept as a plain attribute of the module being built.
    """
    if isinstance(x, torch.nn.Parameter) or not (torch.jit.is_tracing() or is_compiling() or is_exporting()):
        if x.grad_fn is None or isinstance(x, torch.nn.Parameter):
            x = x if isinstance(x, torch.nn.Parameter) else torch.nn.Parameter(x)
        else:
            module.register_buffer(name, x)
            return
    setattr(module, name, x)


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


def batched_forward(
    model: torch.nn.Module, data: torch.Tensor, device: torch.device, batch_size: int = 128, **kwargs: Any
) -> torch.Tensor:
    r"""Run the forward in micro-batches.

    When the just model.forward(data) does not fit into device memory, e.g. on laptop GPU.
    In the end, it transfers the output to the device of the input data tensor.
    E.g. running HardNet on 8000x1x32x32 tensor.

    Removed from ``kornia.utils.memory`` in 0.8.3 and restored here as public API.

    Args:
        model: Any torch model, which outputs a single tensor as an output.
        data: Input data of Bx(Any) shape.
        device: which device should we run on.
        batch_size: "micro-batch" size.
        **kwargs: any other arguments, which accepts model.

    Returns:
        output of the model.

    Example:
        >>> import torch
        >>> from kornia.core.utils import batched_forward
        >>> model = torch.nn.Identity()
        >>> x = torch.rand(300, 2)
        >>> out = batched_forward(model, x, torch.device("cpu"), batch_size=128)
        >>> bool(torch.allclose(out, x))
        True

    """
    model_dev = model.to(device)
    B: int = len(data)
    bs: int = batch_size
    if B > batch_size:
        out_list = []
        n_batches = int(B // bs + 1)
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            end = min((batch_idx + 1) * bs, B)
            if st >= end:
                continue
            out_list.append(model_dev(data[st:end].to(device), **kwargs))
        out = torch.cat(out_list, 0)
        return out.to(data.device)
    return model_dev(data.to(device), **kwargs).to(data.device)
