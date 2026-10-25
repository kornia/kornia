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
import warnings
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, fields, is_dataclass
from types import FrameType, UnionType
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import torch
import torch.nn.functional as F
from torch.linalg import inv_ex

from kornia.core._small_linalg import (
    _adjugate_2x2,
    _adjugate_3x3,
    _adjugate_4x4,
    _det_perm_2x2,
    _det_perm_3x3,
    _det_perm_4x4,
    _inverse_3x3_cross,
    _inverse_3x3_scalar,
)
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_TYPE
from kornia.core.exceptions import DeviceError, TypeCheckError


def _torch_grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """Run :func:`torch.nn.functional.grid_sample` on 4-D or 5-D input, CPU float16/bfloat16 in float32.

    torch's CPU half ``grid_sample`` returns corrupt values on sources of about 128 px and more, and
    torch 2.5.1 has no such kernel (#5751). Other devices and dtypes call ``grid_sample`` unchanged.
    """
    if input.device.type == "cpu" and input.dtype in (torch.float16, torch.bfloat16) and grid.dtype == input.dtype:
        return F.grid_sample(
            input.float(), grid.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
        ).to(input.dtype)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def _to_device_nonblocking(
    tensor: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` (and ``dtype``), without waiting for queued work on a CUDA target.

    A blocking host-to-device copy from pageable memory ends in ``cudaStreamSynchronize``, so the host waits
    until the GPU has finished everything queued before it. With ``non_blocking=True`` CUDA stages a pageable
    source before the call returns, so the source may be changed or freed straight away. A dtype change is made
    on the host before the copy, as the blocking copy makes it, so the copied bits are the same, NaN payloads
    included.

    Only an eager copy from pageable CPU memory to CUDA is non-blocking:

    - A pinned source is copied asynchronously by a non-blocking copy and could be changed by its owner before the
      copy lands, so it keeps the blocking copy.
    - Under ``torch.compile`` and ``torch.export`` the copy stays blocking, as it was: Inductor (torch 2.14)
      miscompiles a non-blocking copy of a CPU tensor computed inside the graph, such as parameters sampled in a
      compiled ``forward``, and returns wrong values.
    - A copy to the CPU stays blocking, because the host could otherwise read it before it arrives, and MPS and
      other targets keep the blocking copy, since they may read the source after the call returns.

    Args:
        tensor: The tensor to move.
        device: The target device.
        dtype: The target dtype, or ``None`` to keep the dtype of ``tensor``.

    Returns:
        ``tensor`` itself when it already has the target device and dtype, a copy otherwise.
    """
    if torch.jit.is_scripting():
        non_blocking = tensor.device.type == "cpu" and device.type == "cuda" and not tensor.is_pinned()
    else:
        non_blocking = (
            tensor.device.type == "cpu" and device.type == "cuda" and not is_compiling() and not tensor.is_pinned()
        )
    if non_blocking:
        if dtype is not None and dtype != tensor.dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device=device, non_blocking=True)
    return tensor.to(device=device, dtype=dtype)


def _as_tensor_nonblocking(data: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return ``torch.as_tensor(data, device=device, dtype=dtype)``, moving a tensor without waiting for the GPU.

    ``torch.as_tensor`` of a tensor is ``tensor.to(device, dtype)``, a blocking copy for a CPU tensor and a CUDA
    ``device``. A tensor is moved without waiting for queued work instead, with the same values; any other ``data``
    goes through ``torch.as_tensor`` unchanged.
    """
    if isinstance(data, torch.Tensor):
        return _to_device_nonblocking(data, device, dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    return importlib.util.find_spec("torch_xla") is not None


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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    return torch.device(dev)


def get_cuda_or_mps_device_if_available() -> torch.device:
    """Check OS and platform and run get_cuda_device_if_available or get_mps_device_if_available.

    Returns:
        torch.device

    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return get_mps_device_if_available()
    return get_cuda_device_if_available()


def _extract_device_dtype(tensor_list: List[Optional[Any]]) -> Tuple[torch.device, torch.dtype]:
    """Check that the tensors in the list share one device and one dtype, and return them.

    Entries that are not tensors (``None`` included) are skipped. Without any tensor, the result is
    (``torch.get_default_device()``, ``torch.get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]

    Raises:
        DeviceError: if two tensors are on different devices. Every device is checked before any dtype, so this
            error wins when the devices and the dtypes both differ, wherever the mismatches sit in the list.
        TypeCheckError: if all the tensors share one device and two of them have different dtypes.

    """
    tensors = [tensor for tensor in tensor_list if isinstance(tensor, torch.Tensor)]
    for tensor in tensors[1:]:
        if tensor.device != tensors[0].device:
            raise DeviceError(
                f"Passed tensors are not on the same device: expected {tensors[0].device}, got {tensor.device}.",
                actual_devices=[tensors[0].device, tensor.device],
                expected_device=tensors[0].device,
            )
    for tensor in tensors[1:]:
        if tensor.dtype != tensors[0].dtype:
            raise TypeCheckError(
                f"Passed tensors do not have the same dtype: expected {tensors[0].dtype}, got {tensor.dtype}.",
                actual_type=tensor.dtype,
                expected_type=tensors[0].dtype,
            )
    if not tensors:
        # `torch.empty(0).device` reads the current default device and, unlike
        # `torch.get_default_device()`, is traceable by dynamo — so this helper stays
        # fullgraph-compilable even when a caller can't prove a tensor is in the list.
        return (torch.empty(0).device, torch.get_default_dtype())
    return (tensors[0].device, tensors[0].dtype)


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
        non-zero vector, including one that holds a NaN, keeps ``normalize``'s value and gradient.
    """
    x = input.float() if input.dtype == torch.float16 else input
    # `amax` rather than a squared norm, so a tiny non-zero vector cannot underflow into the zero branch.
    # `== 0` rather than `> 0`: `amax` propagates NaN and `NaN > 0` is False, which sent a vector holding
    # a NaN down the zero branch and hid the NaN that `normalize` returns.
    zero = x.abs().amax(dim=dim, keepdim=True) == 0
    out = torch.where(zero, torch.zeros_like(x), F.normalize(x, dim=dim, eps=1e-12))
    return out.to(input.dtype)


def _inverse_3x3_closed_form(input: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse for batched 3x3 matrices, dispatching on the execution mode.

    Used as an ONNX-traceable fallback to ``torch.linalg.inv``: the legacy ONNX
    exporter does not lower ``aten::linalg_inv`` (as of opset 17). Computed via
    the adjugate / determinant formula, which is composed entirely of basic
    arithmetic ops that all standard ONNX opsets support.

    The arithmetic lives in :mod:`kornia.core._small_linalg`; this function owns only the
    choice between the two kernels, which is execution-mode policy and therefore stays here.

    A float16 or bfloat16 input is inverted in float32 and the result cast back, as
    :func:`_torch_inverse_cast` does. The determinant is a sum of products of three entries, so it leaves
    float16's range long before the entries do: the pixel-normalization matrix of a 3000 px image,
    with entries ``2 / 2999``, has a subnormal float16 determinant, 6 % off, and one of 11600 px a zero one.
    Running ``cross`` in float32 also avoids the missing bfloat16 ``cross`` kernel on MPS in
    torch 2.5.1.

    Args:
        input: Tensor of shape ``(..., 3, 3)``.

    Returns:
        Tensor of shape ``(..., 3, 3)`` containing the matrix inverse for each
        leading-dim slice, in the dtype of ``input``. Numerically equivalent to ``torch.linalg.inv`` for
        well-conditioned matrices; behavior on singular matrices is undefined
        (no explicit check, same as ``torch.linalg.inv`` itself).
    """
    half = input.dtype in (torch.float16, torch.bfloat16)
    x = input.float() if half else input
    if not _is_tracing_or_exporting():  # noqa: SIM108 - each branch documents its own capture path
        # Eager: three fused ``cross`` ops beat nine scalar cofactor expressions and four
        # stacks, because kernel launches dominate on matrices this small.
        out = _inverse_3x3_cross(x)
    else:
        # Under tracing/export (legacy ONNX / jit.trace / dynamo ONNX) stick to the plain scalar
        # adjugate. NOTE the original rationale here -- "whereas ``cross`` may not [lower]" -- does not
        # hold on the torch versions CI runs: measured, ``torch.linalg.cross`` lowers on 2.5.1 (legacy
        # exporter) and 2.9.1 and 2.14.0 (both exporters). The legacy exporter emits
        # ``Slice``/``Mul``/``Sub``/``Concat``; the dynamo exporter emits ``Split`` in place of ``Slice``,
        # on 2.9.1 as on 2.14.0. The split is kept because it is still the safer capture path (scalar
        # arithmetic needs no per-dtype kernel at all, and ``cross`` has real kernel gaps -- no bfloat16
        # on MPS in torch 2.5.1, which the float32 promotion above keeps half input away from), not
        # because ``cross`` fails to export.
        # Collapsing the two branches is a behavior change and belongs in its own PR.
        out = _inverse_3x3_scalar(x)
    return out.to(input.dtype) if half else out


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


def _det_perm_closed_form(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determinant of batched square matrices up to 4x4 and the permanent of their absolute values.

    Raises:
        NotImplementedError: for shapes other than ``(..., n, n)`` with ``n`` in 2, 3, 4.
    """
    n = input.shape[-1]
    if input.shape[-2] == n and n == 2:
        return _det_perm_2x2(input)
    if input.shape[-2] == n and n == 3:
        return _det_perm_3x3(input)
    if input.shape[-2] == n and n == 4:
        return _det_perm_4x4(input)
    raise NotImplementedError(f"Closed-form determinant supports 2x2, 3x3 and 4x4 matrices, got {list(input.shape)}")


def _has_closed_form_inverse(input: torch.Tensor) -> bool:
    n = input.shape[-1]
    return input.shape[-2] == n and n in (2, 3, 4)


def _closed_form_inverse(input: torch.Tensor) -> torch.Tensor:
    """Inverse of batched 2x2, 3x3 or 4x4 matrices as ``adj / det``, computed on a scaled copy.

    The entries of the adjugate and the determinant are products of up to ``n`` entries, which overflow or
    underflow the dtype long before the inverse does: for ``1e13 * I`` or ``1e-13 * I`` of order 4 in
    float32 they are ``inf`` or ``0`` while the inverse is ``1e-13 * I`` or ``1e13 * I``. So the matrix is
    balanced first, each row divided by the power of two below its largest magnitude and then each column of
    that by the power of two below its own, ``inv(A) = D_c inv(D_r A D_c) D_r``, and the rows and columns of
    the result are scaled back by the same factors. A power of two keeps every rounding step the one of the
    unscaled computation, so the result is bit for bit the former ``adj / det`` wherever its adjugate and
    determinant were normal numbers; where one of them overflowed or went subnormal the former result was
    ``inf``, ``nan``, zero or rounded (``diag(2 ** -100, 2 ** -40)`` in float32 has a subnormal determinant
    and was off by 1.2e-7), and the new one is the inverse.

    The scaling works on the exponents and never forms the row-scaled matrix: ``[[2 ** 100, 2 ** -60],
    [2 ** 100, -2 ** -60]]`` in float32 would lose its second column to underflow on the way, and its inverse
    is finite. The combined exponent of an entry can exceed what the dtype holds while the entry itself fits,
    so it is applied as three powers of two of about a third of it each; every intermediate lies between the
    start and the end, and is exact wherever the end is a normal number. A zero row or column is left alone, and
    so is one holding a NaN, whose exponent would index the table of powers of two out of range (a NaN casts to
    the most negative int64 on x86 and CUDA); its determinant is 0 or NaN either way.

    Raises:
        NotImplementedError: for shapes other than ``(..., n, n)`` with ``n`` in 2, 3, 4.
    """
    exponent = torch.floor(torch.log2(input.abs()))  # -inf at a zero entry, which no maximum below picks
    row = exponent.amax(-1, keepdim=True)
    row = torch.where(torch.isfinite(row), row, torch.zeros_like(row))
    col = (exponent - row).amax(-2, keepdim=True)
    col = torch.where(torch.isfinite(col), col, torch.zeros_like(col))
    adj, det = _adjugate_closed_form(_times_power_of_two(input, -(row + col)))
    return _times_power_of_two(adj / det[..., None, None], -(col.transpose(-2, -1) + row.transpose(-2, -1)))


def _times_power_of_two(x: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """``x * 2 ** exponent`` in three steps of about a third of the exponent each.

    No step and no intermediate leaves the dtype while the result is inside it. ``exponent`` is integer
    valued and is clipped to three times the largest exponent of a normal number of the dtype, beyond which a
    nonzero result is ``inf`` or ``0`` anyway and a zero one stays ``0`` instead of turning into ``0 * inf``.
    The powers of two are read from a table, not computed by ``pow``, which is not exact on every device
    (MPS), and the scaling has to be exact to leave every rounding step as it was.
    """
    largest = 1022 if x.dtype == torch.float64 else 126
    exponent = exponent.clamp(-3.0 * largest, 3.0 * largest)
    step = torch.round(exponent / 3)
    powers = torch.tensor([2.0**k for k in range(-largest, largest + 2)], dtype=x.dtype, device=x.device)
    scale = powers[(step + largest).to(torch.int64)]
    return x * scale * scale * powers[(exponent - 2 * step + largest).to(torch.int64)]


def _torch_inverse_cast(input: torch.Tensor, check_errors: bool = True) -> torch.Tensor:
    """Make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.

    Under graph capture (``torch.jit.trace``, legacy ``torch.onnx.export`` and the dynamo
    ``torch.onnx.export(..., dynamo=True)`` / ``torch.export`` path) on 2x2, 3x3 and 4x4
    matrices, falls back to a closed-form adjugate inverse so the resulting graph does not
    include ``aten::linalg_inv``, which neither ONNX exporter lowers. ``torch.jit.is_tracing()``
    is JIT-script-safe (unlike ``torch.onnx.is_in_onnx_export``, which contains an ``import``
    statement).

    Singular input is not checked here. In eager mode ``torch.linalg.inv`` decides, and raises, from the
    pivots of its own factorization; under capture the adjugate is divided by the determinant, which gives
    non-finite or large values. Which near-singular matrices raise, and which return large values, depends
    on the torch version, the device and the capture mode. Callers that need a verdict use
    :func:`safe_inverse_with_mask`, whose mask follows one rule everywhere.

    ``torch.linalg.inv`` reads its factorization status back on the host to raise for a singular matrix, which on
    CUDA waits for every queued kernel. ``check_errors=False`` uses ``torch.linalg.inv_ex`` instead: the same
    factorization and the same bits, without the read. Pass it only for a matrix that cannot be singular, such as a
    pixel normalization kornia builds itself; a singular matrix then returns non-finite values instead of raising.
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)
    if _is_tracing_or_exporting() and _has_closed_form_inverse(input):
        return _closed_form_inverse(input.to(dtype)).to(input.dtype)
    if not check_errors:
        return inv_ex(input.to(dtype)).inverse.to(input.dtype)
    return torch.linalg.inv(input.to(dtype)).to(input.dtype)


def _torch_histc_cast(input: torch.Tensor, bins: int, min: Union[float, bool], max: Union[float, bool]) -> torch.Tensor:
    """Apply torch.histc to any real dtype, returning the counts in float32 or float64.

    The input is cast to float32 (float64 stays float64) before binning: ``torch.histc`` has no CPU
    kernel for integer input, and on MPS in torch 2.5.1 none for anything but float32.

    The counts are returned in that compute dtype, **not** cast back to the input dtype. A count grows
    with the number of values, not with their range: float16 holds integers exactly only up to 2048 and
    rounds them to ``inf`` from 65520, bfloat16 only up to 256, and an integer dtype can wrap (uint8 past
    255). They stay floating rather than becoming int64: ``torch.histc`` counts in the floating dtype it
    bins in, so a float32 count stops being exact past 2**24 per accumulating thread (single-threaded on
    the CPU, 2**25 + 3 equal values come back as 2**24), an int64 result would be no more exact, and a
    float64 input keeps float64 counts for float64 arithmetic downstream.
    """
    KORNIA_CHECK_IS_TENSOR(input, "Input must be torch.Tensor")
    dtype = _normalize_to_float32_or_float64(input.dtype)
    return torch.histc(input.to(dtype), bins, min, max)


def _torch_svd_cast(input: torch.Tensor, full_matrices: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd

    For numerical stability, fp32 inputs are promoted to fp64 (except on MPS where fp64 is unsupported).

    An MPS input past the shader-compilation ceiling is decomposed on the CPU and moved back, which
    is what keeps the batched minimal solvers behind ``RANSAC`` working on Apple silicon.
    ``full_matrices`` selects full or reduced factors, as in ``torch.linalg.svd``.
    """
    if is_mps_tensor_safe(input):
        dtype = torch.float32
    elif input.dtype == torch.float32:
        dtype = torch.float64
    else:
        dtype = _normalize_to_float32_or_float64(input.dtype)

    x = input.to(dtype)
    # torch 2.14's MPS ``linalg.svd`` raises "Failed to created pipeline state object" -- a Metal
    # shader compilation failure, not memory pressure -- once an input holds 8192 elements
    # or more, whatever the per-matrix shape. The bound is inclusive: 8192 is the first failing
    # size rather than the last working one, measured as ``(511, 4, 4)`` = 8176 passing and
    # ``(512, 4, 4)`` = 8192 raising. It is inlined rather than named because this function is
    # scripted (via ``zca_mean``) and TorchScript cannot close over a module global.
    if is_mps_tensor_safe(x) and x.numel() >= 8192:
        # SVD is per-matrix, so decomposing the whole batch on the CPU gives the same result; the
        # casts back to the MPS device keep the autograd graph intact.
        U, S, Vh = torch.linalg.svd(x.cpu(), full_matrices=full_matrices)
        out1, out2, out3H = U.to(x.device), S.to(x.device), Vh.to(x.device)
    else:
        out1, out2, out3H = torch.linalg.svd(x, full_matrices=full_matrices)
    # Since kornia requires torch>=2.5.1, we can always use .mH
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

    x = input.to(dtype)
    # ``svdvals`` shares the shader-compilation ceiling documented in
    # ``_torch_svd_cast``: on torch 2.14's MPS backend an input holding
    # 8192 elements or more fails to build its Metal pipeline state. This path
    # reaches ``solve_pnp_dlt``, which raised for any batch large enough to
    # cross it.
    if is_mps_tensor_safe(x) and x.numel() >= 8192:
        out = torch.linalg.svdvals(x.cpu()).to(x.device)
    else:
        # Since kornia requires torch>=2.5.1, we can always use torch.linalg.svdvals
        out = torch.linalg.svdvals(x)
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
    KORNIA_CHECK_IS_TENSOR(A, "A must be torch.Tensor")
    KORNIA_CHECK_IS_TENSOR(B, "B must be torch.Tensor")
    dtype = torch.float32 if is_mps_tensor_safe(A) else torch.float64

    out = torch.linalg.solve(A.to(dtype), B.to(dtype))

    # cast back to the input dtype
    return out.to(A.dtype)


def _rows_finite(x: torch.Tensor) -> torch.Tensor:
    """Whether every entry of each trailing matrix of ``x`` is finite, as a mask over its batch."""
    return x.isfinite().all(-1).all(-1)


def _is_singular(A: torch.Tensor) -> torch.Tensor:
    """Whether each 2x2, 3x3 or 4x4 matrix of ``A`` counts as singular, as a mask over its batch.

    One rule decides, the same in eager mode and under graph capture, on every torch version and device,
    and unchanged by scaling the matrix: a matrix is singular when its closed-form determinant is within
    the rounding error of that determinant,

    ``|det A| <= 8 * n * eps * perm |A|``,

    where ``perm |A|``, the permanent of the absolute values, is the sum of the absolute values of the terms
    of the determinant expansion, ``eps`` the machine epsilon of the dtype of ``A`` and ``n`` its order. The
    permanent, unlike ``||A||^n``, does not grow with a translation: a homography that moves by 5000 pixels
    has ``det 1`` and ``perm 1``. A matrix with a non-finite entry is not flagged; its inverse is not finite
    and the callers reject it on that account.

    Both sides are linear in every row and every column, so the rule is read on the matrix with each row,
    then each column, divided by its largest magnitude. That leaves the ratio unchanged and keeps the
    products of ``n`` entries from overflowing or underflowing: unscaled, ``1e10 * I`` of order 4 in float32
    reads ``inf <= inf`` and ``1e-13 * I`` reads ``0 <= 0``, and both would count as singular.

    Raises:
        NotImplementedError: for shapes other than ``(..., n, n)`` with ``n`` in 2, 3, 4.
    """
    if torch.jit.is_scripting():
        # ``torch.finfo`` does not script; the callers hand this float32 or float64 only.
        double = A.dtype == torch.float64
        eps = 2.220446049250313e-16 if double else 1.1920928955078125e-07
        tiny = 2.2250738585072014e-308 if double else 1.1754943508222875e-38
    else:
        eps = torch.finfo(A.dtype).eps
        tiny = torch.finfo(A.dtype).tiny
    A = A / A.abs().amax(-1, keepdim=True).clamp_min(tiny)
    A = A / A.abs().amax(-2, keepdim=True).clamp_min(tiny)
    det, perm = _det_perm_closed_form(A)
    return det.abs() <= (8 * A.shape[-1] * eps) * perm


def safe_solve_with_mask(B: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Solves the system of equations.

    Avoids crashing because of singular matrix input and outputs the mask of valid solution.

    A system is valid when ``A`` is not singular and the solution is finite in the dtype of ``B``. A 2x2,
    3x3 or 4x4 ``A`` is singular when ``|det A| <= 8 * n * eps * perm |A|``, the closed-form determinant
    against the permanent of the absolute values (:func:`_is_singular`), which is the same on every torch
    version, or when its LU factorization has a zero pivot; a larger ``A`` by the zero pivot alone. An
    invalid system is solved with the identity in place of ``A``, so its row of
    ``X`` holds ``B`` and its row of the returned LU factor is the identity's. The differentiated solve
    therefore never sees a singular matrix, and the gradient with respect to the valid systems (and to
    any parameter they share with an invalid one) stays finite.

    Args:
        B: right-hand side of shape :math:`(*, N, K)` or :math:`(*, N)`.
        A: square matrices of shape :math:`(*, N, N)`.

    Returns:
        The solution :math:`(*, N, K)`, the LU factor of the solved matrices :math:`(*, N, N)` and the
        validity mask :math:`(*)`.
    """
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    KORNIA_CHECK_IS_TENSOR(B, "B must be torch.Tensor")
    KORNIA_CHECK_IS_TENSOR(A, "A must be torch.Tensor")
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32

    n_dim_B = len(B.shape)
    n_dim_A = len(A.shape)
    if n_dim_A - n_dim_B == 1:
        B = B.unsqueeze(-1)

    A_cast = A.to(dtype)
    B_cast = B.to(dtype)

    # Decide validity on a detached pass, then solve a system whose invalid matrices are replaced by the
    # identity. Masking the output instead is not enough: the backward of ``lu_factor`` / ``lu_solve`` at
    # a singular matrix is non-finite, and ``0 * nan`` leaks it into every shared parameter.
    # Since kornia requires torch>=2.5.1, we can always use torch.linalg.lu_factor_ex and torch.linalg.lu_solve
    A_detached = A_cast.detach()
    LU_detached, pivots_detached, info = torch.linalg.lu_factor_ex(A_detached)
    X_detached = torch.linalg.lu_solve(LU_detached, pivots_detached, B_cast.detach())
    valid_mask: torch.Tensor = (info == 0) & _rows_finite(X_detached.to(B.dtype))
    if _has_closed_form_inverse(A):
        valid_mask = valid_mask & ~_is_singular(A_detached)

    eye = torch.eye(A_cast.shape[-1], device=A_cast.device, dtype=dtype)
    A_safe = torch.where(valid_mask[..., None, None], A_cast, eye)
    A_LU, pivots, _ = torch.linalg.lu_factor_ex(A_safe)
    X = torch.linalg.lu_solve(A_LU, pivots, B_cast)

    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def safe_inverse_with_mask(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Perform inverse.

    Avoids crashing because of non-invertable matrix input and outputs the mask of valid solution.

    A matrix is valid when it is not singular and its inverse is finite in the dtype of ``A``. A 2x2, 3x3
    or 4x4 matrix is singular when ``|det A| <= 8 * n * eps * perm |A|``, the closed-form determinant
    against the permanent of the absolute values (:func:`_is_singular`), which is the same in eager mode and
    under graph capture and on every torch version, or, in eager mode, when ``inv_ex`` reports a zero
    pivot; a larger matrix by the zero pivot alone. An invalid matrix is inverted as the
    identity, so its row of the output is the identity. The differentiated inverse therefore never sees a
    singular matrix, and the gradient with respect to the valid matrices (and to any parameter they
    share with an invalid one) stays finite.

    Args:
        A: square matrices of shape :math:`(*, N, N)`.

    Returns:
        The inverse :math:`(*, N, N)` and the validity mask :math:`(*)`.
    """
    KORNIA_CHECK_IS_TENSOR(A, "A must be torch.Tensor")

    dtype_original = A.dtype
    dtype = _normalize_to_float32_or_float64(dtype_original)
    A_cast = A.to(dtype)
    eye = torch.eye(A_cast.shape[-1], device=A_cast.device, dtype=dtype)

    # Decide validity on a detached pass, then invert a batch whose invalid matrices are replaced by the
    # identity. Masking the output instead is not enough: the backward of ``inv`` reuses its non-finite
    # output, and ``0 * nan`` leaks it into every shared parameter.
    A_detached = A_cast.detach()
    if _is_tracing_or_exporting() and _has_closed_form_inverse(A):
        # ``linalg_inv_ex`` has no ONNX lowering; the adjugate form is basic arithmetic. It is computed on a
        # scaled copy (:func:`_closed_form_inverse`), so a regular matrix at a scale whose adjugate or
        # determinant leaves the dtype keeps its finite inverse and its ``True`` mask, as in eager mode.
        mask = ~_is_singular(A_detached)
        inverse_detached = _closed_form_inverse(torch.where(mask[..., None, None], A_detached, eye))
        mask = mask & _rows_finite(inverse_detached.to(dtype_original))
        inverse = _closed_form_inverse(torch.where(mask[..., None, None], A_cast, eye))
        return inverse.to(dtype_original), mask

    inverse_detached, info = inv_ex(A_detached)
    mask = (info == 0) & _rows_finite(inverse_detached.to(dtype_original))
    if _has_closed_form_inverse(A):
        mask = mask & ~_is_singular(A_detached)
    inverse, _ = inv_ex(torch.where(mask[..., None, None], A_cast, eye))
    return inverse.to(dtype_original), mask


def is_autocast_enabled(both: bool = True) -> bool:
    """Check if torch autocast is enabled.

    Args:
        both: if True, consider the autocast regions of the CPU, CUDA, MPS and XPU device types.

    Returns:
        If ``both`` is True, whether autocast is enabled for any of the CPU, CUDA, MPS or XPU device types, on every
        supported torch version. If ``both`` is False, ``torch.is_autocast_enabled()`` without a device type, whose
        device type depends on the torch version: it never reports CPU autocast, and on torch 2.5.1 it does not
        report MPS autocast either.

    """
    if both:
        return any(torch.is_autocast_enabled(device_type) for device_type in ("cpu", "cuda", "mps", "xpu"))

    return torch.is_autocast_enabled()


def _autocast_disabled(device_type: str) -> AbstractContextManager[Any]:
    """Return a context that turns off the caller's autocast region for ``device_type``, if one is on.

    Outside autocast it returns a ``nullcontext``: entering ``torch.autocast(enabled=False)`` anyway costs a few
    microseconds per call, a few percent of a small solve, and raises for a device type without autocast, such as
    ``meta``. Every device type torch has an autocast region for is covered, ``xla``, ``hpu`` and a PrivateUse1 backend
    too.
    """
    # ``torch.amp.is_autocast_available`` would ask first, but Dynamo cannot trace it on torch 2.5.1. ``meta`` is
    # answered before asking: under torch.compile, Dynamo turns the RuntimeError below into its own internal error
    # instead of passing it to the ``except``.
    if device_type == "meta":
        return nullcontext()
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except RuntimeError:  # a device type without autocast, such as ``meta``
        enabled = False
    if enabled:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


# These helpers moved into ``torch.compiler`` over time; resolve them once at import.
_torch_is_compiling = getattr(torch.compiler, "is_compiling", None) or getattr(torch._dynamo, "is_compiling", None)
_torch_is_exporting = getattr(torch.compiler, "is_exporting", None)


def _in_torch_or_kornia(frame: FrameType) -> bool:
    """Whether ``frame`` runs code of torch or kornia, to attribute a warning to the first frame outside them."""
    return str(frame.f_globals.get("__name__", "")).split(".", 1)[0] in ("torch", "kornia")


def _warn_at_caller(message: str) -> None:
    """Warn with a ``UserWarning`` attributed to the first frame outside kornia and torch.

    For a warning raised where no fixed ``stacklevel`` names the user's line: inside a ``forward`` that
    ``nn.Module.__call__`` reaches through torch frames, or in code that other kornia classes call too.
    """
    level = 2  # warnings.warn's stacklevel for the frame that called this function
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None and _in_torch_or_kornia(frame):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, UserWarning, stacklevel=level if frame is not None else 2)


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
    # torch < 2.6 has no export flag, so ``is_compiling`` is the fallback: it is true under
    # ``torch.compile`` and ``torch.export`` alike. Newer releases differ - ``torch.compile`` can
    # report ``False`` for the export flag - which is why call sites must pick the predicate they
    # actually mean.
    return is_compiling()


def is_exporting() -> bool:
    """Whether execution is inside a graph capture by ``torch.export`` or the dynamo ONNX exporter.

    Used to switch to export-safe arithmetic (closed-form inverses, ``sort``-based medians, ...) and
    to skip in-``forward`` side effects (e.g. stashing per-call state on ``self``) that
    ``torch.export`` rejects, without changing the captured output. Always ``False`` inside
    TorchScript, so the guard is safe to call from scripted functions.

    The flag is export-specific, not capture-specific: that is what Torch reports from 2.12 on (CI
    checks 2.14), and it is the meaning kornia relies on. Releases before 2.12 also report it inside a
    ``torch.compile`` trace (2.9.1, and 2.5.1 through the fallback above), so a compiled call can take
    different paths across versions (#5037). The call sites that keep this predicate run their eager
    checks and routes inside ``torch.compile`` from 2.12 on, so some inputs of those operators do not
    compile with ``fullgraph=True`` there. It is false under ``torch.jit.trace`` and the legacy ONNX exporter. Where
    the intent is "inside any Dynamo capture" - skipping a data-dependent check that breaks
    ``fullgraph`` - use :func:`is_compiling`, which covers ``torch.compile`` and ``torch.export`` alike
    on every supported release. Keep this predicate for what only an export needs:
    :func:`crop_by_indices` routing to its dedicated export implementation, where letting a compiled
    call take that route changes the sampling the eager and compiled paths agree on, and skipping
    per-call state on ``self`` that ``torch.export`` rejects but a compiled call records and a caller
    reads back afterwards, such as the ``transform_matrix`` of an augmentation.
    """
    if torch.jit.is_scripting():
        return False
    return _is_exporting_eager()


def register_module_state(module: torch.nn.Module, name: str, x: torch.Tensor) -> None:
    """Store tensor ``x`` on ``module`` as ``name`` so it is optimizable, movable and serializable.

    ``x`` must be a floating-point or complex tensor, or an existing ``nn.Parameter``: an integer or bool tensor can
    neither require grad nor follow the module's dtype conversions, so it raises ``TypeCheckError`` naming its dtype.
    An existing ``nn.Parameter`` is kept, and a tensor that takes no part in autograd (no ``grad_fn`` and
    ``requires_grad=False``) becomes an ``nn.Parameter``, so the module is optimizable. ``nn.Parameter(x)`` would
    re-root any other tensor as a new leaf: a group built from ``Se3.exp(v)`` would stop propagating gradients to
    ``v``, and a group built from the caller's own leaf that requires grad would take its gradient away from that
    leaf. Such a tensor is registered as a buffer instead, which keeps it and its history while ``.to()``,
    ``state_dict()`` and ``load_state_dict()`` still reach it under the same key. Under graph capture
    (``torch.jit.trace``, ``torch.compile``, ``torch.export`` and the dynamo ONNX exporter) neither a parameter nor a
    buffer can be created inside the traced region, so the tensor is kept as a plain attribute of the module being
    built.

    Raises:
        TypeCheckError: if ``x`` is neither a floating-point nor a complex tensor, nor an ``nn.Parameter``.

    """
    if not (isinstance(x, torch.nn.Parameter) or x.is_floating_point() or x.is_complex()):
        raise TypeCheckError(
            f"{type(module).__name__} stores {name.lstrip('_')!r} as module state, which needs a floating-point or "
            f"complex tensor; got {x.dtype}. Cast it first, for example with `.float()`.",
            actual_type=x.dtype,
        )
    if isinstance(x, torch.nn.Parameter) or not (torch.jit.is_tracing() or is_compiling() or is_exporting()):
        if isinstance(x, torch.nn.Parameter) or (x.grad_fn is None and not x.requires_grad):
            x = x if isinstance(x, torch.nn.Parameter) else torch.nn.Parameter(x)
        else:
            module.register_buffer(name, x)
            return
    setattr(module, name, x)


def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dictionaries.

    A dataclass becomes a dict of its fields through :func:`dataclasses.asdict`, and lists, tuples,
    namedtuples and dicts are rebuilt with their items converted; any other value is returned as it is.
    Values reached through a dataclass are deep-copied by :func:`dataclasses.asdict`, tensors included,
    so a tensor with autograd history (not a leaf) raises ``RuntimeError``. Outside any dataclass the
    containers are rebuilt, but their items are not copied.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {key: dataclass_to_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # a namedtuple's constructor takes one argument per field, so expand positionally
        return type(obj)(*(dataclass_to_dict(item) for item in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(dataclass_to_dict(item) for item in obj)
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj


T = TypeVar("T")


def _is_sequence_of(value: Any, origin: type) -> bool:
    """Whether ``value`` is a list or tuple that ``dict_to_dataclass`` converts for a sequence annotation."""
    return type(value) in (list, tuple) and issubclass(origin, Sequence) and not issubclass(origin, (str, bytes))


def _fits_annotation(value: Any, annotation: Any) -> bool:
    """Whether ``value`` can be held by ``annotation``, as ``dict_to_dataclass`` converts it."""
    target = get_origin(annotation) or annotation
    if annotation is Any or not isinstance(target, type):
        return True  # Any, a TypeVar, a Literal, ...: the value cannot be ruled out
    if is_dataclass(target):
        return isinstance(value, (dict, target))
    try:
        return _is_sequence_of(value, target) or isinstance(value, target)
    except TypeError:  # a class that rejects isinstance, such as a Protocol that is not runtime-checkable
        return True


def _value_for_annotation(value: Any, annotation: Any) -> Any:
    """Return ``value`` with each dict that ``annotation`` types as a dataclass converted to that dataclass."""
    if isinstance(annotation, type) and is_dataclass(annotation):
        return value if isinstance(value, annotation) else dict_to_dataclass(value, annotation)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union or origin is UnionType:
        # Optional[X] and other unions: follow the one member that can hold the value, if there is exactly one
        fitting = [arg for arg in args if _fits_annotation(value, arg)]
        return _value_for_annotation(value, fitting[0]) if len(fitting) == 1 else value
    if not (isinstance(origin, type) and args):
        return value
    if _is_sequence_of(value, origin):
        if origin is tuple and args[-1] is not Ellipsis:  # Tuple[X, Y]: one annotation per position
            if len(args) != len(value):
                return value
            return type(value)(_value_for_annotation(item, arg) for item, arg in zip(value, args))
        return type(value)(_value_for_annotation(item, args[0]) for item in value)
    if type(value) is dict and issubclass(origin, Mapping) and len(args) == 2:
        return {key: _value_for_annotation(item, args[1]) for key, item in value.items()}
    return value


def dict_to_dataclass(dict_obj: Dict[str, Any], dataclass_type: Type[T]) -> T:
    """Recursively convert dictionaries to dataclass instances.

    A field annotated with a dataclass gets its dict value converted to that dataclass, and so does a
    dataclass inside the field's ``Optional[...]`` or other union, the elements of a plain ``list`` or ``tuple``
    value under a sequence annotation, or the values of a plain ``dict`` value under a mapping annotation (a
    namedtuple or a ``dict`` subclass is kept whole). String annotations, including every annotation
    in a module with ``from __future__ import annotations``, are resolved with :func:`typing.get_type_hints`.
    A value that is already an instance of the annotated dataclass is kept as it is, and values of other
    fields are passed on unchanged.

    A union is followed only when exactly one of its members can hold the value: a dict for a
    ``Union[A, B]`` of two dataclasses, or for ``Union[A, dict]``, stays a dict. When the annotations
    cannot be resolved, for example a string that names a class defined inside a function, they are used
    as written, and a string annotation converts nothing.

    Args:
        dict_obj: the field values, keyed by field name.
        dataclass_type: the dataclass to build.

    Returns:
        An instance of ``dataclass_type``.

    Raises:
        TypeCheckError: if ``dict_obj`` is not a dict, or a value annotated with a dataclass outside a union
            is neither a dict nor an instance of that dataclass.
        BaseError: if ``dataclass_type`` is not a dataclass.

    """
    KORNIA_CHECK_TYPE(dict_obj, dict, "Input conf must be dict")
    KORNIA_CHECK(is_dataclass(dataclass_type), "dataclass_type must be a dataclass")
    try:
        hints = get_type_hints(dataclass_type)
    except (AttributeError, NameError, SyntaxError, TypeError):
        hints = {}
    field_types: dict[str, Any] = {f.name: hints.get(f.name, f.type) for f in fields(dataclass_type)}
    constructor_args = {
        key: _value_for_annotation(value, field_types[key]) if key in field_types else value
        for key, value in dict_obj.items()
    }
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
    KORNIA_CHECK(batch_size > 0, lambda: f"batch_size must be positive, got {batch_size}")
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
