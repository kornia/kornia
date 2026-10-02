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

from typing import List, Optional

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_SHAPE, KORNIA_CHECK_SHAPE
from kornia.core.exceptions import ValueCheckError

__all__ = ["bottom_hat", "closing", "dilation", "erosion", "gradient", "opening", "reconstruction", "top_hat"]


def _validate_morphology_inputs(
    tensor: torch.Tensor, kernel: torch.Tensor, structuring_element: Optional[torch.Tensor], border_type: str
) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"Input image must have a floating-point dtype. Got {tensor.dtype}")
    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")
    if structuring_element is not None and structuring_element.shape != kernel.shape:
        raise ValueError(
            f"`structuring_element` shape must match `kernel` shape. "
            f"Got {structuring_element.shape} and {kernel.shape}."
        )
    if border_type not in ["geodesic", "constant", "reflect", "replicate", "circular"]:
        raise ValueError(
            f"Unknown `border_type`: {border_type}. "
            "Expected one of ['geodesic', 'constant', 'reflect', 'replicate', 'circular']."
        )


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


@torch.jit.unused
def _can_reduce_in_place(padded: torch.Tensor, offsets: torch.Tensor) -> bool:
    if torch.jit.is_tracing() or torch._C._are_functorch_transforms_active():
        return False
    return all(torch.autograd.forward_ad.unpack_dual(t).tangent is None for t in (padded, offsets))


def _shift_cell(
    planes: torch.Tensor,
    plane: torch.Tensor,
    offsets: torch.Tensor,
    i: int,
    j: int,
    height: int,
    width: int,
    flat: bool,
) -> torch.Tensor:
    """One shifted view of the ``shift`` engine, read from the identity or the image plane of ``planes``."""
    source = planes[..., i : i + height, j : j + width].index_select(0, plane[i, j].view(1))[0]
    if flat:
        return source
    # Two-dimensional offset so PyTorch applies tensor-tensor dtype promotion.
    return source + offsets[i : i + 1, j : j + 1]


def _shift_reduce(
    padded: torch.Tensor,
    offsets: torch.Tensor,
    kernel: torch.Tensor,
    height: int,
    width: int,
    flat: bool,
    dilate: bool,
    inplace: bool,
    reduction_value: Optional[float],
) -> torch.Tensor:
    """Running max (``dilate``) or min over the ``k_h * k_w`` shifted views of ``padded`` plus their offsets.

    The ``shift`` engine: output pixel ``(y, x)`` reduces ``padded[y + i, x + j] + offsets[i, j]`` over the
    kernel positions, the same max-plus expression ``unfold`` evaluates, without materialising the window
    tensor. ``torch.compile`` fuses the loop. A cell where ``kernel`` is zero takes ``reduction_value``, the
    reduction identity; ``None`` leaves the offset unchanged.
    ``flat`` says that every offset is zero (a floating-point image and no ``structuring_element``), so the
    addition is skipped; a ``-0.0`` pixel then stays ``-0.0``, where adding a ``+0.0`` offset made it ``+0.0``.

    Forward-only, a floating-point image reads each cell from a two-plane buffer, plane 0 filled with the
    identity and plane 1 holding ``padded`` in the compute dtype, through ``index_select`` on
    ``kernel[i, j] != 0``. That is a plain copy per cell, where a ``masked_fill_`` or ``where`` with a
    broadcast scalar mask is not vectorised on CPU and cost three adds per cell; the select is exact for
    every input, ``nan`` and infinities included, and stays data-independent, so the loop compiles and
    exports as before. Under autograd the excluded cells are masked instead (see the note in the body).

    Forward-only this holds that buffer, one output-sized intermediate and one shifted view whatever the
    kernel area, so peak memory is flat in ``k_h k_w`` where ``unfold`` and ``convolution`` grow with it (at
    15 x 15, B=8 x 3 x 256^2 float32 on CPU, 69 MiB of peak RSS growth against 1380 MiB for ``unfold``).
    Under autograd the opposite is true: every ``torch.maximum`` saves both operands, so ``2 (k_h k_w - 1)``
    output-sized tensors stay alive until backward (2717 MiB against 1627 MiB for ``unfold`` in that cell on
    CUDA). :func:`_resolve_engine` accounts for both regimes.
    """
    kh, kw = offsets.shape
    # The two-plane select is a forward-only device: ``index_select`` backpropagates by zero-filling the
    # whole buffer and scattering into it once per cell, which made forward + backward 1.2 to 1.4x slower
    # than masking (CPU float32, 3 x 3 and 7 x 7). While a backward graph is being recorded, or for a
    # a call without a reduction value, the masked form is kept.
    recording = torch.is_grad_enabled() and (padded.requires_grad or offsets.requires_grad)
    if reduction_value is None or recording:
        # Keep each offset two-dimensional so PyTorch applies tensor-tensor dtype promotion. Indexing
        # down to a scalar would instead apply wrapped-scalar rules and silently keep ``padded.dtype``.
        output = padded[..., 0:height, 0:width] + offsets[0:1, 0:1]
        if reduction_value is not None:
            output = torch.where(kernel[0, 0] != 0, output, torch.full_like(output, reduction_value))
        # ``unfold`` reduces the kernel-height dimension first and then kernel width; keep that traversal so
        # tied values are met in the same order. Which of two tied operands a backend's ``max``/``min``
        # returns is its own choice (CPU keeps the first, MPS the second), so the sign of a zero result is
        # not preserved between engines or devices; the value is.
        for j in range(kw):
            for i in range(kh):
                if i == 0 and j == 0:
                    continue
                shifted = padded[..., i : i + height, j : j + width] + offsets[i : i + 1, j : j + 1]
                if reduction_value is not None:
                    shifted.masked_fill_(kernel[i, j] == 0, reduction_value)
                if inplace:
                    torch.maximum(output, shifted, out=output) if dilate else torch.minimum(output, shifted, out=output)
                else:
                    output = torch.maximum(output, shifted) if dilate else torch.minimum(output, shifted)
        return output

    # Out-of-place on purpose: ``vmap`` and forward-mode AD batch or dualise ``padded``, which an in-place
    # copy into a fresh buffer could not receive.
    compute_dtype = torch.promote_types(padded.dtype, offsets.dtype)
    padded = padded.to(dtype=compute_dtype)
    planes = torch.stack((torch.full_like(padded, reduction_value), padded))
    # 1 selects the image plane, 0 the identity plane. An excluded cell then adds its offset to the identity,
    # so zero it there: an infinite structuring element entry would otherwise turn the identity into nan.
    included = kernel != 0
    plane = included.to(torch.long)
    offsets = torch.where(included, offsets, torch.zeros_like(offsets))

    output = _shift_cell(planes, plane, offsets, 0, 0, height, width, flat)
    # Same traversal order as the masked loop above and as ``unfold``.
    for j in range(kw):
        for i in range(kh):
            if i == 0 and j == 0:
                continue
            shifted = _shift_cell(planes, plane, offsets, i, j, height, width, flat)
            if inplace:
                torch.maximum(output, shifted, out=output) if dilate else torch.minimum(output, shifted, out=output)
            else:
                output = torch.maximum(output, shifted) if dilate else torch.minimum(output, shifted)
    return output


def _resolve_engine(
    engine: str, tensor: torch.Tensor, recording_grad: bool = False, dtype: Optional[torch.dtype] = None
) -> str:
    """Map ``engine="auto"`` to the preferred engine for ``tensor``; leave other values unchanged.

    ``recording_grad`` says whether this call will build a backward graph, which changes the ranking.
    ``dtype`` is the dtype the max-plus terms are computed in; it defaults to ``tensor.dtype`` and is
    wider when the kernel or the structuring element promotes the computation, which is what the
    dtype rule below has to see. For finite inputs, the two engines selected by ``auto`` (``unfold``
    and ``shift``) return equal forward output, so switching between them never changes a value; only
    the sign of a zero can differ, because a backend's ``max``/``min`` may return either tied operand, and
    because forward-only ``shift`` skips adding the zero offsets of a call without ``structuring_element``,
    which turn a ``-0.0`` pixel into ``+0.0`` in the ``unfold`` dilation.

    Benchmarks in :mod:`benchmarks.morphology.engines` (x86 CPU, an RTX 4090 and an Apple M1,
    ``dilation``, B x 3 x 256 x 256) give three CPU/CUDA regimes:

    - CUDA: ``unfold`` is the broadly faster engine forward (23 of 24 cells) and forward + backward
      (21 of 24), so it is always the CUDA choice.
    - CPU without a backward graph: ``shift`` wins every cell, by 4-13x in float32 and 12-20x in half
      precision, and its peak memory is flat in the kernel area instead of growing with it.
    - CPU with a backward graph: the running max saves ``2 (k_h k_w - 1)`` intermediates, so ``shift``
      loses to ``unfold`` in float32 (up to 2.5x at 15 x 15) and float64 (up to 3.4x), while still
      winning 11 of 12 half-precision cells.

    MPS keeps ``shift`` in both cases, measured on an Apple M1: ``unfold`` is not the fastest engine
    in any of the 16 forward + backward cells, and ``shift`` beats it there by 2.6-10x, so the CPU
    float32 grad branch deliberately does not extend to Metal. ``convolution`` is faster than
    ``shift`` at small kernels on MPS but collapses at 15 x 15 (2854 ms against 354 ms at B=8), which
    is why it is not the choice either.

    The grad branch is a backward-pass speed rule, not a derivative-preserving one: forward-mode AD
    (``torch.func.jvp``, ``torch.autograd.forward_ad``) records no backward graph, so it takes
    ``shift`` off CUDA like any forward-only call, and its tangent at tied maxima is the ``shift``
    one.
    """
    if engine == "auto":
        if tensor.device.type == "cuda":
            return "unfold"
        if dtype is None:
            dtype = tensor.dtype
        is_float32_or_64 = dtype in (torch.float32, torch.float64)
        if tensor.device.type == "cpu" and recording_grad and is_float32_or_64:
            return "unfold"
        return "shift"
    return engine


def _records_grad(tensor: torch.Tensor, structuring_element: Optional[torch.Tensor]) -> bool:
    """Whether an op over these inputs will build a backward graph.

    ``torch.no_grad()`` leaves ``requires_grad`` set on the inputs but records nothing, so grad mode has
    to be checked too: inference on a tensor that happens to require grad should take the forward-only
    engine. The structuring element counts because ``dilation`` and ``erosion`` differentiate through
    the neighborhood they build from it. The kernel does not: it enters only through the ``kernel == 0``
    mask, so a kernel that requires grad still produces an output that does not. Forward-mode AD
    (``torch.func.jvp``) is invisible here by design; see :func:`_resolve_engine`.
    """
    if not torch.is_grad_enabled():
        return False
    if tensor.requires_grad:
        return True
    return structuring_element is not None and structuring_element.requires_grad


def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the dilated image applying the same kernel in each channel.

    .. image:: _static/img/dilation.png

    The kernel must have 2 dimensions.

    Convention:
        - ``dilation`` reflects the structuring element: it is the Minkowski dilation
          :math:`\delta_B f(x) = \max_{b \in B} f(x - b)`, as in ``scipy.ndimage``; :func:`erosion` does not
          reflect.
        - ``border_type="geodesic"`` ignores the pixels outside the image; the other modes carry torch's pad
          names. :doc:`Conventions & Pitfalls </get-started/conventions>` maps both, and the kernel
          conventions, onto scipy, scikit-image and OpenCV.
        - Under ``geodesic`` a window with no kernel cell inside the image is empty and returns the reduction
          identity, ``-inf`` here and ``+inf`` in :func:`erosion`, as scipy and scikit-image do; so does every
          window of a kernel with no non-zero cell. The composite operations inherit these infinities.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
        Dilated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    _validate_morphology_inputs(tensor, kernel, structuring_element, border_type)

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask, so a bool or integer kernel does not set the compute dtype: a
        # floating-point image lends it its own. A float kernel keeps its dtype, which may widen the result.
        nb_dtype = tensor.dtype if not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
    else:
        neighborhood = structuring_element.clone()

    # The max-plus terms compute in the promoted dtype, which the dtype rule of ``auto`` has to see: a
    # float16 image with a float32 kernel computes, and differentiates, in float32.
    compute_dtype = torch.promote_types(tensor.dtype, neighborhood.dtype)
    recording_grad = _records_grad(tensor, structuring_element)
    engine = _resolve_engine(engine, tensor, recording_grad, compute_dtype)

    # pad
    # The kernel is reflected below (Minkowski dilation), so the window is anchored at the reflected origin.
    pad_e: List[int] = [se_w - origin[1] - 1, origin[1], se_h - origin[0] - 1, origin[0]]
    is_geodesic = border_type == "geodesic"
    if border_type == "geodesic":
        output: torch.Tensor = F.pad(tensor, pad_e, mode="constant", value=-float("inf"))
    elif border_type == "constant":
        output = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    reduction_min: float = -float("inf")
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output = output + neighborhood.flip((0, 1))
        output.masked_fill_(
            kernel.flip((0, 1)).view(1, 1, 1, 1, se_h, se_w) == 0,
            reduction_min,
        )
        output, _ = torch.max(output, 4)
        output, _ = torch.max(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        output = output.to(dtype=compute_dtype)
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=compute_dtype)
        conv_neighborhood = neighborhood.masked_fill(kernel == 0, 0.0)

        # ``conv2d`` multiplies every window cell by the one-hot weight, and ``inf * 0`` or ``nan * 0`` would
        # spread to the whole window. Convolve zeros in their place, then route a code for each non-finite
        # value (1: +inf, 2: -inf, 3: nan) through the same one-hot weight, which reproduces it exactly.
        special = torch.zeros_like(output)
        if output.is_floating_point():
            special = special.masked_fill(torch.isposinf(output), 1.0)
            special = special.masked_fill(torch.isneginf(output), 2.0)
            special = special.masked_fill(torch.isnan(output), 3.0)
            output = output.masked_fill(special != 0, 0.0)
            special = F.conv2d(special.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0)
        output = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad),
            reshape_kernel,
            padding=0,
            bias=conv_neighborhood.view(-1).flip(0).to(dtype=compute_dtype),
        )

        if output.is_floating_point():
            output = output.masked_fill(special == 1, float("inf"))
            output = output.masked_fill(special == 2, -float("inf"))
            output = output.masked_fill(special == 3, float("nan"))

        output = output.masked_fill(
            kernel.view(-1).flip(0).view(1, -1, 1, 1) == 0,
            reduction_min,
        )

        if is_geodesic:
            valid = torch.ones((1, 1, H, W), dtype=output.dtype, device=output.device)
            valid = F.pad(valid, pad_e, mode="constant", value=0.0)
            valid = F.conv2d(valid, reshape_kernel, padding=0)
            output = output.masked_fill(valid == 0, reduction_min)

        output = output.max(dim=1).values
        output = output.view(B, C, H, W)
    elif engine == "shift":
        offsets = neighborhood.flip((0, 1))
        inplace = False
        if not torch.jit.is_scripting():
            inplace = not recording_grad and _can_reduce_in_place(output, offsets)
        output = _shift_reduce(
            output,
            offsets,
            kernel.flip((0, 1)),
            tensor.shape[-2],
            tensor.shape[-1],
            structuring_element is None,
            True,
            inplace,
            reduction_min,
        )
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'auto', 'convolution', 'shift' or 'unfold'")
    return output.view_as(tensor)


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the eroded image applying the same kernel in each channel.

    .. image:: _static/img/erosion.png

    The kernel must have 2 dimensions.

    Convention:
        As :func:`dilation`, except that ``erosion`` does **not** reflect the structuring element: it is the
        Minkowski erosion :math:`\varepsilon_B f(x) = \min_{b \in B} f(x + b)`, as in ``scipy.ndimage`` and
        OpenCV. ``structuring_element`` is subtracted before the minimum.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
        Eroded image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    _validate_morphology_inputs(tensor, kernel, structuring_element, border_type)

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask, so a bool or integer kernel does not set the compute dtype: a
        # floating-point image lends it its own. A float kernel keeps its dtype, which may widen the result.
        nb_dtype = tensor.dtype if not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
    else:
        neighborhood = structuring_element.clone()

    # The max-plus terms compute in the promoted dtype, which the dtype rule of ``auto`` has to see: a
    # float16 image with a float32 kernel computes, and differentiates, in float32.
    compute_dtype = torch.promote_types(tensor.dtype, neighborhood.dtype)
    recording_grad = _records_grad(tensor, structuring_element)
    engine = _resolve_engine(engine, tensor, recording_grad, compute_dtype)

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    is_geodesic = border_type == "geodesic"
    if border_type == "geodesic":
        output: torch.Tensor = F.pad(tensor, pad_e, mode="constant", value=float("inf"))
    elif border_type == "constant":
        output = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    reduction_max: float = float("inf")
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output = output - neighborhood
        output.masked_fill_(
            kernel.view(1, 1, 1, 1, se_h, se_w) == 0,
            reduction_max,
        )
        output, _ = torch.min(output, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        output = output.to(dtype=compute_dtype)
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=compute_dtype)
        conv_neighborhood = neighborhood.masked_fill(kernel == 0, 0.0)

        # ``conv2d`` multiplies every window cell by the one-hot weight, and ``inf * 0`` or ``nan * 0`` would
        # spread to the whole window. Convolve zeros in their place, then route a code for each non-finite
        # value (1: +inf, 2: -inf, 3: nan) through the same one-hot weight, which reproduces it exactly.
        special = torch.zeros_like(output)
        if output.is_floating_point():
            special = special.masked_fill(torch.isposinf(output), 1.0)
            special = special.masked_fill(torch.isneginf(output), 2.0)
            special = special.masked_fill(torch.isnan(output), 3.0)
            output = output.masked_fill(special != 0, 0.0)
            special = F.conv2d(special.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0)
        output = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad),
            reshape_kernel,
            padding=0,
            bias=-conv_neighborhood.view(-1).to(dtype=compute_dtype),
        )

        if output.is_floating_point():
            output = output.masked_fill(special == 1, float("inf"))
            output = output.masked_fill(special == 2, -float("inf"))
            output = output.masked_fill(special == 3, float("nan"))

        output = output.masked_fill(
            kernel.view(-1).view(1, -1, 1, 1) == 0,
            reduction_max,
        )

        if is_geodesic:
            valid = torch.ones((1, 1, H, W), dtype=output.dtype, device=output.device)
            valid = F.pad(valid, pad_e, mode="constant", value=0.0)
            valid = F.conv2d(valid, reshape_kernel, padding=0)
            output = output.masked_fill(valid == 0, reduction_max)

        output = output.min(dim=1).values
        output = output.view(B, C, H, W)
    elif engine == "shift":
        offsets = -neighborhood
        inplace = False
        if not torch.jit.is_scripting():
            inplace = not recording_grad and _can_reduce_in_place(output, offsets)
        output = _shift_reduce(
            output,
            offsets,
            kernel,
            tensor.shape[-2],
            tensor.shape[-1],
            structuring_element is None,
            False,
            inplace,
            reduction_max,
        )
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'auto', 'convolution', 'shift' or 'unfold'")

    return output


def opening(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the opened image, (that means, dilation after an erosion) applying the same kernel in each channel.

    .. image:: _static/img/opening.png

    The kernel must have 2 dimensions.

    Convention:
        ``opening`` is ``dilation(erosion(tensor))`` with the same arguments in both halves. As only
        :func:`dilation` reflects the kernel, it is a morphological opening (anti-extensive and idempotent) for
        an asymmetric kernel too, under ``geodesic`` or ``circular``; a non-flat ``structuring_element`` or
        ``engine="convolution"`` holds both properties only to roundoff, and ``constant``, ``reflect`` and
        ``replicate`` can break them.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Opened image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> opened_img = opening(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return dilation(
        erosion(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def closing(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the closed image, (that means, erosion after a dilation) applying the same kernel in each channel.

    .. image:: _static/img/closing.png

    The kernel must have 2 dimensions.

    Convention:
        ``closing`` is ``erosion(dilation(tensor))`` with the same arguments in both halves. As only
        :func:`dilation` reflects the kernel, it is a morphological closing (extensive and idempotent) for an
        asymmetric kernel too, under ``geodesic`` or ``circular``; a non-flat ``structuring_element`` or
        ``engine="convolution"`` holds both properties only to roundoff, and ``constant``, ``reflect`` and
        ``replicate`` can break them.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Closed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> closed_img = closing(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return erosion(
        dilation(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        ),
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


# Morphological Gradient
def gradient(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the morphological gradient of an image.

    .. image:: _static/img/gradient.png

    That means, (dilation - erosion) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    Convention:
        ``gradient`` is ``dilation(tensor) - erosion(tensor)`` with the same arguments; see :func:`dilation`.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Gradient image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> gradient_img = gradient(tensor, kernel)

    """
    return dilation(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    ) - erosion(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def top_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the top hat transformation of an image.

    .. image:: _static/img/top_hat.png

    That means, (image - opened_image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :func:`~kornia.morphology.opening` for details.

    Convention:
        ``top_hat`` is ``tensor - opening(tensor)`` with the same arguments.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Top hat transformed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> top_hat_img = top_hat(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return tensor - opening(
        tensor,
        kernel=kernel,
        structuring_element=structuring_element,
        origin=origin,
        border_type=border_type,
        border_value=border_value,
        max_val=max_val,
        engine=engine,
    )


def bottom_hat(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = "geodesic",
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the bottom hat transformation of an image.

    .. image:: _static/img/bottom_hat.png

    That means, (closed_image - image) applying the same kernel in each channel.
    The kernel must have 2 dimensions.

    See :func:`~kornia.morphology.closing` for details.

    Convention:
        ``bottom_hat`` is ``closing(tensor) - tensor`` with the same arguments.

    Args:
        tensor: Floating-point image with shape :math:`(B, C, H, W)`; any other dtype raises a ``TypeError``.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the ``origin`` cell over which the operation is applied, and their
            magnitude is ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s, and any other shape raises a ``ValueError``.
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise an error
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``; under ``geodesic`` the pixels outside the image are excluded instead, and under
            ``reflect``, ``replicate`` and ``circular`` it is ignored.
        max_val: No effect; excluded kernel cells and ``geodesic`` padding use the reduction identity
            :math:`\mp\infty`. Retained for backward compatibility.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand, and because without a ``structuring_element`` forward-only ``"shift"`` skips adding the
            zero offsets, which turn a ``-0.0`` pixel into ``+0.0`` in :func:`dilation` with ``"unfold"``.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype of ``structuring_element``,
            else of ``kernel``) and records a backward graph takes ``"unfold"``, where ``"shift"`` is slower.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`2 (k_h k_w - 1)` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Bottom hat transformed image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> bottom_hat_img = bottom_hat(tensor, kernel)

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    return (
        closing(
            tensor,
            kernel=kernel,
            structuring_element=structuring_element,
            origin=origin,
            border_type=border_type,
            border_value=border_value,
            max_val=max_val,
            engine=engine,
        )
        - tensor
    )


def _reconstruct_up(
    seed: torch.Tensor,
    mask: torch.Tensor,
    kernel: torch.Tensor,
    num_iters: Optional[int],
    check_every: int,
    engine: str,
) -> torch.Tensor:
    output = torch.minimum(seed, mask)

    if num_iters is not None:
        for _ in range(num_iters):
            output = torch.minimum(dilation(output, kernel, engine=engine), mask)
        return output

    # Each step can only raise a pixel, so the loop has converged once a batch of steps raises none.
    changed = True
    while changed:
        previous = output
        for _ in range(check_every):
            output = torch.minimum(dilation(output, kernel, engine=engine), mask)
        changed = bool((output > previous).any())

    return output


def reconstruction(
    seed: torch.Tensor,
    mask: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    method: str = "dilation",
    num_iters: Optional[int] = None,
    check_every: int = 4,
    engine: str = "auto",
) -> torch.Tensor:
    r"""Return the morphological reconstruction of ``seed`` bounded by ``mask``, applied to each channel.

    Reconstruction by dilation repeats the geodesic dilation :math:`\min(\delta_B(x), \text{mask})`, starting
    from :math:`\min(\text{seed}, \text{mask})`, until the result stops changing. Reconstruction by erosion is
    its dual, :math:`-\text{reconstruction}(-\text{seed}, -\text{mask})`. See L. Vincent, "Morphological
    grayscale reconstruction in image analysis: applications and efficient algorithms", IEEE TIP 1993.
    Under autograd every step keeps its intermediates, so memory grows with the number of steps.

    Convention:
        Matches ``skimage.morphology.reconstruction``: the default ``kernel`` is a :math:`3 \times 3` square,
        its center cell is always part of the neighborhood, and both methods spread a pixel's value to the
        kernel's offsets, as :func:`dilation` does. scikit-image's ``dilation`` spreads the other way, but its
        ``reconstruction`` agrees. The image border is ``geodesic``. Unlike scikit-image, a ``seed`` above
        ``mask`` (below it for ``"erosion"``) is clipped to ``mask`` instead of raising.

    Args:
        seed: Floating-point starting image with shape :math:`(B, C, H, W)`.
        mask: Floating-point image bounding the reconstruction, with the same shape as ``seed``.
        kernel: Offsets from the center that a pixel's value spreads to in one step, with shape
            :math:`(k_h, k_w)` and odd sizes. Non-zero cells mark an offset; their magnitude and dtype are
            ignored. Default: ``None``, which uses a :math:`3 \times 3` square.
        method: ``"dilation"`` (default) or ``"erosion"``.
        num_iters: Number of steps to run. Default: ``None``, which runs until the result stops changing.
            That can take far more steps than :math:`\max(H, W)` when ``mask`` has winding paths. A fixed
            count always runs that many steps, even past convergence. It has no data-dependent exit, so
            ``torch.compile`` captures it as one graph, but the loop is unrolled and compile time grows with the
            count.
        check_every: Number of steps between convergence checks when ``num_iters`` is ``None``. Each check syncs
            the device with the host, and steps past convergence are wasted, so a larger value suits inputs that
            take many steps. For inputs without NaN, the output does not depend on it. Default: ``4``.
        engine: ``"unfold"``, ``"shift"`` or ``"auto"`` (default), passed to :func:`dilation`. ``"convolution"``
            is rejected: it is not exact on every backend, and an inexact step can keep the loop from converging.

    Returns:
        Reconstructed image with shape :math:`(B, C, H, W)` and the promoted dtype of ``seed`` and ``mask``.

    Raises:
        TypeCheckError: if ``seed``, ``mask`` or ``kernel`` is not a tensor.
        ShapeError: if ``seed`` is not 4-dimensional, ``mask`` has another shape, or ``kernel`` is not
            2-dimensional.
        ValueCheckError: if ``engine`` is ``"convolution"``, also with checks disabled.
        BaseError: if ``seed`` or ``mask`` is not floating point, a ``kernel`` size is even, ``method`` or
            ``engine`` is not one of the values above, ``num_iters`` is negative, or ``check_every`` is not
            positive.

    Example:
        >>> mask = torch.rand(1, 3, 5, 5)
        >>> seed = mask * 0.5
        >>> output = reconstruction(seed, mask)

    """
    KORNIA_CHECK_IS_TENSOR(seed)
    KORNIA_CHECK_IS_TENSOR(mask)
    KORNIA_CHECK_SHAPE(seed, ["B", "C", "H", "W"])
    KORNIA_CHECK_SAME_SHAPE(seed, mask)
    KORNIA_CHECK(seed.is_floating_point(), f"`seed` must have a floating-point dtype. Got {seed.dtype}.")
    KORNIA_CHECK(mask.is_floating_point(), f"`mask` must have a floating-point dtype. Got {mask.dtype}.")
    KORNIA_CHECK(method in ["dilation", "erosion"], f"Unknown `method`: {method}. Expected 'dilation' or 'erosion'.")
    # Not a KORNIA_CHECK, which `disable_checks()`, `KORNIA_CHECKS=0` and `python -O` turn off: an inexact
    # `conv2d` step makes the convergence loop oscillate forever (macOS CPU float32), not return a wrong value.
    if engine == "convolution":
        raise ValueCheckError("Unsupported `engine`: convolution. Expected one of ['auto', 'unfold', 'shift'].")
    KORNIA_CHECK(
        engine in ["auto", "unfold", "shift"],
        f"Unsupported `engine`: {engine}. Expected one of ['auto', 'unfold', 'shift'].",
    )
    KORNIA_CHECK(num_iters is None or num_iters >= 0, f"`num_iters` must be non-negative. Got {num_iters}.")
    KORNIA_CHECK(check_every >= 1, f"`check_every` must be positive. Got {check_every}.")

    if kernel is None:
        kernel = torch.ones(3, 3, device=seed.device, dtype=torch.bool)

    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["KH", "KW"])
    se_h, se_w = kernel.shape
    KORNIA_CHECK(se_h % 2 == 1 and se_w % 2 == 1, f"Kernel sizes must be odd. Got {se_h} x {se_w}.")

    # The kernel is only a membership mask, so a bool copy keeps its dtype out of the result. The center cell
    # keeps each pixel in its own neighborhood, as in scikit-image. It also makes every step non-decreasing,
    # which the convergence test relies on.
    kernel = kernel != 0
    kernel[se_h // 2, se_w // 2] = True

    if method == "erosion":
        return -_reconstruct_up(-seed, -mask, kernel, num_iters, check_every, engine)

    return _reconstruct_up(seed, mask, kernel, num_iters, check_every, engine)
