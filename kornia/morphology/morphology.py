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

__all__ = ["bottom_hat", "closing", "dilation", "erosion", "gradient", "opening", "top_hat"]


def _validate_morphology_inputs(
    kernel: torch.Tensor, structuring_element: Optional[torch.Tensor], border_type: str
) -> None:
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


def _shift_reduce(
    padded: torch.Tensor,
    offsets: torch.Tensor,
    kernel: torch.Tensor,
    height: int,
    width: int,
    dilate: bool,
    inplace: bool,
    reduction_value: Optional[float],
) -> torch.Tensor:
    """Running max (``dilate``) or min over the ``k_h * k_w`` shifted views of ``padded`` plus their offsets.

    The ``shift`` engine: output pixel ``(y, x)`` reduces ``padded[y + i, x + j] + offsets[i, j]`` over the
    kernel positions, the same max-plus expression ``unfold`` evaluates, without materialising the window
    tensor. ``torch.compile`` fuses the loop. A cell where ``kernel`` is zero takes ``reduction_value``, the
    reduction identity; ``None``, for a non-float image, leaves the ``max_val`` already folded into ``offsets``.

    Forward-only this holds one output-sized intermediate whatever the kernel area, so peak memory is flat
    in ``k_h k_w`` where ``unfold`` and ``convolution`` grow with it (24 MiB against 1627 MiB at 15 x 15,
    B=8 x 3 x 256^2 float32 on CUDA). Under autograd the opposite is true: every ``torch.maximum`` saves
    both operands, so ``2 (k_h k_w - 1)`` output-sized tensors stay alive until backward (2717 MiB
    against 1627 MiB in that same cell). :func:`_resolve_engine` accounts for both regimes.
    """
    kh, kw = offsets.shape
    # Keep each offset two-dimensional so PyTorch applies tensor-tensor dtype promotion. Indexing
    # down to a scalar would instead apply wrapped-scalar rules and silently keep ``padded.dtype``.
    output = padded[..., 0:height, 0:width] + offsets[0:1, 0:1]
    if reduction_value is not None:
        output = torch.where(
            kernel[0, 0] != 0,
            output,
            torch.full_like(output, reduction_value),
        )
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


def _resolve_engine(
    engine: str, tensor: torch.Tensor, recording_grad: bool = False, dtype: Optional[torch.dtype] = None
) -> str:
    """Map ``engine="auto"`` to the preferred engine for ``tensor``; leave other values unchanged.

    ``recording_grad`` says whether this call will build a backward graph, which changes the ranking.
    ``dtype`` is the dtype the max-plus terms are computed in; it defaults to ``tensor.dtype`` and is
    wider when the kernel or the structuring element promotes the computation, which is what the
    dtype rule below has to see. For finite inputs, the two engines selected by ``auto`` (``unfold``
    and ``shift``) return equal forward output, so switching between them never changes a value; only
    the sign of a zero can differ, because a backend's ``max``/``min`` may return either tied operand.

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
        - Known defects: a non-float image is not rejected
          (`#4735 <https://github.com/kornia/kornia/issues/4735>`_); and
          ``engine="convolution"`` returns the image dtype where ``unfold`` and ``shift`` return the dtype
          promoted with ``structuring_element``, or with ``kernel`` when none is given
          (`#4762 <https://github.com/kornia/kornia/issues/4762>`_).

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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

    _validate_morphology_inputs(kernel, structuring_element, border_type)

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask, so a bool or integer kernel does not set the compute dtype: a
        # floating-point image lends it its own. A float kernel keeps its dtype, which may widen the result.
        nb_dtype = tensor.dtype if tensor.is_floating_point() and not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
    else:
        neighborhood = structuring_element.clone()
    float_image = tensor.is_floating_point()
    if not float_image:
        # A non-float image is not supported (#4735); keep the finite ``max_val`` arithmetic it has always had.
        neighborhood[kernel == 0] = -max_val

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
        geodesic_value = -float("inf") if float_image else -max_val
        output: torch.Tensor = F.pad(tensor, pad_e, mode="constant", value=geodesic_value)
    elif border_type == "constant":
        output = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    reduction_min: float = -float("inf")
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output = output + neighborhood.flip((0, 1))
        if float_image:
            output.masked_fill_(
                kernel.flip((0, 1)).view(1, 1, 1, 1, se_h, se_w) == 0,
                reduction_min,
            )
        output, _ = torch.max(output, 4)
        output, _ = torch.max(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=output.dtype)
        conv_neighborhood = neighborhood.masked_fill(kernel == 0, 0.0) if float_image else neighborhood

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
            bias=conv_neighborhood.view(-1).flip(0).to(dtype=output.dtype),
        )

        if output.is_floating_point():
            output = output.masked_fill(special == 1, float("inf"))
            output = output.masked_fill(special == 2, -float("inf"))
            output = output.masked_fill(special == 3, float("nan"))

        if float_image:
            output = output.masked_fill(
                kernel.view(-1).flip(0).view(1, -1, 1, 1) == 0,
                reduction_min,
            )

        if is_geodesic and float_image:
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
            True,
            inplace,
            reduction_min if float_image else None,
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
        OpenCV. ``structuring_element`` is subtracted before the minimum. The known defects listed in
        :func:`dilation` apply here too.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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

    _validate_morphology_inputs(kernel, structuring_element, border_type)

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask, so a bool or integer kernel does not set the compute dtype: a
        # floating-point image lends it its own. A float kernel keeps its dtype, which may widen the result.
        nb_dtype = tensor.dtype if tensor.is_floating_point() and not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
    else:
        neighborhood = structuring_element.clone()
    float_image = tensor.is_floating_point()
    if not float_image:
        # A non-float image is not supported (#4735); keep the finite ``max_val`` arithmetic it has always had.
        neighborhood[kernel == 0] = -max_val

    # The max-plus terms compute in the promoted dtype, which the dtype rule of ``auto`` has to see: a
    # float16 image with a float32 kernel computes, and differentiates, in float32.
    compute_dtype = torch.promote_types(tensor.dtype, neighborhood.dtype)
    recording_grad = _records_grad(tensor, structuring_element)
    engine = _resolve_engine(engine, tensor, recording_grad, compute_dtype)

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    is_geodesic = border_type == "geodesic"
    if border_type == "geodesic":
        geodesic_value = float("inf") if float_image else max_val
        output: torch.Tensor = F.pad(tensor, pad_e, mode="constant", value=geodesic_value)
    elif border_type == "constant":
        output = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    reduction_max: float = float("inf")
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output = output - neighborhood
        if float_image:
            output.masked_fill_(
                kernel.view(1, 1, 1, 1, se_h, se_w) == 0,
                reduction_max,
            )
        output, _ = torch.min(output, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=output.dtype)
        conv_neighborhood = neighborhood.masked_fill(kernel == 0, 0.0) if float_image else neighborhood

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
            bias=-conv_neighborhood.view(-1).to(dtype=output.dtype),
        )

        if output.is_floating_point():
            output = output.masked_fill(special == 1, float("inf"))
            output = output.masked_fill(special == 2, -float("inf"))
            output = output.masked_fill(special == 3, float("nan"))

        if float_image:
            output = output.masked_fill(
                kernel.view(-1).view(1, -1, 1, 1) == 0,
                reduction_max,
            )

        if is_geodesic and float_image:
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
            False,
            inplace,
            reduction_max if float_image else None,
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
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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
        tensor: Image with shape :math:`(B, C, H, W)`.
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
        max_val: No effect on a floating-point image, whose excluded kernel cells and ``geodesic`` padding take
            the reduction identity :math:`\mp\infty`; kept for backward compatibility. A non-float image,
            which is not supported (`#4735 <https://github.com/kornia/kornia/issues/4735>`_), still uses it.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
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
