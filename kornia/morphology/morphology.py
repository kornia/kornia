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


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


def _shift_reduce(padded: torch.Tensor, offsets: torch.Tensor, height: int, width: int, dilate: bool) -> torch.Tensor:
    """Running max (``dilate``) or min over the ``k_h * k_w`` shifted views of ``padded`` plus their offsets.

    The ``shift`` engine: output pixel ``(y, x)`` reduces ``padded[y + i, x + j] + offsets[i, j]`` over the
    kernel positions, the same max-plus expression ``unfold`` evaluates, without materialising the window
    tensor. ``torch.compile`` fuses the loop.

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
    # ``unfold`` reduces the kernel-height dimension first and then kernel width; keep that traversal so
    # tied values are met in the same order. Which of two tied operands a backend's ``max``/``min``
    # returns is its own choice (CPU keeps the first, MPS the second), so the sign of a zero result is
    # not preserved between engines or devices; the value is.
    for j in range(kw):
        for i in range(kh):
            if i == 0 and j == 0:
                continue
            shifted = padded[..., i : i + height, j : j + width] + offsets[i : i + 1, j : j + 1]
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
        See :doc:`Conventions & Pitfalls </get-started/conventions>` for the table that compares the
        reflection, centring and border rules of ``kornia.morphology`` with scipy, scikit-image and OpenCV.

        - ``kernel`` is a flat **membership mask** of shape :math:`(k_h, k_w)`, laid over the image's
          :math:`(H, W)` axes. Every non-zero entry is a member of the neighborhood and its magnitude is
          ignored, so negative and fractional entries are members too; use ``structuring_element`` for
          weights. Entries of ``structuring_element`` under a zero ``kernel`` cell do not reach the output.
        - ``dilation`` **reflects** the structuring element. It is the Minkowski dilation
          :math:`\delta_B f(x) = \max_{b \in B} f(x - b)`, the convention of ``scipy.ndimage.grey_dilation``.
          scikit-image does not reflect, so for an asymmetric kernel it returns kornia's dilation by the
          flipped kernel. OpenCV does not reflect either, but it also anchors an even-sized kernel one cell
          earlier, so it returns kornia's dilation by the flipped kernel at
          ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]``, which for an odd-sized kernel is the default origin.
          :func:`erosion` does not reflect; scipy and OpenCV agree with it for odd and even sizes alike,
          while scikit-image centres an even-sized footprint one cell earlier, so
          ``skimage.morphology.erosion`` matches kornia's ``erosion`` at that same ``origin``.
        - ``origin`` is the ``[row, col]`` index of the structuring-element cell placed on the output pixel,
          defaulting to ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike. scipy spells the same
          choice as an offset from ``k // 2``, and OpenCV's ``anchor`` is an index in ``(x, y)`` order.
        - ``border_type="geodesic"`` (the default) ignores the pixels outside the image, which is
          scikit-image's ``mode="ignore"`` and OpenCV's default border. scikit-image's own default is
          ``mode="reflect"``, so a comparison against it has to pass ``mode="ignore"`` explicitly.
        - The other border modes carry torch's names, which do not match scipy's and scikit-image's:
          ``reflect`` is their ``mirror``, ``replicate`` their ``nearest`` and ``circular`` their ``wrap``,
          while their ``reflect`` is a rule this function has no name for. ``geodesic`` is not ``replicate``:
          the two can differ once the structuring element can reach outside the image, including when its origin
          cell is a member and the gaps are elsewhere (``kernel=[[1, 0, 1, 0, 1]]``). They coincide for a
          rectangle of ones, where every pixel the replicate pad duplicates is already in the window.
        - :func:`opening` and :func:`closing` reuse ``kernel`` in both halves, so they are morphological
          openings and closings *up to the* ``max_val`` *sentinel*: a window that reaches outside the image
          can leave ``x - max_val`` in the output, and adding ``max_val`` back in a later stage returns ``x``
          quantised to that sentinel's spacing. While :math:`|x|` stays well below ``max_val``,
          anti-extensivity, extensivity and idempotence can miss by a fraction of that spacing; once
          :math:`|x|` approaches ``max_val`` the sentinel clips the data instead and they miss by the clip
          (see the first warning below). :func:`top_hat` and :func:`bottom_hat` are their one-line
          definitions (``tensor - opening`` and ``closing - tensor``), and :func:`gradient` is the one-line
          ``dilation - erosion``. The kernel, origin and border conventions above apply to all seven.

    .. warning::
        ``max_val`` is a finite stand-in for infinity, not an infinity. It is padded into the border --
        ``-max_val`` in :func:`dilation`, ``+max_val`` in :func:`erosion` -- and carried into the masked-out
        neighborhood cells with the same signs, so such a cell contributes ``x - max_val`` in
        :func:`dilation` and ``x + max_val`` in :func:`erosion`. It therefore reaches the output whenever a
        window is empty or the image range approaches it, and it bounds the accuracy of
        ``engine="convolution"``. The same sentinel is used in all seven functions. Keep it well above
        :math:`|x|` and representable in the input dtype: in ``float16`` it must stay at or below 65504,
        or the store raises. Tracked in `#4734 <https://github.com/kornia/kornia/issues/4734>`_.

    .. warning::
        Only floating-point input is supported. ``uint8`` input does not survive the geodesic pad, which
        stores :math:`\mp` ``max_val``: on CPU it raises and on MPS the sentinel wraps modulo 256 instead of
        raising; under the other ``border_type`` values it runs but silently returns ``float32``. ``int64``
        is silently wrong once the image range approaches ``max_val``. ``bool`` input is not rejected either:
        :func:`dilation` returns the correct dilation plus a ``True`` border ring as wide as the pad the
        kernel needs, left by the geodesic pad, which is ``True`` in ``bool`` -- so only an image no larger
        than that ring comes back all ``True``, and the result is exact with a :math:`1 \times 1` kernel or
        with ``border_type="constant"``. :func:`erosion` and :func:`gradient` raise a torch error
        (``NotImplementedError`` on recent torch, ``RuntimeError`` on older releases) on ``bool``, as do the
        ``reflect`` and ``replicate`` pads. Tracked in
        `#4735 <https://github.com/kornia/kornia/issues/4735>`_.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning above.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    # The kernel is reflected below (Minkowski dilation), so the window is anchored at the reflected origin.
    pad_e: List[int] = [se_w - origin[1] - 1, origin[1], se_h - origin[0] - 1, origin[0]]
    if border_type == "geodesic":
        border_value = -max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    # The max-plus terms compute in the promoted dtype, which the dtype rule of ``auto`` has to see: a
    # float16 image with a float32 kernel computes, and differentiates, in float32.
    compute_dtype = torch.promote_types(tensor.dtype, neighborhood.dtype)
    engine = _resolve_engine(engine, tensor, _records_grad(tensor, structuring_element), compute_dtype)
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
        output, _ = torch.max(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=output.dtype)
        output, _ = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad),
            reshape_kernel,
            padding=0,
            bias=neighborhood.view(-1).flip(0).to(dtype=output.dtype),
        ).max(dim=1)
        output = output.view(B, C, H, W)
    elif engine == "shift":
        output = _shift_reduce(output, neighborhood.flip((0, 1)), tensor.shape[-2], tensor.shape[-1], True)
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
        Conventions as in :func:`dilation`, with one difference: ``erosion`` does **not** reflect the
        structuring element. It is the Minkowski erosion
        :math:`\varepsilon_B f(x) = \min_{b \in B} f(x + b)`, the convention of ``scipy.ndimage.grey_erosion``,
        ``skimage.morphology.erosion`` and ``cv2.erode``. scipy and OpenCV agree with kornia pixel for pixel
        for odd-sized and even-sized kernels alike while ``|x|`` stays well below ``max_val``; scikit-image
        centres an even-sized footprint one cell earlier, so it matches kornia's ``erosion`` at
        ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]`` instead.
        An empty window returns ``max_val`` here, ``inf`` in scipy and scikit-image and ``FLT_MAX`` in
        OpenCV.

        ``dilation`` and ``erosion`` with the same ``kernel`` and ``origin`` are an adjoint pair. Their
        duality under negation is ``erosion(tensor, kernel, origin=origin)`` equals
        ``-dilation(-tensor, kernel.flip((0, 1)), origin=[k_h - 1 - origin[0], k_w - 1 - origin[1]])``, which
        is exact for a flat structuring element and ``border_value=0``; a non-flat ``structuring_element``
        has to be flipped with the kernel, and a non-zero ``border_value`` has to be negated.

        The two ``.. warning::`` blocks in :func:`dilation` describe this function too, except that
        ``erosion`` raises a torch error on ``bool`` input rather than returning a result (the exception
        class depends on the torch version).

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type = "constant"
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    # The max-plus terms compute in the promoted dtype, which the dtype rule of ``auto`` has to see: a
    # float16 image with a float32 kernel computes, and differentiates, in float32.
    compute_dtype = torch.promote_types(tensor.dtype, neighborhood.dtype)
    engine = _resolve_engine(engine, tensor, _records_grad(tensor, structuring_element), compute_dtype)
    if engine == "unfold":
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.min(output - neighborhood, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel).to(dtype=output.dtype)
        output, _ = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad),
            reshape_kernel,
            padding=0,
            bias=-neighborhood.view(-1).to(dtype=output.dtype),
        ).min(dim=1)
        output = output.view(B, C, H, W)
    elif engine == "shift":
        output = _shift_reduce(output, -neighborhood, tensor.shape[-2], tensor.shape[-1], False)
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
        ``opening`` is ``dilation(erosion(tensor))`` with the same ``kernel`` in both halves. Because
        :func:`dilation` reflects the structuring element and :func:`erosion` does not, the composition is a
        morphological opening -- anti-extensive and idempotent -- for an asymmetric kernel as well, up to the
        ``max_val`` sentinel: while :math:`|x|` stays well below ``max_val``, a window that reaches outside
        the image round-trips the sentinel and can move a pixel by a fraction of its spacing; once
        :math:`|x|` approaches ``max_val`` the sentinel clips the data instead and the invariants miss by the
        clip (see the first warning in :func:`dilation`).

        With ``engine="unfold"``, a ``border_type`` of ``geodesic``, ``replicate`` or ``circular`` and an
        image range well below ``max_val``, the invariants are exact for ``[[0, 1, 1]]`` and for
        ``[[0, 0, 0], [0, 1, 1], [0, 1, 0]]`` at the default origin and for ``ones(3, 3)`` at
        ``origin=[0, 0]``; ``engine="convolution"`` and the ``constant`` and ``reflect`` borders can break
        them even for those kernels. ``[[1, 0, 0]]``, whose window leaves the image on one side only, stays
        anti-extensive but loses idempotence, by less than one ULP of ``max_val`` and by nothing at all in
        ``float64``.

        ``scipy.ndimage.grey_opening`` and ``skimage.morphology.opening`` are openings too once their border
        is an infinity (``mode="ignore"`` in scikit-image, ``cval=-inf`` in scipy); at their shared default
        ``mode="reflect"`` neither is anti-extensive for a kernel that omits its own origin, such as
        ``[[1, 0, 0]]``. OpenCV's ``MORPH_OPEN`` composes without a flip and is not one for an asymmetric
        kernel -- it alters a block that ``opening`` leaves untouched. Conventions otherwise as in
        :func:`dilation`.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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
        ``closing`` is ``erosion(dilation(tensor))`` with the same ``kernel`` in both halves, so it is a
        morphological closing -- extensive and idempotent -- for an asymmetric kernel as well, up to the
        ``max_val`` sentinel: while :math:`|x|` stays well below ``max_val``, a window that reaches outside
        the image round-trips the sentinel and can move a pixel by a fraction of its spacing; once
        :math:`|x|` approaches ``max_val`` the sentinel clips the data instead and the invariants miss by the
        clip (see the first warning in :func:`dilation`).

        With ``engine="unfold"``, a ``border_type`` of ``geodesic``, ``replicate`` or ``circular`` and an
        image range well below ``max_val``, the invariants are exact for ``[[0, 1, 1]]`` and for
        ``[[0, 0, 0], [0, 1, 1], [0, 1, 0]]`` at the default origin and for ``ones(3, 3)`` at
        ``origin=[0, 0]``; ``engine="convolution"`` and the ``constant`` and ``reflect`` borders can break
        them even for those kernels. ``[[1, 0, 0]]``, whose window leaves the image on one side only, stays
        idempotent but loses extensivity, by less than one ULP of ``max_val`` and by nothing at all in
        ``float64``.

        ``scipy.ndimage.grey_closing`` and ``skimage.morphology.closing`` are closings too once their border
        is an infinity (``mode="ignore"`` in scikit-image, ``cval=+inf`` in scipy); at their shared default
        ``mode="reflect"`` neither is extensive for a kernel that omits its own origin, such as
        ``[[1, 0, 0]]``. OpenCV's ``MORPH_CLOSE`` composes without a flip and is not one for an asymmetric
        kernel -- it alters a block that ``closing`` leaves untouched. Conventions otherwise as in
        :func:`dilation`.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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
        ``gradient`` is ``dilation(tensor) - erosion(tensor)`` with the same ``kernel`` and the same
        options, so the conventions of :func:`dilation` and :func:`erosion` apply to it unchanged.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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
        ``top_hat`` is ``tensor - opening(tensor)`` with the same ``kernel`` and the same options, so the
        conventions of :func:`dilation` apply to it unchanged.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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
        ``bottom_hat`` is ``closing(tensor) - tensor`` with the same ``kernel`` and the same options, so the
        conventions of :func:`dilation` apply to it unchanged.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied, and their magnitude is
            ignored. Its shape is :math:`(k_h, k_w)`, laid over the image's :math:`(H, W)` axes.
            For a full neighborhood pass a ``kernel`` of all ones.
        structuring_element: Non-flat structuring element, added to the neighbor values before the maximum
            in :func:`dilation` and subtracted before the minimum in :func:`erosion`. Its shape must equal
            ``kernel``'s; a mismatch is reported by an internal ``IndexError``
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        origin: ``[row, col]`` index of the structuring-element cell placed on the output pixel.
            Default: ``None``, which uses ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike
            (``[1, 1]`` for a :math:`2 \times 2` kernel).
        border_type: How the image borders are handled. Default: ``geodesic``, which ignores the values
            that are outside the image when applying the operation. The other accepted values are the
            :func:`torch.nn.functional.pad` modes ``constant``, ``reflect``, ``replicate`` and ``circular``.
            ``reflect`` and ``circular`` additionally require the pad the kernel needs to stay below the
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. An unrecognised value is reported by ``torch`` without naming this argument
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` any value other than ``0.0`` raises a
            ``RuntimeError``, because it is always forwarded to :func:`torch.nn.functional.pad`
            (`#4736 <https://github.com/kornia/kornia/issues/4736>`_).
        max_val: Finite stand-in for the infinite elements of the kernel. See the first warning in
            :func:`dilation`.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). The ``"unfold"``
            and ``"shift"`` engines compute the same max-plus expression and, for finite inputs, return equal
            output; only the sign of a zero can differ, because a backend's ``max``/``min`` may return either
            tied operand.
            ``"auto"`` picks ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a CPU call
            which computes in float32 or float64 (the image dtype, or the wider dtype the kernel promotes it
            to) and records a backward graph takes ``"unfold"``, where ``"shift"`` is up to 3.4x slower. The
            measurements behind that rule are recorded in the ``_resolve_engine`` source.
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
