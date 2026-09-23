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
          flipped kernel. OpenCV does not reflect either, so ``anchor=(a_x, a_y)`` returns kornia's dilation
          by the flipped kernel at ``origin=[k_h - 1 - a_y, k_w - 1 - a_x]``; its default anchor gives
          ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]``, the default origin for an odd-sized kernel and one cell
          earlier for an even-sized one.
          :func:`erosion` does not reflect; scipy and OpenCV agree with it for odd and even sizes alike,
          while scikit-image centres an even-sized footprint one cell earlier, so
          ``skimage.morphology.erosion`` matches kornia's ``erosion`` at that same ``origin``.
        - ``origin`` is the ``[row, col]`` index of the structuring-element cell placed on the output pixel,
          defaulting to ``[k_h // 2, k_w // 2]`` for odd and for even sizes alike. scipy spells the same
          choice as an offset from ``k // 2``, and OpenCV's ``anchor`` is an index in ``(x, y)`` order.
        - ``border_type="geodesic"`` (the default) ignores the pixels outside the image, which is
          scikit-image's ``mode="ignore"`` and OpenCV's default border. scikit-image's own default is
          ``mode="reflect"``, so a comparison against it has to pass ``mode="ignore"`` explicitly. scipy has
          no such mode: ``mode="constant"`` with ``cval=-np.inf`` for ``grey_dilation`` and ``cval=np.inf``
          for ``grey_erosion`` reproduces it. Each of the three matches this border rule on every window
          that holds an in-image kernel cell, and only while :math:`|x|` stays well below ``max_val``
          (see the first warning).
        - The other border modes carry torch's names, which do not match scipy's and scikit-image's:
          ``reflect`` is their ``mirror``, ``replicate`` their ``nearest`` and ``circular`` their ``wrap``,
          while their ``reflect`` is a rule this function has no name for. ``geodesic`` is not ``replicate``:
          the two can differ once the structuring element can reach outside the image, including when its origin
          cell is a member and the gaps are elsewhere (``kernel=[[1, 0, 1, 0, 1]]``). They coincide for a
          rectangle of ones with a flat structuring element and :math:`|x|` well below ``max_val``, where
          every pixel the replicate pad duplicates is already in the window; a non-flat
          ``structuring_element`` adds a different value to the duplicate, and ``engine="convolution"`` rounds
          the two pads differently.
        - :func:`opening` and :func:`closing` reuse ``kernel`` in both halves, so under
          ``border_type="geodesic"`` or ``"circular"``, with a flat structuring element and any engine but
          ``"convolution"``, they are morphological openings and closings *up to the* ``max_val``
          *sentinel*: under ``geodesic`` a window that reaches outside the image can leave ``x - max_val``
          (``x + max_val`` in an erosion) in the output, and the later stage that adds (subtracts) ``max_val``
          back returns ``x`` quantised to that sentinel's spacing. While :math:`|x|` stays well below
          ``max_val``, anti-extensivity, extensivity and idempotence can then miss by a fraction of that
          spacing (``circular`` pads real pixels and, for a kernel with a non-zero cell, misses nothing); once
          :math:`|x|` approaches ``max_val`` the sentinel clips the data instead and they miss by the clip (see
          the first warning below). :func:`opening` lists what the other borders, a non-flat
          ``structuring_element`` and ``engine="convolution"`` break.
          :func:`top_hat` and :func:`bottom_hat` are their one-line definitions (``tensor - opening`` and
          ``closing - tensor``), and :func:`gradient` is the one-line ``dilation - erosion``. The kernel,
          origin and border conventions above apply to all seven.

    .. warning::
        ``max_val`` is a finite stand-in for infinity, not an infinity. It is carried into the masked-out
        neighborhood cells, so such a cell contributes ``x - max_val`` in :func:`dilation` and
        ``x + max_val`` in :func:`erosion`, and under the default ``border_type="geodesic"`` it is also
        padded into the border -- ``-max_val`` in :func:`dilation`, ``+max_val`` in :func:`erosion`. It
        therefore reaches the output whenever a geodesic window is empty or the range of ``x`` plus
        ``structuring_element`` approaches it, and the geodesic pad bounds the accuracy of
        ``engine="convolution"``. The same sentinel is used in all seven functions. Keep it well above that
        range and finite in the operands' dtypes: a finite ``max_val`` above 65504 makes the geodesic pad of
        a ``float16`` image raise on CPU and CUDA but round on MPS, and storing it into a ``float16`` kernel or
        structuring element -- which includes a ``bool`` or integer kernel on a ``float16`` image, since the
        image lends it its dtype -- raises on MPS and on CPU with torch 2.5.1 but rounds on CPU and CUDA with
        torch 2.14 (to ``-65504``, or to ``-inf`` from 65520); that store converts the scalar before it looks
        for a zero cell, so a kernel of all ones raises there too.
        Tracked in `#4734 <https://github.com/kornia/kornia/issues/4734>`_.

    .. warning::
        Only floating-point images are supported; on one, a ``bool`` or integer ``kernel`` is fine (see the
        first item below). The ``max_val`` sentinel is stored into the image's geodesic pad *and* into the
        kernel -- into ``structuring_element`` instead when one is given -- so what a non-float call does
        depends on the pair of dtypes. The ``unfold`` and ``shift`` engines return
        their promoted dtype (``torch.promote_types`` of ``tensor`` and ``kernel``, or of ``tensor`` and
        ``structuring_element``):
        under a border other than ``geodesic``, a ``uint8`` image with a ``float64`` kernel returns
        ``float64``, not ``float32``. ``engine="convolution"`` casts the kernel and the sentinel to the
        image's dtype and returns that dtype instead: on CPU a ``uint8`` image then gets an out-of-range cast
        of the sentinel, which wraps or saturates depending on the platform, the torch release and the kernel
        cell (dilating ``[10, 20, 30]`` by ``[[1, 1, 0]]`` under ``border_type="constant"`` gives
        ``[240, 250, 30]`` where ``-max_val`` wraps, while a ``-max_val`` saturated to ``0`` makes the masked
        cell a member), MPS and CUDA reject every integer image, and a ``bool`` image raises on all three
        (`#4762 <https://github.com/kornia/kornia/issues/4762>`_).

        - Without a ``structuring_element``, a floating-point image lends its dtype to a ``bool`` or integer
          ``kernel``, which is then only a membership mask and returns exactly what the same kernel in the
          image's dtype does. On a non-float image the masked-out cells store ``-max_val`` in the kernel's own
          dtype instead. A ``uint8`` kernel (for any positive ``max_val``) or an ``int8`` kernel (above
          ``128``, the default included) then raises an overflow ``RuntimeError`` under every ``border_type``
          whose pad accepts the image (the next two items say which do not). A ``bool`` kernel stores it as
          ``True``: every function but :func:`dilation` contains an erosion, which raises a torch error
          (``NotImplementedError`` on recent torch, ``RuntimeError`` on older releases) except under
          ``engine="convolution"`` with an integer image on CPU, where it runs under every such border and a
          ``False`` cell contributes ``x - 1`` in the image's dtype, wrapping at its bounds (``0 - 1`` is
          ``255`` for ``uint8``, ``-128 - 1`` is ``127`` for ``int8``); :func:`dilation` is silently wrong
          under every ``border_type`` as soon as the kernel holds a ``False`` cell, which then contributes
          ``x + 1`` (wrapping likewise) instead of being left out -- for a ``bool`` image, ``True`` everywhere
          under every border that accepts one. With a floating ``structuring_element`` the kernel is only the
          ``kernel == 0`` mask, and a ``uint8`` or ``bool`` kernel returns what the floating kernel does.
        - The geodesic pad stores :math:`\mp` ``max_val`` in the image's dtype. A ``uint8`` image raises an
          overflow ``RuntimeError`` there on CPU and CUDA whenever the kernel needs a pad, while on MPS the
          sentinel wraps modulo 256 instead of raising; an ``int8`` image does the same on CPU and MPS (its pad
          reads ``-16`` and ``16``). Under the other ``border_type`` values a ``uint8`` or ``int8`` image with a
          floating kernel runs and returns the dtype described above. An ``int64`` image is silently wrong once its
          range approaches ``max_val``, and a ``float32`` kernel promotes it to ``float32``, which cannot
          hold every integer above :math:`2^{24}` whatever ``max_val`` is.
        - On CPU and CUDA a ``bool`` image stores the geodesic pad as ``True``. With a floating kernel, or a
          ``bool`` kernel with no ``False`` cell, :func:`dilation` returns the correct dilation plus a ``True``
          (or ``1``) border ring, as wide on each side as the kernel's members reach past that edge: the whole
          pad for a rectangle of ones, the right side only for ``[[1, 1, 0, 0, 0]]``. The ring alone fills only an
          image no larger than itself (an all-``False`` :math:`1 \times 5` under ``ones(1, 3)`` keeps three
          ``False`` pixels); the :math:`1 \times 5` of the issue comes back all ``True`` because the ring and
          the true dilation of its centre pixel together cover it. The result is exact with a
          :math:`1 \times 1` kernel, with ``border_type="constant"`` and with ``circular``, which pads the
          image's own values. With a floating kernel and the ``shift`` engine, :func:`erosion` is exact under
          the geodesic pad, because ``True`` cannot lower a minimum, and :func:`gradient` inherits the ring
          from :func:`dilation`; under ``unfold`` (the ``"auto"`` choice on CUDA) the erosion raises. On CPU
          and CUDA the ``reflect`` and ``replicate`` pads raise on a ``bool`` image. MPS stores the raw bytes of
          :math:`\mp` ``max_val`` modulo 256 in the pad instead of ``True`` (240 and 16 for ``1e4``), which
          torch 2.14 reads as ``True`` unless they are ``0`` and torch 2.5.1 as signed integers, so there the
          ring and the exactness of :func:`erosion` depend on ``max_val``: ``1e4`` leaves no ring on torch 2.5.1
          (``-16`` with a floating kernel), and ``9984`` leaves no ring and erodes the border to ``0`` on both.

        Tracked in `#4735 <https://github.com/kornia/kornia/issues/4735>`_.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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

    _validate_morphology_inputs(kernel, structuring_element, border_type)

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
    if border_type == "constant":
        output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask: a bool or integer kernel cannot hold ``-max_val``, so a
        # floating-point image lends it its dtype. A float kernel keeps its own, which may widen the result.
        nb_dtype = tensor.dtype if tensor.is_floating_point() and not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
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
        ``skimage.morphology.erosion`` and ``cv2.erode``. scipy (with ``mode="constant", cval=np.inf``) and
        OpenCV agree with kornia pixel for pixel for odd-sized and even-sized kernels alike wherever the window
        holds at least one in-image cell of the kernel and :math:`|x|` stays well below ``max_val``;
        scikit-image centres an even-sized footprint one cell earlier, so it matches kornia's ``erosion`` at
        ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]`` instead.
        Under ``border_type="geodesic"`` a window with no in-image kernel cell is empty. scipy and
        scikit-image return ``inf`` there and OpenCV the dtype's largest value (``FLT_MAX`` for ``float32``),
        while kornia returns a finite value that depends on the image. For a flat kernel with at least one
        non-zero cell, an image and kernel of one dtype (a floating-point image lends its dtype to a ``bool``
        or integer kernel) and any engine but ``"convolution"``, it is ``s + min(0, m)`` rounded to that dtype,
        where ``s`` is the value that dtype stores for ``max_val`` (``9984`` in ``bfloat16``) and ``m`` the
        smallest in-image pixel under a masked-out cell of that window (``min(0, m)`` is ``0`` when there is
        none). A negative pixel there can pull it below ``max_val``: ``x=[[-2.]]`` with ``kernel=[[0, 1]]``
        and ``origin=[0, 0]`` returns ``9998`` in ``float32``, and ``x=[[-34.]]`` returns ``9920`` in
        ``bfloat16`` (``9984 - 34`` rounded), not ``9984`` (``1e4 - 34`` rounded). :func:`dilation` mirrors it
        with ``-s + max(0, M)``, ``M`` the largest such pixel.

        Under ``border_type="geodesic"`` or ``"circular"``, with a flat structuring element and any engine but
        ``"convolution"``, ``dilation`` and ``erosion`` with the same ``kernel`` and ``origin`` are an adjoint
        pair -- ``dilation(x) <= y`` everywhere exactly when ``x <= erosion(y)`` everywhere -- while no window
        is empty and :math:`|x|` stays well below ``max_val``. The finite sentinel breaks the pair
        otherwise: with ``max_val=1``, ``x = y = [[-2.]]``, ``kernel=[[0, 1]]`` and ``origin=[0, 0]``,
        ``dilation(x) <= y`` is false while ``x <= erosion(y)`` is true. The ``constant``, ``reflect`` and
        ``replicate`` pads can break it with no window empty: under ``border_type="constant"`` with
        ``ones(3, 3)`` and ``x = y = -1`` everywhere, the same two tests disagree. The duality of
        ``dilation`` and ``erosion`` under negation is ``erosion(tensor, kernel, origin=origin)`` equals
        ``-dilation(-tensor, kernel.flip((0, 1)), origin=[k_h - 1 - origin[0], k_w - 1 - origin[1]])``, which
        is exact up to the sign of a zero for a flat structuring element and, under ``border_type="constant"``,
        ``border_value=0``; a non-flat ``structuring_element`` has to be flipped with the kernel, and under
        ``constant`` a non-zero ``border_value`` has to be negated (the other borders ignore it).

        The two ``.. warning::`` blocks in :func:`dilation` describe this function too, including what a
        ``bool`` kernel does to an erosion of a non-float image.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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

    _validate_morphology_inputs(kernel, structuring_element, border_type)

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type = "constant"
    if border_type == "constant":
        output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)
    else:
        output = F.pad(tensor, pad_e, mode=border_type)

    # computation
    if structuring_element is None:
        # ``kernel`` is only a membership mask: a bool or integer kernel cannot hold ``-max_val``, so a
        # floating-point image lends it its dtype. A float kernel keeps its own, which may widen the result.
        nb_dtype = tensor.dtype if tensor.is_floating_point() and not kernel.is_floating_point() else kernel.dtype
        neighborhood = torch.zeros_like(kernel, dtype=nb_dtype)
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
        morphological opening -- anti-extensive and idempotent -- for an asymmetric kernel as well, within the
        border, engine and structuring-element limits below and up to the ``max_val`` sentinel: while
        :math:`|x|` stays well below ``max_val``, a window that reaches outside the image round-trips the
        sentinel and can move a pixel by a fraction of its spacing; once :math:`|x|` approaches ``max_val``
        the sentinel clips the data instead and the invariants miss by the clip (see the first warning in
        :func:`dilation`).

        With a flat structuring element, any ``engine`` but ``"convolution"`` (the default ``"auto"`` never
        picks it), a ``border_type`` of ``geodesic``, ``replicate`` or ``circular`` and :math:`|x|` well
        below ``max_val``, the invariants are exact for ``[[0, 1, 1]]`` and for
        ``[[0, 0, 0], [0, 1, 1], [0, 1, 0]]`` at the default origin and for ``ones(3, 3)`` at
        ``origin=[0, 0]``; the ``constant`` and ``reflect`` borders can break them even for those kernels,
        and so can a non-flat ``structuring_element``, which is subtracted and added back and rounds on the
        way (``structuring_element=[[0.7]]`` with ``ones(1, 1)`` opens ``0.1`` to ``0.10000002`` in
        ``float32``), and ``engine="convolution"`` wherever its ``conv2d`` rounds.
        ``[[1, 0, 0]]``, whose window leaves the image on one side only, depends on the border: under the
        default ``geodesic`` it can miss idempotence, and on negative data anti-extensivity too, by less than
        one ULP of ``max_val`` in the image's dtype; under ``replicate`` it stays idempotent but is not
        anti-extensive at all; under ``circular`` it is exact.

        ``skimage.morphology.opening`` with ``mode="ignore"`` mirrors its footprint in the second half, so it
        is this opening at ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]``, where its erosion half anchors (see
        :func:`erosion`): the default origin for an odd-sized kernel, one cell earlier for an even one. It is
        bit-equal to ``opening`` at that origin on random frames for odd and even kernels alike while no
        window is empty (an empty one returns an infinity there and a finite value here: the sentinel, or an
        ordinary-looking one such as ``0``).
        ``scipy.ndimage.grey_opening`` has no ignore mode, and a single ``cval=-inf`` pads its erosion half
        with ``-inf`` as well, so it is anti-extensive but differs from ``opening`` at the border and can
        return ``-inf`` there (the whole last column for ``[[1, 0, 0]]``). At their shared default
        ``mode="reflect"`` neither is anti-extensive for ``[[1, 0, 0]]``, while the symmetric ``[[1, 0, 1]]``,
        which omits its origin too, stays anti-extensive there. OpenCV's ``MORPH_OPEN`` composes without a
        flip, so it is an opening only for a kernel symmetric about its anchor: an asymmetric kernel, or an
        even-sized one at the default anchor such as ``ones(2, 2)``, makes it alter a block that ``opening``
        leaves untouched. Conventions otherwise as in :func:`dilation`.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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
        morphological closing -- extensive and idempotent -- for an asymmetric kernel as well, within the
        border, engine and structuring-element limits below and up to the ``max_val`` sentinel: while
        :math:`|x|` stays well below ``max_val``, a window that reaches outside the image round-trips the
        sentinel and can move a pixel by a fraction of its spacing; once :math:`|x|` approaches ``max_val``
        the sentinel clips the data instead and the invariants miss by the clip (see the first warning in
        :func:`dilation`).

        With a flat structuring element, any ``engine`` but ``"convolution"`` (the default ``"auto"`` never
        picks it), a ``border_type`` of ``geodesic``, ``replicate`` or ``circular`` and :math:`|x|` well
        below ``max_val``, the invariants are exact for ``[[0, 1, 1]]`` and for
        ``[[0, 0, 0], [0, 1, 1], [0, 1, 0]]`` at the default origin and for ``ones(3, 3)`` at
        ``origin=[0, 0]``; the ``constant`` and ``reflect`` borders can break them even for those kernels,
        and so can a non-flat ``structuring_element``, which is added and subtracted back and rounds on the
        way (``structuring_element=[[0.3]]`` with ``ones(1, 1)`` closes ``0.1`` to ``0.09999999`` in
        ``float32``), and ``engine="convolution"`` wherever its ``conv2d`` rounds.
        ``[[1, 0, 0]]``, whose window leaves the image on one side only, depends on the border: under the
        default ``geodesic`` it can miss extensivity, and on negative data idempotence too, by less than one
        ULP of ``max_val`` in the image's dtype; under ``replicate`` it stays idempotent but is not extensive
        at all; under ``circular`` it is exact.

        ``skimage.morphology.closing`` with ``mode="ignore"`` mirrors its footprint in the second half, which
        makes it kornia's closing by the *flipped* kernel at the default origin -- bit-equal to
        ``closing(x, kernel.flip((0, 1)))`` on random frames for odd-sized and even-sized kernels alike while
        no window is empty (an empty one returns an infinity there and a finite value here: the sentinel, or
        an ordinary-looking one) -- so it is a closing too, but a different one for an asymmetric kernel.
        ``scipy.ndimage.grey_closing`` has no ignore mode, and a single ``cval=+inf`` pads its dilation half
        with ``+inf`` as well, so it is extensive but differs from ``closing`` at the border and can return
        ``inf`` there (the whole first column for ``[[1, 0, 0]]``). At their shared default ``mode="reflect"``
        neither is extensive for ``[[1, 0, 0]]``, while the symmetric ``[[1, 0, 1]]``, which omits its origin
        too, stays extensive there. OpenCV's ``MORPH_CLOSE`` composes without a flip, so it is a closing only
        for a kernel symmetric about its anchor: an asymmetric kernel, or an even-sized one at the default
        anchor such as ``ones(2, 2)``, makes it alter a block that ``closing`` leaves untouched. Conventions
        otherwise as in :func:`dilation`.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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
        conventions of :func:`opening` apply to it unchanged, its scikit-image counterpart included:
        ``skimage.morphology.white_tophat`` with ``mode="ignore"`` is ``top_hat`` at
        ``origin=[(k_h - 1) // 2, (k_w - 1) // 2]`` while no window is empty.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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
        conventions of :func:`closing` apply to it unchanged, its scikit-image counterpart included:
        ``skimage.morphology.black_tophat`` with ``mode="ignore"`` is ``bottom_hat`` by the flipped kernel
        while no window is empty.

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
            image size (``reflect``) or at most match it (``circular``), and raise a ``RuntimeError``
            otherwise. Any other value raises a ``ValueError``.
        border_value: Value to fill past edges of input. It is used only when ``border_type`` is
            ``constant``: under ``geodesic`` it is silently overwritten with :math:`\mp` ``max_val``, and
            under ``reflect``, ``replicate`` and ``circular`` it is ignored.
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
