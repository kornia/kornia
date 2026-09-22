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
    its left operand, so ``k_h k_w - 1`` output-sized tensors stay alive until backward (2717 MiB against
    1627 MiB in that same cell). :func:`_resolve_engine` accounts for both regimes.
    """
    kh, kw = offsets.shape
    output = padded[..., 0:height, 0:width] + offsets[0, 0]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            shifted = padded[..., i : i + height, j : j + width] + offsets[i, j]
            output = torch.maximum(output, shifted) if dilate else torch.minimum(output, shifted)
    return output


def _resolve_engine(engine: str, tensor: torch.Tensor, recording_grad: bool = False) -> str:
    """Map ``engine="auto"`` to the preferred engine for ``tensor``; leave other values unchanged.

    ``recording_grad`` says whether this call will build a backward graph, which changes the ranking.
    All three engines return bitwise equal forward output, so switching on it never changes a value.

    Benchmarks in :mod:`benchmarks.morphology.engines` (x86 CPU and an RTX 4090, ``dilation``,
    B x 3 x 256 x 256) give three regimes:

    - CUDA: ``unfold`` is the broadly faster engine forward (23 of 24 cells) and forward + backward
      (21 of 24), so it is always the CUDA choice.
    - CPU without a backward graph: ``shift`` wins every cell, by 4-13x in float32 and 12-20x in half
      precision, and its peak memory is flat in the kernel area instead of growing with it.
    - CPU with a backward graph: the running max saves ``k_h k_w - 1`` intermediates, so ``shift``
      loses to ``unfold`` in float32 (up to 2.5x at 15 x 15) and float64 (up to 3.4x), while still
      winning 11 of 12 half-precision cells.

    MPS keeps ``shift`` in both cases: its ``unfold`` is an order of magnitude slower forward there,
    and the forward + backward sweep has not been run on a Metal device.
    """
    if engine == "auto":
        if tensor.device.type == "cuda":
            return "unfold"
        is_float32_or_64 = tensor.dtype in (torch.float32, torch.float64)
        if tensor.device.type == "cpu" and recording_grad and is_float32_or_64:
            return "unfold"
        return "shift"
    return engine


def _records_grad(tensor: torch.Tensor, kernel: torch.Tensor, structuring_element: Optional[torch.Tensor]) -> bool:
    """Whether an op over these inputs will build a backward graph.

    ``torch.no_grad()`` leaves ``requires_grad`` set on the inputs but records nothing, so grad mode has
    to be checked too: inference on a tensor that happens to require grad should take the forward-only
    engine. The kernel and the structuring element count because ``dilation`` and ``erosion``
    differentiate through the neighborhood they build from them.
    """
    if not torch.is_grad_enabled():
        return False
    if tensor.requires_grad or kernel.requires_grad:
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a non-flat
            structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
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
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
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

    engine = _resolve_engine(engine, tensor, _records_grad(tensor, kernel, structuring_element))
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
        output = _shift_reduce(
            output, neighborhood.flip((0, 1)).to(dtype=output.dtype), tensor.shape[-2], tensor.shape[-1], True
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element (torch.Tensor, optional): Structuring element used for the grayscale dilation.
            It may be a non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if border_type is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
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

    engine = _resolve_engine(engine, tensor, _records_grad(tensor, kernel, structuring_element))
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
        output = _shift_reduce(
            output, (-neighborhood).to(dtype=output.dtype), tensor.shape[-2], tensor.shape[-1], False
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       torch.Tensor: Opened image with shape :math:`(B, C, H, W)`.

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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default is None and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default is None and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
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

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``"unfold"``, ``"convolution"``, ``"shift"`` or ``"auto"`` (default). All three engines
            compute the same max-plus expression and return bitwise equal output. ``"auto"`` picks
            ``"unfold"`` on CUDA, and off CUDA the exact ``"shift"`` engine, except that a float32 or
            float64 CPU call which records a backward graph takes ``"unfold"``, where ``"shift"`` is up
            to 3.4x slower. See :func:`_resolve_engine` for the measurements.
            ``"convolution"`` runs through the backend's ``conv2d`` and inherits its precision: a float32
            convolution that computes in reduced precision (macOS CPU, CUDA with TF32 enabled) rounds the
            output. ``"shift"`` takes a running max or min over the :math:`k_h k_w` shifted views of the
            padded image. It is exact, and forward-only it needs no :math:`k_h k_w`-sized intermediate;
            under autograd it instead saves :math:`k_h k_w - 1` output-sized tensors for the backward
            pass, which is more memory than ``"unfold"``, not less.

    Returns:
       Top hat transformed image with shape :math:`(B, C, H, W)`.

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
