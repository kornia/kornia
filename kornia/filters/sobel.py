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

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

from .filter import _to_floating
from .kernels import (
    _normalize_kernel2d_2nd_order,
    get_spatial_gradient_kernel2d,
    get_spatial_gradient_kernel3d,
    normalize_kernel2d,
)


def spatial_gradient(input: torch.Tensor, mode: str = "sobel", order: int = 1, normalized: bool = True) -> torch.Tensor:
    r"""Compute the first or second image derivatives in x and y with Sobel or difference kernels.

    ``out = spatial_gradient(input)`` returns the x and y derivatives of each channel of a ``(B, C, H, W)`` image,
    stacked on a new axis 2.

    .. image:: _static/img/spatial_gradient.png

    Convention:
        - The derivatives are the kernels of :func:`~kornia.filters.get_spatial_gradient_kernel2d`, in that
          function's order and with its sign; see its Convention block.
        - For ``order=1`` with the default ``normalized=True``, an axis-aligned linear ramp with slope ``s`` has
          response ``s`` in the corresponding channel away from the replicated border. With
          ``normalized=False``, the Sobel response is ``8 s``. :ref:`Filtering <filtering-conventions>` maps both
          scales onto OpenCV, scipy and scikit-image.
        - The border is always replicated, so for ``order=1`` with ``normalized=True``, a linear ramp's derivative
          is half its slope on the first and last pixel along that axis. There is no ``border_type``, unlike
          :func:`~kornia.filters.filter2d` and :func:`~kornia.filters.laplacian`, whose default is ``'reflect'``.

    Args:
        input: image with shape ``(B, C, H, W)``.
        mode: kernel set, ``'sobel'`` or ``'diff'``, case-insensitive.
        order: ``1`` for the first derivatives, ``2`` for the second.
        normalized: if ``True``, scale the kernels so that every output channel estimates the derivative
          itself away from the replicated border: exact on linear surfaces for ``order=1`` and on quadratic
          surfaces for ``order=2``. If ``False``, return the raw kernel responses.

    Return:
        the derivatives of the input feature map, with shape ``(B, C, 2, H, W)`` holding
        :math:`(dx, dy)` for ``order=1`` and ``(B, C, 3, H, W)`` holding :math:`(dxx, dxy, dyy)`
        for ``order=2``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_edges.html>`__.

    See Also:
        - :class:`SpatialGradient`: the module form.
        - :func:`sobel`: the magnitude of the first derivatives, with no added axis.
        - :func:`spatial_gradient3d`: the same for volumes, with ``'diff'`` kernels only.
        - :func:`laplacian`: one second-derivative response per channel, on its own scale.
        - :func:`get_spatial_gradient_kernel2d`: the kernels it applies.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    # in its own dtype, uint8 wraps the differences and an integer kernel truncates the half and quarter taps (#5155)
    input = _to_floating(input)

    # The first-order Sobel kernels are fixed. Construct them directly instead
    # of allocating one kernel, transposing it, stacking both directions, and
    # normalizing at runtime. Keep the generic construction for complex inputs,
    # whose division semantics are part of the public API.
    if mode.lower() == "sobel" and order == 1 and input.is_floating_point():
        if normalized:
            kernel = torch.tensor(
                [
                    [[-0.125, 0.0, 0.125], [-0.25, 0.0, 0.25], [-0.125, 0.0, 0.125]],
                    [[-0.125, -0.25, -0.125], [0.0, 0.0, 0.0], [0.125, 0.25, 0.125]],
                ],
                device=input.device,
                dtype=input.dtype,
            )
        else:
            kernel = torch.tensor(
                [
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                ],
                device=input.device,
                dtype=input.dtype,
            )
    else:
        kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
        if normalized:
            kernel = normalize_kernel2d(kernel) if order == 1 else _normalize_kernel2d_2nd_order(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with "replicate for spatial dims, but with torch.zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, "replicate")
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)


def spatial_gradient3d(input: torch.Tensor, mode: str = "diff", order: int = 1) -> torch.Tensor:
    r"""Compute the first or second derivatives of a volume along x, y and depth with difference kernels.

    ``out = spatial_gradient3d(input)`` returns the derivatives of each channel of a ``(B, C, D, H, W)`` volume,
    stacked on a new axis 2.

    Convention:
        - The derivatives are the kernels of :func:`~kornia.filters.get_spatial_gradient_kernel3d`, in that
          function's order; see its Convention block for their units. There is no ``normalized`` argument: a
          floating input gives derivative units.
        - The border is always replicated, as in :func:`~kornia.filters.spatial_gradient`.
        - An integer or bool input is differentiated in float32 and the output is float32, on the input's scale,
          as :func:`~kornia.filters.filter3d` does (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).

    Args:
        input: volume with shape ``(B, C, D, H, W)``.
        mode: kernel set, case-insensitive; only ``'diff'`` is implemented.
        order: ``1`` for the first derivatives, ``2`` for the second.

    Return:
        the derivatives, with shape ``(B, C, 3, D, H, W)`` holding :math:`(dx, dy, dz)` for ``order=1`` or
        ``(B, C, 6, D, H, W)`` holding :math:`(dxx, dyy, dzz, dxy, dyz, dxz)` for ``order=2``.

    See Also:
        - :class:`SpatialGradient3d`: the module form.
        - :func:`spatial_gradient`: the same for images, with ``'sobel'`` kernels and a ``normalized`` argument.
        - :func:`get_spatial_gradient_kernel3d`: the kernels it applies.

    Examples:
        >>> input = torch.rand(1, 4, 2, 4, 4)
        >>> output = spatial_gradient3d(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "D", "H", "W"])
    # in its own dtype, uint8 wraps the differences and an integer kernel truncates the half and quarter taps (#5155)
    input = _to_floating(input)

    b, c, d, h, w = input.shape
    dev = input.device
    dtype = input.dtype
    if mode.lower() == "diff" and order == 1:
        # we go for the special case implementation due to conv3d bad speed
        x: torch.Tensor = F.pad(input, 6 * [1], "replicate")
        center = slice(1, -1)
        left = slice(0, -2)
        right = slice(2, None)
        out = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        out[..., 0, :, :, :] = x[..., center, center, right] - x[..., center, center, left]
        out[..., 1, :, :, :] = x[..., center, right, center] - x[..., center, left, center]
        out[..., 2, :, :, :] = x[..., right, center, center] - x[..., left, center, center]
        out = 0.5 * out
    else:
        # prepare kernel
        # allocate kernel
        kernel = get_spatial_gradient_kernel3d(mode, order, device=dev, dtype=dtype)

        tmp_kernel = kernel.repeat(c, 1, 1, 1, 1)

        # Pad with "replicate for spatial dims, but with torch.zeros for channel
        spatial_pad = [
            kernel.size(2) // 2,
            kernel.size(2) // 2,
            kernel.size(3) // 2,
            kernel.size(3) // 2,
            kernel.size(4) // 2,
            kernel.size(4) // 2,
        ]
        out_ch: int = 6 if order == 2 else 3
        out = F.conv3d(F.pad(input, spatial_pad, "replicate"), tmp_kernel, padding=0, groups=c).view(
            b, c, out_ch, d, h, w
        )
    return out


def sobel(input: torch.Tensor, normalized: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Compute the Sobel gradient magnitude of each channel.

    ``out = sobel(input)`` returns the gradient magnitude of each channel of a ``(B, C, H, W)`` image as an
    edge-strength map of the same shape.

    .. image:: _static/img/sobel.png

    Convention:
        - The output is :math:`\sqrt{g_x^2 + g_y^2 + \epsilon}` per channel, where :math:`(g_x, g_y)` is
          ``spatial_gradient(input, 'sobel', normalized=normalized)``; see the Convention block on
          :func:`~kornia.filters.spatial_gradient`. Away from the replicated border, an axis-aligned ramp of slope
          ``s`` gives :math:`\sqrt{s^2 + \epsilon}`, about :math:`|s|`, with the default ``normalized=True``;
          ``normalized=False`` gives about :math:`8 |s|`, the scale :func:`~kornia.filters.canny` thresholds.
        - ``eps`` sits inside the square root, so a flat region returns :math:`\sqrt{\epsilon}`, ``1e-3`` by
          default, not 0.

    Args:
        input: image with shape ``(B, C, H, W)``.
        normalized: if ``True``, the L1 norm of each Sobel kernel is 1.
        eps: added under the square root so that the gradient of a flat pixel is finite.

    Return:
        the gradient magnitudes, with shape ``(B, C, H, W)``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_edges.html>`__.

    See Also:
        - :class:`Sobel`: the module form.
        - :func:`spatial_gradient`: the signed x and y derivatives, on an added axis.
        - :func:`canny`: thins the edges by non-maximum suppression and thresholds them.
        - :func:`~kornia.morphology.gradient`: a non-linear edge strength, the dilation minus the erosion.

    Example:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = sobel(input)  # 1x3x4x4
        >>> output.shape
        torch.Size([1, 3, 4, 4])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    # comput the x/y gradients
    edges: torch.Tensor = spatial_gradient(input, normalized=normalized)

    # unpack the edges
    gx: torch.Tensor = edges[:, :, 0]
    gy: torch.Tensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)

    return magnitude


class SpatialGradient(nn.Module):
    r"""Compute the first or second image derivatives with a stored kernel set and order (module form).

    ``grad = SpatialGradient(mode, order, normalized)`` stores its arguments, and ``out = grad(input)`` is
    :func:`spatial_gradient` on a ``(B, C, H, W)`` image.

    Convention:
        See the Convention block on :func:`~kornia.filters.spatial_gradient`.

    Args:
        mode: kernel set, ``'sobel'`` or ``'diff'``, case-insensitive.
        order: ``1`` for the first derivatives, ``2`` for the second.
        normalized: whether to scale the kernels to derivative units, as in :func:`spatial_gradient`.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, 2, H, W)`` for ``order=1`` and ``(B, C, 3, H, W)`` for ``order=2``

    See Also:
        - :func:`spatial_gradient`: the one-call form.
        - :class:`SpatialGradient3d`: the same for volumes.
        - :class:`Sobel`: the magnitude of the first derivatives.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = SpatialGradient()(input)  # 1x3x2x4x4

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, 2, -1, -1]

    def __init__(self, mode: str = "sobel", order: int = 1, normalized: bool = True) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order}, normalized={self.normalized}, mode={self.mode})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute horizontal and vertical spatial derivatives.

        The module applies Sobel or difference kernels, depending on
        ``self.mode``, to estimate image gradients along the width and height
        axes. First-order derivatives describe local slope; second-order
        derivatives describe how that slope changes.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Gradient tensor from :func:`spatial_gradient`. For first-order
            derivatives the shape is ``(B, C, 2, H, W)``, where
            the derivative axis of size ``2`` stores the response along
            :math:`x` (width) and :math:`y` (height). For second-order
            derivatives it is ``(B, C, 3, H, W)``, holding
            :math:`(dxx, dxy, dyy)`.
        """
        return spatial_gradient(input, self.mode, self.order, self.normalized)


class SpatialGradient3d(nn.Module):
    r"""Compute the first or second derivatives of a volume with a stored kernel set and order (module form).

    ``grad = SpatialGradient3d(mode, order)`` stores its arguments, and ``out = grad(input)`` is
    :func:`spatial_gradient3d` on a ``(B, C, D, H, W)`` volume.

    Convention:
        See the Convention block on :func:`~kornia.filters.spatial_gradient3d`.

    Args:
        mode: kernel set, case-insensitive; only ``'diff'`` is implemented.
        order: ``1`` for the first derivatives, ``2`` for the second.

    Shape:
        - Input: ``(B, C, D, H, W)``
        - Output: ``(B, C, 3, D, H, W)`` for ``order=1`` and ``(B, C, 6, D, H, W)`` for ``order=2``

    See Also:
        - :func:`spatial_gradient3d`: the one-call form.
        - :class:`SpatialGradient`: the same for images.

    Examples:
        >>> input = torch.rand(1, 4, 2, 4, 4)
        >>> output = SpatialGradient3d()(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1, -1, -1]

    def __init__(self, mode: str = "diff", order: int = 1) -> None:
        super().__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(order={self.order}, mode={self.mode})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute spatial derivatives through depth, height, and width.

        This is the volumetric counterpart of :class:`SpatialGradient`. It
        estimates changes along the three spatial axes of a five-dimensional
        tensor, which is useful for 3D images, videos represented as volumes,
        or feature maps with an explicit depth dimension.

        Args:
            input: Volume tensor with shape ``(B, C, D, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the channel count,
                :math:`D` is the depth, :math:`H` is the height, and
                :math:`W` is the width.

        Returns:
            Derivative tensor from :func:`spatial_gradient3d`. The output keeps
            the input batch and channel dimensions and includes a derivative
            axis for the spatial directions requested by ``self.order``.
        """
        return spatial_gradient3d(input, self.mode, self.order)


class Sobel(nn.Module):
    r"""Compute the Sobel gradient magnitude of each channel (module form).

    ``edges = Sobel(normalized, eps)`` stores its arguments, and ``out = edges(input)`` is :func:`sobel` on a
    ``(B, C, H, W)`` image.

    Convention:
        See the Convention block on :func:`~kornia.filters.sobel`.

    Args:
        normalized: if ``True``, the L1 norm of each Sobel kernel is 1.
        eps: added under the square root, as in :func:`sobel`.

    Shape:
        - Input: ``(B, C, H, W)``
        - Output: ``(B, C, H, W)``

    See Also:
        - :func:`sobel`: the one-call form.
        - :class:`SpatialGradient`: the signed x and y derivatives, on an added axis.
        - :class:`Canny`: thins the edges by non-maximum suppression and thresholds them.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # 1x3x4x4

    """

    def __init__(self, normalized: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalized={self.normalized})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the Sobel gradient magnitude for every image channel.

        Sobel filtering estimates horizontal and vertical derivatives and then
        combines them into a magnitude response. Large values indicate strong
        local intensity changes, which often correspond to edges or texture
        boundaries.

        Args:
            input: Image tensor with shape ``(B, C, H, W)``, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape ``(B, C, H, W)`` containing the gradient
            magnitude for each channel. The ``self.eps`` value is used by the
            underlying function to keep magnitude computation numerically
            stable near zero.
        """
        return sobel(input, self.normalized, self.eps)
