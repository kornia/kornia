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

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.core.utils import is_autocast_enabled, is_compiling

from .blur import _HAS_MKLDNN, _ONEDNN_LARGE_INPUT, _needs_convolution_for_extreme_cpu_values
from .filter import filter2d
from .kernels import (
    _check_kernel_size,
    _check_laplacian_kernel_size,
    _unpack_2d_ks,
    get_laplacian_kernel2d,
    normalize_kernel2d,
)


def _laplacian_slices_eligible(input: torch.Tensor) -> bool:
    """Select the slice implementation where it beats depthwise convolution.

    On CPU the slices win, except against oneDNN on large inputs. On CUDA Inductor fuses
    them into one kernel, but eager slices lose to cuDNN at larger batches.
    """
    if is_autocast_enabled() or input.dtype not in (torch.float32, torch.float64):
        return False
    if input.device.type == "cpu":
        return not _HAS_MKLDNN or input.numel() < _ONEDNN_LARGE_INPUT
    return input.device.type == "cuda" and is_compiling()


def _check_laplacian_size(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    """Unpack ``kernel_size`` and reject what no Laplacian kernel can have: even, non-positive or 1x1 sizes."""
    ky, kx = _unpack_2d_ks(kernel_size)
    _check_kernel_size((ky, kx))
    _check_laplacian_kernel_size((ky, kx))
    return ky, kx


def laplacian(
    input: torch.Tensor, kernel_size: tuple[int, int] | int, border_type: str = "reflect", normalized: bool = True
) -> torch.Tensor:
    r"""Create an operator that returns a tensor using a Laplacian filter.

    .. image:: _static/img/laplacian.png

    The operator filters each channel of the given tensor with a Laplacian kernel.
    It supports batched operation.

    Convention:
        - The kernel is ``get_laplacian_kernel2d(kernel_size)``, correlated with each channel as by
          :func:`~kornia.filters.filter2d`; see the Convention blocks on
          :func:`~kornia.filters.get_laplacian_kernel2d` for the stencil, its sign and ``kernel_size``, and on
          :func:`~kornia.filters.filter2d` for the border modes.
        - ``normalized=True``, the default, divides the kernel by its absolute sum :math:`2 (kH \cdot kW - 1)`, 16
          for size 3. Unlike ``normalized`` in :func:`~kornia.filters.spatial_gradient`, this does not give
          derivative units: size 3 returns :math:`3 \nabla^2 / 16`. :ref:`Filtering <filtering-conventions>`
          compares both scales with scipy and OpenCV.
        - Known defects:

          - ``kernel_size=1`` passes validation, and the normalised :math:`1 \times 1` kernel is ``0 / 0``, so the
            output is all NaN (`#5175 <https://github.com/kornia/kornia/issues/5175>`_).
          - an integer input casts the kernel to its dtype, as :func:`~kornia.filters.filter2d` does, so the
            normalised kernel truncates to 0 and the output is all zeros
            (`#5155 <https://github.com/kornia/kornia/issues/5155>`_).
          - ``border_type`` is checked case-insensitively but used as given, so ``'REFLECT'`` raises
            (`#5156 <https://github.com/kornia/kornia/issues/5156>`_).

    Args:
        input: the input image tensor with shape :math:`(B, C, H, W)`.
        kernel_size: the size of the kernel. It should be odd and positive, and at least 3 along one axis.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Return:
        the Laplacian response with shape :math:`(B, C, H, W)`.

    Raises:
        BaseError: if a size is even or not positive, if ``kernel_size`` is a sequence of other than 2 sizes, or if
            it is ``1`` or ``(1, 1)``: a :math:`1 \times 1` kernel is all zeros, and its normalized form is ``0 / 0``.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = laplacian(input, 3)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        str(border_type).lower() in {"constant", "reflect", "replicate", "circular"},
        f"Invalid border, {border_type}. Expected one of {{'constant', 'reflect', 'replicate', 'circular'}}",
    )
    # the check is case-insensitive, so pad with the lower-case spelling as well
    border_type = str(border_type).lower()

    ky, kx = _check_laplacian_size(kernel_size)

    if not _laplacian_slices_eligible(input) or _needs_convolution_for_extreme_cpu_values(input, ky * kx):
        kernel = get_laplacian_kernel2d((ky, kx), device=input.device, dtype=input.dtype)[None]
        if normalized:
            kernel = normalize_kernel2d(kernel)
        return filter2d(input, kernel, border_type)

    # The Laplacian kernel contains ones everywhere except at its centre,
    # which is ``1 - ky * kx``. Compute its response as the sum of each
    # neighbourhood minus ``ky * kx`` times the centre instead of materializing
    # and depthwise-convolving the dense kernel. Summing each axis first needs
    # only ``ky + kx - 2`` elementwise additions and lets torch.compile fuse it.
    scale = 2 * (ky * kx - 1)
    if normalized:
        # Scale before summing so large finite inputs cannot overflow on an otherwise
        # finite normalized response.
        input = input / scale

    padded = F.pad(input, (kx // 2, kx // 2, ky // 2, ky // 2), mode=border_type)
    height, width = input.shape[-2:]

    rows = padded[..., :height, :]
    for offset in range(1, ky):
        rows = rows + padded[..., offset : offset + height, :]

    output = rows[..., :, :width]
    for offset in range(1, kx):
        output = output + rows[..., :, offset : offset + width]

    return output - (ky * kx) * input


class Laplacian(nn.Module):
    r"""Create an operator that returns a tensor using a Laplacian filter.

    The operator filters each channel of the given tensor with a Laplacian kernel.
    It supports batched operation.

    Convention:
        See the Convention block on :func:`~kornia.filters.laplacian`.

    Args:
        kernel_size: the size of the kernel. It should be odd and positive, and at least 3 along one axis.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Raises:
        BaseError: if a size is even or not positive, if ``kernel_size`` is a sequence of other than 2 sizes, or if
            it is ``1`` or ``(1, 1)``. The size is checked when the module is built, as
            :func:`~kornia.filters.laplacian` checks it when called.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = Laplacian(5)
        >>> output = laplace(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])

    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, border_type: str = "reflect", normalized: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized

        _check_laplacian_size(kernel_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"normalized={self.normalized}, "
            f"border_type={self.border_type})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the second-order Laplacian response of an image tensor.

        The Laplacian filter measures rapid local intensity changes by
        combining second derivatives along the spatial axes. It is commonly
        used for edge detection, focus measures, and highlighting fine image
        detail.

        Args:
            input: Image tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the Laplacian
            response for each batch item and channel. Positive and negative
            values represent opposite directions of local curvature.
        """
        return laplacian(input, self.kernel_size, self.border_type, self.normalized)
