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

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.core.utils import is_autocast_enabled, is_compiling

from .filter import filter2d, filter2d_separable
from .kernels import _unpack_2d_ks, get_box_kernel1d, get_box_kernel2d

_HAS_MKLDNN = torch.backends.mkldnn.is_available()
# From this many elements (e.g. 32 RGB 256x256 images) eager CPU pooling and slice sums
# overtake oneDNN's depthwise convolution; below it oneDNN wins (i7-14700K, warmed threads).
_ONEDNN_LARGE_INPUT = 1 << 22


def _box_blur_pool_eligible(input: torch.Tensor, kernel_size: tuple[int, int] | int, separable: bool) -> bool:
    """Select average pooling where it beats convolution.

    Pooling wins on CUDA and on CPUs without oneDNN. With oneDNN, eager pooling only wins
    on large inputs and compiled pooling only for small windows.
    """
    # Pooling does not participate in autocast in the same way as convolution.
    # Keep the convolution implementation there to preserve the established
    # output dtype and precision contract.
    if not input.is_floating_point() or is_autocast_enabled():
        return False
    if not (input.device.type == "cpu" and _HAS_MKLDNN):
        return True
    size = max(_unpack_2d_ks(kernel_size))
    if is_compiling():
        return size <= (5 if separable else 3)
    return input.numel() >= _ONEDNN_LARGE_INPUT and size <= (15 if separable else 5)


def _needs_convolution_for_extreme_cpu_values(input: torch.Tensor, num_terms: int) -> bool:
    """Keep the eager CPU shortcut for ordinary values, with a rare convolution fallback.

    A single min/max reduction avoids an image-sized finite-value mask. Leave
    captured graphs and function transforms alone: scalar data-dependent dispatch
    cannot be traced or vmapped, and must not synchronize GPU execution.
    """
    if (
        input.device.type != "cpu"
        or num_terms <= 1
        or input.numel() == 0
        or torch.jit.is_scripting()
        or torch.jit.is_tracing()
        or is_compiling()
        or torch._C._are_functorch_transforms_active()
    ):
        return False
    # Leave a factor of two for rounding in the unnormalized accumulation.
    limit = torch.finfo(input.dtype).max / (2 * num_terms)
    minimum, maximum = torch.aminmax(input.detach())
    return not (-limit <= minimum.item() and maximum.item() <= limit)


def _box_blur_pool(
    input: torch.Tensor, kernel_size: tuple[int, int] | int, border_type: str, separable: bool
) -> torch.Tensor:
    """Average local windows without constructing kernels or expanding them over channels."""
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    ky, kx = _unpack_2d_ks(kernel_size)
    KORNIA_CHECK(ky > 0 and kx > 0, f"Kernel dimensions must be positive. Got {kernel_size}")
    KORNIA_CHECK(
        str(border_type).lower() in {"constant", "reflect", "replicate", "circular"},
        f"Invalid border, {border_type}. Expected one of constant, reflect, replicate, circular",
    )

    # A one-dimensional window needs only one pass, even in separable mode.
    windows = ((1, kx), (ky, 1)) if separable and ky > 1 and kx > 1 else ((ky, kx),)
    for height, width in windows:
        if border_type == "constant" and height % 2 == 1 and width % 2 == 1:
            # Native symmetric zero padding avoids allocating a padded image.
            input = F.avg_pool2d(
                input, (height, width), stride=1, padding=(height // 2, width // 2), count_include_pad=True
            )
        else:
            # Even windows need asymmetric padding to retain the input size.
            padding = ((width - 1) // 2, width // 2, (height - 1) // 2, height // 2)
            input = F.avg_pool2d(F.pad(input, padding, mode=border_type), (height, width), stride=1)
    return input


def box_blur(
    input: torch.Tensor, kernel_size: tuple[int, int] | int, border_type: str = "reflect", separable: bool = True
) -> torch.Tensor:
    r"""Blur an image using the box filter.

    .. image:: _static/img/box_blur.png

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        input: the image to blur with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        separable: use two one-dimensional passes (the default), reducing work
          for larger kernels. Floating inputs use average pooling outside autocast
          where it is faster (on CPUs with oneDNN, only for large inputs or small
          compiled windows); complex inputs and autocast use convolution. The dense implementation
          may differ by floating-point roundoff. Ordinary eager CPU execution
          falls back to convolution for extreme input ranges; captured graphs
          and function transforms retain native pooling arithmetic.

    Returns:
        the blurred torch.Tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])

    """
    KORNIA_CHECK_IS_TENSOR(input)

    if _box_blur_pool_eligible(input, kernel_size, separable):
        ky, kx = _unpack_2d_ks(kernel_size)
        num_terms = max(ky, kx) if separable else ky * kx
        if not _needs_convolution_for_extreme_cpu_values(input, num_terms):
            return _box_blur_pool(input, kernel_size, border_type, separable)

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        kernel_y = get_box_kernel1d(ky, device=input.device, dtype=input.dtype)
        kernel_x = get_box_kernel1d(kx, device=input.device, dtype=input.dtype)
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
        out = filter2d(input, kernel, border_type)

    return out


class BoxBlur(nn.Module):
    r"""Blur an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: use two one-dimensional passes (the default), reducing work
          for larger kernels. Floating inputs use average pooling outside autocast
          where it is faster (on CPUs with oneDNN, only for large inputs or small
          compiled windows); complex inputs and autocast use convolution. The dense implementation
          may differ by floating-point roundoff. Ordinary eager CPU execution
          falls back to convolution for extreme input ranges; captured graphs
          and function transforms retain native pooling arithmetic.

    Returns:
        the blurred input torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])

    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, border_type: str = "reflect", separable: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.separable = separable

        ky, kx = _unpack_2d_ks(kernel_size)
        KORNIA_CHECK(ky > 0 and kx > 0, f"Kernel dimensions must be positive. Got {kernel_size}")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Older releases persisted derived box kernels. Like NMS, accept these
        # keys for strict loading, including when nested in another module.
        # BoxBlur now always represents a fixed uniform average.
        for name in ("kernel", "kernel_x", "kernel_y"):
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Average each pixel with its local neighborhood.

        The box blur uses a rectangular averaging kernel where every location
        inside the window has the same weight. It is a simple low-pass filter:
        high-frequency details are reduced, while broader image structures are
        preserved. Depending on the module configuration, the kernel is applied
        either as a full two-dimensional filter or as two separable one-
        dimensional passes.

        Args:
            input: Image or feature tensor with shape :math:`(B, C, H, W)`,
                where :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the locally
            averaged result. The output keeps the same batch, channel, and
            spatial layout as ``input``; border pixels are handled according to
            the configured border mode.
        """
        return box_blur(input, self.kernel_size, self.border_type, self.separable)
