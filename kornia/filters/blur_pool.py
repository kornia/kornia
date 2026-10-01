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

import operator

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE

from .kernels import _check_kernel_size, get_pascal_kernel_2d

__all__ = [
    "BlurPool2D",
    "EdgeAwareBlurPool2D",
    "MaxBlurPool2D",
    "blur_pool2d",
    "edge_aware_blur_pool2d",
    "max_blur_pool2d",
]


class BlurPool2D(nn.Module):
    r"""Compute blur (anti-aliasing) and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lceil\frac{H}{\text{stride}}\right\rceil, \quad
              W_{out} = \left\lceil\frac{W}{\text{stride}}\right\rceil

    Examples:
        >>> from kornia.filters.blur_pool import BlurPool2D
        >>> input = torch.eye(5)[None, None]
        >>> bp = BlurPool2D(kernel_size=3, stride=2)
        >>> bp(input)
        tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])

    """

    def __init__(self, kernel_size: tuple[int, int] | int, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Downsample a feature map after applying an anti-aliasing blur.

        Blur pooling first smooths the input with a normalized Pascal kernel
        and then samples it with the configured stride. The blur step reduces
        aliasing artifacts that can appear when high-frequency image or feature
        content is subsampled directly.

        Args:
            input: Feature map tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the input height, and :math:`W` is the
                input width.

        Returns:
            Downsampled tensor of shape :math:`(B, C, H_{out}, W_{out})`, with
            the sizes given in the Shape section of the class. They depend only
            on the input size and ``self.stride``.
        """
        self.kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _blur_pool_by_kernel2d(input, self.kernel.repeat((input.shape[1], 1, 1, 1)), self.stride)


class MaxBlurPool2D(nn.Module):
    r"""Compute pools and blurs and downsample a given feature map.

    Equivalent to ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lceil\frac{H - \text{max\_pool\_size} + 1}{\text{stride}}\right\rceil, \quad
              W_{out} = \left\lceil\frac{W - \text{max\_pool\_size} + 1}{\text{stride}}\right\rceil

    Returns:
        torch.Tensor: the transformed torch.tensor.

    Examples:
        >>> import torch.nn as nn
        >>> from kornia.filters.blur_pool import BlurPool2D
        >>> input = torch.eye(5)[None, None]
        >>> mbp = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)
        >>> mbp(input)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
        >>> seq = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), BlurPool2D(kernel_size=3, stride=2))
        >>> seq(input)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])

    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, stride: int = 2, max_pool_size: int = 2, ceil_mode: bool = False
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply max pooling and then anti-aliased blur downsampling.

        This layer keeps the local peak response with max pooling before
        applying the blur-pool downsampling step. It is useful in convolutional
        networks where pooling should remain less sensitive to small spatial
        shifts while still reducing the feature resolution.

        Args:
            input: Feature map tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the channel count,
                :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor of shape :math:`(B, C, H_{out}, W_{out})` after max pooling
            and blur pooling, with the sizes given in the Shape section of the
            class. They depend only on the input size, ``self.max_pool_size``
            and ``self.stride``.
        """
        self.kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _max_blur_pool_by_kernel2d(
            input, self.kernel.repeat((input.size(1), 1, 1, 1)), self.stride, self.max_pool_size, self.ceil_mode
        )


class EdgeAwareBlurPool2D(nn.Module):
    """Apply an edge-aware anti-aliasing filter during downsampling.

    This module performs blur pooling while preserving edges by using an
    edge-intensity threshold.

    Args:
        kernel_size: The size of the Gaussian blur kernel.
        edge_threshold: The threshold for detecting edges. Default: 1.25.
        edge_dilation_kernel_size: The kernel size for dilating the edge map. It must be an odd positive integer.
            Default: 3.
    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, edge_threshold: float = 1.25, edge_dilation_kernel_size: int = 3
    ) -> None:
        super().__init__()
        edge_dilation_kernel_size = operator.index(edge_dilation_kernel_size)
        _check_kernel_size(edge_dilation_kernel_size)
        self.kernel_size = kernel_size
        self.edge_threshold = edge_threshold
        self.edge_dilation_kernel_size = edge_dilation_kernel_size

    def forward(self, input: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Downsample while adapting the blur near image or feature edges.

        Edge-aware blur pooling estimates where strong local changes occur and
        uses that information to avoid over-smoothing across boundaries before
        reducing the spatial resolution. The operation is designed for feature
        maps where edges should remain sharp after downsampling.

        Args:
            input: Feature map tensor with shape :math:`(B, C, H, W)`, where
                :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the input height, and :math:`W` is the
                input width.
            epsilon: Small positive value used to keep edge-aware divisions
                numerically stable when local normalization terms are close to
                zero.

        Returns:
            Edge-aware downsampled tensor with shape
            :math:`(B, C, H_{out}, W_{out})`. The output preserves channel
            order while reducing the spatial dimensions according to the
            configured pooling parameters.
        """
        return edge_aware_blur_pool2d(
            input, self.kernel_size, self.edge_threshold, self.edge_dilation_kernel_size, epsilon
        )


def blur_pool2d(input: torch.Tensor, kernel_size: tuple[int, int] | int, stride: int = 2) -> torch.Tensor:
    r"""Compute blurs and downsample a given feature map.

    .. image:: _static/img/blur_pool2d.png

    See :class:`~kornia.filters.BlurPool2D` for details.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        input: torch.Tensor to apply operation to.
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lceil\frac{H}{\text{stride}}\right\rceil, \quad
              W_{out} = \left\lceil\frac{W}{\text{stride}}\right\rceil

    Returns:
        the transformed torch.Tensor.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_operators.html>`__.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> blur_pool2d(input, 3)
        tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])

    """
    kernel = get_pascal_kernel_2d(kernel_size, norm=True, device=input.device, dtype=input.dtype).repeat(
        (input.size(1), 1, 1, 1)
    )
    return _blur_pool_by_kernel2d(input, kernel, stride)


def max_blur_pool2d(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    stride: int = 2,
    max_pool_size: int = 2,
    ceil_mode: bool = False,
) -> torch.Tensor:
    r"""Compute pools and blurs and downsample a given feature map.

    .. image:: _static/img/max_blur_pool2d.png

    See :class:`~kornia.filters.MaxBlurPool2D` for details.

    Args:
        input: torch.Tensor to apply operation to.
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_operators.html>`__.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> max_blur_pool2d(input, 3)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel = get_pascal_kernel_2d(kernel_size, norm=True, device=input.device, dtype=input.dtype).repeat(
        (input.shape[1], 1, 1, 1)
    )
    return _max_blur_pool_by_kernel2d(input, kernel, stride, max_pool_size, ceil_mode)


def _blur_pool_conv2d(input: torch.Tensor, kernel: torch.Tensor, stride: int) -> torch.Tensor:
    """Correlate every channel with its kernel at ``stride``, zero-padded so the output keeps ``ceil(H / stride)`` rows.

    The padding is ``(k - 1) // 2`` pixels before and ``k // 2`` after along each spatial axis, the amounts
    antialiased-cnns uses. For an odd ``k`` the two are equal and ``F.conv2d`` pads itself. An even ``k`` needs one
    pixel more after than before, which ``F.pad`` adds.
    """
    ky, kx = kernel.shape[-2], kernel.shape[-1]
    if ky % 2 == 1 and kx % 2 == 1:
        return F.conv2d(input, kernel, padding=((ky - 1) // 2, (kx - 1) // 2), stride=stride, groups=input.shape[1])
    input = F.pad(input, ((kx - 1) // 2, kx // 2, (ky - 1) // 2, ky // 2))
    return F.conv2d(input, kernel, stride=stride, groups=input.shape[1])


def _blur_pool_by_kernel2d(input: torch.Tensor, kernel: torch.Tensor, stride: int) -> torch.Tensor:
    """Compute blur_pool by a given :math:`CxC_{out}xNxN` kernel."""
    KORNIA_CHECK(
        len(kernel.shape) == 4 and kernel.shape[-2] == kernel.shape[-1],
        f"Invalid kernel shape. Expect CxC_(out, None)xNxN, Got {kernel.shape}",
    )

    return _blur_pool_conv2d(input, kernel, stride)


def _max_blur_pool_by_kernel2d(
    input: torch.Tensor, kernel: torch.Tensor, stride: int, max_pool_size: int, ceil_mode: bool
) -> torch.Tensor:
    """Compute max_blur_pool by a given :math:`CxC_(out, None)xNxN` kernel."""
    KORNIA_CHECK(
        len(kernel.shape) == 4 and kernel.shape[-2] == kernel.shape[-1],
        f"Invalid kernel shape. Expect CxC_outxNxN, Got {kernel.shape}",
    )
    # compute local maxima
    input = F.max_pool2d(input, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode)
    # blur and downsample
    return _blur_pool_conv2d(input, kernel, stride)


def _reflect_pad2d(input: torch.Tensor, padding_y: int, padding_x: int) -> torch.Tensor:
    """Reflect-pad by arbitrary amounts, applying chunks accepted by ``F.pad``."""
    remaining_y, remaining_x = padding_y, padding_x
    while remaining_y > 0 or remaining_x > 0:
        pad_y = min(remaining_y, input.shape[-2] - 1)
        pad_x = min(remaining_x, input.shape[-1] - 1)
        input = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        remaining_y -= pad_y
        remaining_x -= pad_x
    return input


def edge_aware_blur_pool2d(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    edge_threshold: float = 1.25,
    edge_dilation_kernel_size: int = 3,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""Blur the input torch.Tensor while maintaining its edges.

    Args:
        input: the input image to blur with shape :math:`(B, C, H, W)`.
        kernel_size: the kernel size for max pooling.
        edge_threshold: positive threshold for the edge decision rule; edge/non-edge.
        edge_dilation_kernel_size: the kernel size for dilating the edges. It must be an odd positive integer.
        epsilon: for numerical stability.

    Returns:
        The blurred torch.Tensor of shape :math:`(B, C, H, W)`.

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    edge_dilation_kernel_size = operator.index(edge_dilation_kernel_size)
    _check_kernel_size(edge_dilation_kernel_size)
    KORNIA_CHECK(edge_threshold > 0.0, f"edge threshold should be positive, but got '{edge_threshold}'")

    # Keep the edge comparison's fixed 2-pixel halo separate from the blur halo. The
    # blur_pool2d convolution zero-pads by its kernel radius, so reflect-padding by at
    # least that radius prevents zeros from reaching the retained image boundary.
    edge_input = F.pad(input, (2, 2, 2, 2), mode="reflect")
    kernel_size_y, kernel_size_x = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    blur_pad_y, blur_pad_x = max(2, kernel_size_y // 2), max(2, kernel_size_x // 2)
    blur_input = _reflect_pad2d(input, blur_pad_y, blur_pad_x)
    # The 2D Pascal kernel sum can overflow half precision for larger kernels
    # (e.g. 9x9 sums to 65536), producing an all-zero normalized kernel.
    blur_dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
    blurred_input = blur_pool2d(blur_input.to(dtype=blur_dtype), kernel_size=kernel_size, stride=1).to(input.dtype)

    # calculate the edges (add epsilon to avoid taking the log of 0)
    log_input, log_thresh = (edge_input + epsilon).log2(), (torch.tensor(edge_threshold)).log2()
    edges_x = log_input[..., :, 4:] - log_input[..., :, :-4]
    edges_y = log_input[..., 4:, :] - log_input[..., :-4, :]
    edges_x, edges_y = edges_x.mean(dim=-3, keepdim=True), edges_y.mean(dim=-3, keepdim=True)
    edges_x_mask, edges_y_mask = edges_x.abs() > log_thresh.to(edges_x), edges_y.abs() > log_thresh.to(edges_y)
    edges_xy_mask = (edges_x_mask[..., 2:-2, :] + edges_y_mask[..., :, 2:-2]).type_as(input)

    # dilate the content edges to have a soft mask of edges
    dilated_edges = F.max_pool3d(edges_xy_mask, edge_dilation_kernel_size, 1, edge_dilation_kernel_size // 2)

    # slice the padded regions
    blurred_input = blurred_input[..., blur_pad_y:-blur_pad_y, blur_pad_x:-blur_pad_x]

    # fuse the input image on edges and blurry input everywhere else
    return dilated_edges * input + (1.0 - dilated_edges) * blurred_input
