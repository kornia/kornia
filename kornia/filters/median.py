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
from kornia.core.utils import is_autocast_enabled, is_compiling, is_exporting

from .kernels import _unpack_2d_ks, get_binary_kernel2d

_VALID_BORDERS = {"constant", "reflect", "replicate", "circular"}


def _median_network(size: int) -> tuple[tuple[int, int, bool, bool], ...]:
    """Prune Batcher's odd-even mergesort to the middle output wire.

    Batcher, "Sorting networks and their applications", AFIPS 1968,
    doi:10.1145/1468075.1468121. Missing wires are symbolic positive infinity.
    """
    pairs: list[tuple[int, int]] = []

    def merge(start: int, length: int, stride: int) -> None:
        step = 2 * stride
        if step < length:
            merge(start, length, step)
            merge(start + stride, length, step)
            pairs.extend((i, i + stride) for i in range(start + stride, start + length - stride, step))
        else:
            pairs.append((start, start + stride))

    def sort(start: int, length: int) -> None:
        if length > 1:
            sort(start, length // 2)
            sort(start + length // 2, length // 2)
            merge(start, length, 1)

    sort(0, 1 << (size - 1).bit_length())
    needed = {size // 2}
    result = []
    for left, right in reversed(pairs):
        if right >= size:  # Comparing a real wire with positive infinity is a no-op.
            continue
        low, high = left in needed, right in needed
        if low or high:
            result.append((left, right, low, high))
            needed.update((left, right))
    return tuple(reversed(result))


_MEDIAN_NETWORKS = {3: _median_network(9), 5: _median_network(25)}


def _median_blur_network(input: torch.Tensor, size: int, border_type: str) -> torch.Tensor:
    """Select a small-window median without materializing patches or sorting them."""
    radius = size // 2
    padded = F.pad(input, (radius, radius, radius, radius), mode=border_type)
    height, width = input.shape[-2:]
    values = [padded[..., y : y + height, x : x + width] for y in range(size) for x in range(size)]
    for left, right, low, high in _MEDIAN_NETWORKS[size]:
        a, b = values[left], values[right]
        if low:
            values[left] = torch.minimum(a, b)
        if high:
            values[right] = torch.maximum(a, b)
    # The original one-hot convolution propagates any NaN/Inf in a window to
    # every extracted feature (including 0 * Inf). Pool a finite-value mask:
    # max-pooling NaNs directly is version- and dtype-dependent on CPU.
    invalid = F.max_pool2d((~torch.isfinite(input)).to(input.dtype), size, stride=1, padding=radius).bool()
    selected = values[size * size // 2]
    return torch.where(invalid, torch.full_like(selected, float("nan")), selected).contiguous()


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    r"""Compute zero padding tuple."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def median_blur(input: torch.Tensor, kernel_size: tuple[int, int] | int, border_type: str = "reflect") -> torch.Tensor:
    r"""Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before filtering.
        The expected modes are: `'constant'`, `'reflect'`, `'replicate'` or `'circular'`.

    Returns:
        the blurred input torch.Tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    border_type_lower = str(border_type).lower()
    KORNIA_CHECK(
        border_type_lower in _VALID_BORDERS,
        f"Invalid border, {border_type}. Expected one of {_VALID_BORDERS}",
    )
    border_type = border_type_lower

    ky, kx = _unpack_2d_ks(kernel_size)
    if input.shape[1] == 0:
        return input
    # ATen's per-pixel median reduction dominates inference for small windows.
    # A fixed selection network avoids it. Inductor fuses the network into one
    # CUDA kernel; eager CUDA launches one kernel per comparator, which only pays
    # off for 3x3 windows on large inputs. Keep the original path for autograd
    # (in particular its tie indices) and other devices/sizes.
    on_cuda = input.device.type == "cuda" and (is_compiling() or (ky == 3 and input.numel() >= 1 << 20))
    if (
        (input.device.type == "cpu" or on_cuda)
        and not is_autocast_enabled()
        and input.shape[1] != 0
        and input.is_floating_point()
        and not input.requires_grad
        and torch.autograd.forward_ad.unpack_dual(input).tangent is None
        and ky == kx
        and ky in _MEDIAN_NETWORKS
    ):
        return _median_blur_network(input, ky, border_type)

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
    b, c, h, w = input.shape

    # map the local window to single vector
    input = F.pad(input, (padding[1], padding[1], padding[0], padding[0]), mode=border_type)
    features: torch.Tensor = F.conv2d(
        input.reshape(b * c, 1, h + 2 * padding[0], w + 2 * padding[1]),
        kernel,
        padding=0,
        stride=1,
    )

    features = features.view(b, c, ky * kx, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    if is_exporting():
        # ``median.dim`` has no ONNX lowering; a sort picks the same (lower-middle) element.
        return features.sort(dim=2)[0][:, :, (ky * kx - 1) // 2]
    return features.median(dim=2)[0]


class MedianBlur(nn.Module):
    r"""Blur an image using the median filter.

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before filtering.
            The expected modes are: `'constant'`, `'reflect'`, `'replicate'` or `'circular'`.

    Returns:
        the blurred input torch.Tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])

    """

    def __init__(self, kernel_size: tuple[int, int] | int, border_type: str = "reflect") -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Replace each pixel with the median value in its local window.

        Median filtering is a non-linear smoothing operation. Instead of
        averaging neighboring values, it sorts the values inside the kernel
        window and chooses the middle value. This makes it effective for
        reducing impulse-like noise while preserving sharper boundaries than a
        simple mean filter.

        Args:
            input: Image or feature tensor with shape :math:`(B, C, H, W)`,
                where :math:`B` is the batch size, :math:`C` is the number of
                channels, :math:`H` is the height, and :math:`W` is the width.

        Returns:
            Tensor with shape :math:`(B, C, H, W)` containing the median-
            filtered result for each channel independently.
        """
        return median_blur(input, self.kernel_size, self.border_type)
