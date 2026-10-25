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

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def dilation1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    mode: str = "constant",
    value: float = -float("inf"),
) -> torch.Tensor:
    r"""Applies 1D morphological dilation over input tensor.

    Args:
        input: Input tensor of shape (B, C, L) or (C, L) or (L,).
        kernel_size: Size of the 1D flat structuring element.
        stride: Stride of the window (default: 1).
        padding: Padding added to both sides. If None, padding = kernel_size // 2 (same-size output).
            Can be int for symmetric padding or a tuple (pad_left, pad_right) for asymmetric padding.
        mode: Padding mode for :func:`torch.nn.functional.pad`.
        value: Fill value for constant padding.

    Returns:
        Dilated tensor with preserved feature dimensions.
    """
    if isinstance(padding, int):
        pad_left = pad_right = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError("padding tuple must have length 2")
        pad_left = int(padding[0])
        pad_right = int(padding[1])
    elif padding is None:
        p = kernel_size // 2
        pad_left = pad_right = p
    else:
        raise TypeError("padding must be int, tuple of two ints, or None")

    orig_shape = input.shape
    L = orig_shape[-1]
    flat_x = input.reshape(-1, 1, L)

    if pad_left != 0 or pad_right != 0:
        padded = F.pad(flat_x, [pad_left, pad_right], mode=mode, value=value)
    else:
        padded = flat_x

    out = F.max_pool1d(padded, kernel_size=kernel_size, stride=stride)
    out_shape = list(orig_shape[:-1]) + [out.shape[-1]]
    return out.reshape(out_shape)


def erosion1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    mode: str = "constant",
    value: float = float("inf"),
) -> torch.Tensor:
    r"""Applies 1D morphological erosion over input tensor.

    Args:
        input: Input tensor of shape (B, C, L) or (C, L) or (L,).
        kernel_size: Size of the 1D flat structuring element.
        stride: Stride of the window (default: 1).
        padding: Padding added to both sides. If None, padding = kernel_size // 2.
            Can be int for symmetric padding or a tuple (pad_left, pad_right) for asymmetric padding.
        mode: Padding mode for :func:`torch.nn.functional.pad`.
        value: Fill value for constant padding.

    Returns:
        Eroded tensor.
    """
    return -dilation1d(-input, kernel_size=kernel_size, stride=stride, padding=padding, mode=mode, value=-value)


def separable_dilation2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    padding: Optional[Union[int, Tuple[int, int]]] = None,
) -> torch.Tensor:
    r"""Applies 2D separable morphological dilation using orthogonal 1D monotonic passes.

    For rectangular structuring elements, 2D dilation decomposes into:
        Dilation_2D(I, K) = Dilation_V(Dilation_H(I, K_w), K_h)

    This reduces memory overhead and computes in linear time per pixel.

    Args:
        input: Image tensor of shape (B, C, H, W).
        kernel_size: (k_h, k_w) or int for square kernel.
        padding: (pad_h, pad_w) or None for same-size padding.

    Returns:
        Dilated 2D tensor of shape (B, C, H, W).
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if padding is None:
        pad_h, pad_w = kh // 2, kw // 2
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError("padding must be int or tuple of length 2")
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        raise TypeError("padding must be int, tuple of two ints, or None")

    out_h = dilation1d(input, kernel_size=kw, stride=1, padding=pad_w, mode="constant", value=-float("inf"))
    out_trans = out_h.transpose(-2, -1)
    out_v = dilation1d(out_trans, kernel_size=kh, stride=1, padding=pad_h, mode="constant", value=-float("inf"))
    return out_v.transpose(-2, -1)


def separable_erosion2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    padding: Optional[Union[int, Tuple[int, int]]] = None,
) -> torch.Tensor:
    r"""Applies 2D separable morphological erosion using orthogonal 1D monotonic passes.

    Args:
        input: Image tensor of shape (B, C, H, W).
        kernel_size: (k_h, k_w) or int for square kernel.
        padding: (pad_h, pad_w) or None for same-size padding.

    Returns:
        Eroded 2D tensor of shape (B, C, H, W).
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if padding is None:
        pad_h, pad_w = kh // 2, kw // 2
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError("padding must be int or tuple of length 2")
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        raise TypeError("padding must be int, tuple of two ints, or None")

    out_h = erosion1d(input, kernel_size=kw, stride=1, padding=pad_w, mode="constant", value=float("inf"))
    out_trans = out_h.transpose(-2, -1)
    out_v = erosion1d(out_trans, kernel_size=kh, stride=1, padding=pad_h, mode="constant", value=float("inf"))
    return out_v.transpose(-2, -1)
