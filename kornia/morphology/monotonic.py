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
import torch.nn as nn
import torch.nn.functional as F


def _sliding_window_max_1d_impl(
    x: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes 1D sliding window maximum and argmax indices.

    Input: (..., L)
    Returns: (max_vals, argmax_indices)
    """
    orig_shape = x.shape
    L = orig_shape[-1]
    flat_x = x.reshape(-1, L)

    if padding > 0:
        padded_x = F.pad(flat_x, (padding, padding), mode="constant", value=float("-inf"))
    else:
        padded_x = flat_x

    padded_L = padded_x.shape[-1]
    out_L = (padded_L - kernel_size) // stride + 1

    windows = padded_x.unfold(dimension=-1, size=kernel_size, step=stride)
    max_vals, rel_indices = torch.max(windows, dim=-1)

    starts = torch.arange(0, out_L * stride, stride, device=x.device, dtype=torch.int64).unsqueeze(0)
    abs_indices = starts + rel_indices - padding
    abs_indices = torch.clamp(abs_indices, 0, L - 1)

    out_shape = list(orig_shape[:-1]) + [out_L]
    return max_vals.reshape(out_shape), abs_indices.reshape(out_shape)


class _MonotonicExtrema1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, kernel_size: int, stride: int, padding: int, is_dilation: bool):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.is_dilation = is_dilation
        ctx.input_shape = input.shape

        x = input if is_dilation else -input
        out_vals, out_indices = _sliding_window_max_1d_impl(x, kernel_size, stride, padding)

        if not is_dilation:
            out_vals = -out_vals

        ctx.save_for_backward(out_indices)
        return out_vals

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out_indices,) = ctx.saved_tensors
        orig_shape = ctx.input_shape

        grad_input = torch.zeros(orig_shape, dtype=grad_output.dtype, device=grad_output.device)

        L = orig_shape[-1]
        out_L = out_indices.shape[-1]

        flat_grad_in = grad_input.reshape(-1, L)
        flat_grad_out = grad_output.reshape(-1, out_L)
        flat_indices = out_indices.reshape(-1, out_L)

        flat_grad_in.scatter_add_(dim=-1, index=flat_indices, src=flat_grad_out)
        return grad_input, None, None, None, None


def dilation1d(
    input: torch.Tensor, kernel_size: int, stride: int = 1, padding: Optional[int] = None
) -> torch.Tensor:
    r"""Applies 1D morphological dilation using an O(N) monotonic sliding window.

    Args:
        input: Input tensor of shape (B, C, L) or (C, L) or (L,).
        kernel_size: Size of the 1D structuring element.
        stride: Stride of the window (default: 1).
        padding: Padding added to both sides. If None, padding = kernel_size // 2 (same-size output).

    Returns:
        Dilated tensor with preserved feature dimensions.
    """
    if padding is None:
        padding = kernel_size // 2
    return _MonotonicExtrema1dFunction.apply(input, kernel_size, stride, padding, True)


def erosion1d(
    input: torch.Tensor, kernel_size: int, stride: int = 1, padding: Optional[int] = None
) -> torch.Tensor:
    r"""Applies 1D morphological erosion using an O(N) monotonic sliding window.

    Args:
        input: Input tensor of shape (B, C, L) or (C, L) or (L,).
        kernel_size: Size of the 1D structuring element.
        stride: Stride of the window (default: 1).
        padding: Padding added to both sides. If None, padding = kernel_size // 2.

    Returns:
        Eroded tensor.
    """
    if padding is None:
        padding = kernel_size // 2
    return _MonotonicExtrema1dFunction.apply(input, kernel_size, stride, padding, False)


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
    else:
        pad_h, pad_w = padding

    out_h = dilation1d(input, kernel_size=kw, stride=1, padding=pad_w)
    out_trans = out_h.transpose(-2, -1)
    out_v = dilation1d(out_trans, kernel_size=kh, stride=1, padding=pad_h)
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
    else:
        pad_h, pad_w = padding

    out_h = erosion1d(input, kernel_size=kw, stride=1, padding=pad_w)
    out_trans = out_h.transpose(-2, -1)
    out_v = erosion1d(out_trans, kernel_size=kh, stride=1, padding=pad_h)
    return out_v.transpose(-2, -1)
