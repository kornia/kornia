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

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters.kernels import normalize_kernel2d

_VALID_BORDERS = {"constant", "reflect", "replicate", "circular"}
_VALID_PADDING = {"valid", "same"}
_VALID_BEHAVIOUR = {"conv", "corr"}


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv2d,
        or true convolution (kernel is flipped). 2 modes available ``'corr'`` or ``'conv'``.


    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "H", "W"])

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, {border_type}. Expected one of {_VALID_BORDERS}",
    )
    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, {padding}. Expected one of {_VALID_PADDING}",
    )
    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid padding mode, {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )
    # prepare kernel
    b, c, h, w = input.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Convolve a tensor with two 1d kernels, in x and y directions.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel_x: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kW)` or :math:`(B, kW)`.
        kernel_y: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH)` or :math:`(B, kH)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)

        >>> filter2d_separable(input, kernel, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    out_x = filter2d(input, kernel_x[..., None, :], border_type, normalized, padding)
    out = filter2d(out_x, kernel_y[..., None], border_type, normalized, padding)
    return out


def filter3d(
    input: torch.Tensor, kernel: torch.Tensor, border_type: str = "replicate", normalized: bool = False
) -> torch.Tensor:
    r"""Convolve a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.

    Return:
        the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])

    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "D", "H", "W"])
    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "D", "H", "W"])

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, gotcha {border_type}. Expected one of {_VALID_BORDERS}",
    )

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, d, h, w)


def fft_conv(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
    behaviour: str = "corr",
) -> torch.Tensor:
    r"""Apply a 2D convolution (or correlation) using an FFT-based backend.

    This function applies a spatial kernel to a batched tensor using the
    convolution theorem, i.e., convolution in the spatial domain is performed
    as element-wise multiplication in the frequency domain.

    The kernel is applied independently to each channel of the input tensor.
    Depending on the selected padding mode, the output can either preserve
    the input spatial resolution (`'same'`) or return only the valid region
    (`'valid'`). Boundary handling is performed in the spatial domain prior
    to the FFT.

    Args:
        input: Input tensor of shape :math:`(B, C, H, W)`.
        kernel: Convolution kernel of shape :math:`(B, kH, kW)`. Each batch
            element provides one kernel, which is shared across all channels
            of the corresponding input batch.
        border_type: Padding mode applied to the input before convolution.
            Supported values are ``'constant'``, ``'reflect'``,
            ``'replicate'``, and ``'circular'``.
        normalized: If ``True``, the kernel is L1-normalized before applying
            the convolution.
        padding: Padding strategy to use. Supported values are:
            ``'same'`` (output has the same spatial size as the input) or
            ``'valid'`` (no implicit padding).
        behaviour: Convolution mode. If ``'corr'`` (default), performs
            cross-correlation. If ``'conv'``, performs true convolution
            by flipping the kernel spatially.

    Returns:
        Tensor: The filtered tensor. If ``padding='same'``, the output shape
        is :math:`(B, C, H, W)`. If ``padding='valid'``, the output shape is
        :math:`(B, C, H - kH + 1, W - kW + 1)`.

    Note:
        - Internally uses real-valued FFTs (`rfftn` / `irfftn`).
        - Linear convolution is achieved by appropriate spatial padding
          and cropping, avoiding circular convolution artifacts.
        - No stride or dilation is supported.
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ["B", "H", "W"])

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f"Invalid border, {border_type}. Expected one of {_VALID_BORDERS}",
    )

    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f"Invalid padding mode, {padding}. Expected one of {_VALID_PADDING}",
    )

    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f"Invalid behaviour mode, {behaviour}. Expected one of {_VALID_BEHAVIOUR}",
    )

    _, c, _, _ = input.shape
    kh, kw = kernel.shape[-2:]

    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    # Expand kernel across channels
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # Padding (spatial domain)
    if padding == "same":
        padding_shape = _compute_padding([kh, kw])
        input_padded = F.pad(input, padding_shape, mode=border_type)
    else:
        input_padded = input

    padded_h, padded_w = input_padded.shape[-2:]

    input_padded = input_padded.contiguous()
    tmp_kernel = tmp_kernel.contiguous()

    # FFT
    input_fr = torch.fft.rfftn(input_padded, dim=(-2, -1))
    kernel_fr = torch.fft.rfftn(tmp_kernel, s=(padded_h, padded_w), dim=(-2, -1))

    # Correlation via conjugation
    output_fr = input_fr * torch.conj(kernel_fr)

    # Inverse FFT
    output = torch.fft.irfftn(output_fr, s=(padded_h, padded_w), dim=(-2, -1))

    # Crop to valid region
    crop_h = padded_h - kh + 1
    crop_w = padded_w - kw + 1
    output = output[..., :crop_h, :crop_w].contiguous()

    return output
