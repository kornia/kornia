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

from kornia.core.check import KORNIA_CHECK_SHAPE


def _apply_linear_transformation(
    image: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Apply a linear transformation (matrix multiplication + bias) to the image tensor.

    This function branches execution to maximize performance:
    - CPU: Unbinds the channel dimension and accumulates with ``torch.add`` per output channel.
    - GPU: Uses :func:`torch.nn.functional.conv2d` (highly optimized by cuDNN).

    Args:
        image: Input tensor with shape :math:`(*, C_{in}, H, W)`.
        kernel: Weight matrix with shape :math:`(C_{out}, C_{in})`.
        bias: Bias vector with shape :math:`(C_{out})`.

    Returns:
        Output tensor with shape :math:`(*, C_{out}, H, W)`.
    """
    # Empirical benchmarks show that Accumulation is faster on CPU for this specific pattern,
    # while conv2d offers significant speedups on GPU/CUDA.
    # We branch to ensure optimal performance on both devices.
    # BRANCH 1: CPU (Accumulation)
    if image.device.type == "cpu":
        # CPU Optimization: Unbind and accumulate is faster than einsum for small C_in/C_out
        x_unbound = image.unbind(-3)
        out_channels = []

        for i, row in enumerate(kernel):
            # Initialize accumulator with bias (if present) to avoid creating a zero tensor
            acc = bias[i] if bias is not None else torch.tensor(0.0, device=image.device, dtype=image.dtype)

            for j, coeff in enumerate(row):
                # acc += input[j] * coeff
                # Using torch.add with alpha is the most efficient scalar-tensor multiplication
                acc = torch.add(acc, x_unbound[j], alpha=float(coeff))

            out_channels.append(acc)

        return torch.stack(out_channels, dim=-3)

    # GPU/Accelerators path (Conv2d)
    else:
        input_shape = image.shape
        B, C_in, H, W = input_shape[:-3], input_shape[-3], input_shape[-2], input_shape[-1]
        C_out, _ = kernel.shape

        # Reshape input to (-1, C_in, H, W) for conv2d
        input_flat = image.reshape(-1, C_in, H, W)

        # Reshape kernel to (C_out, C_in, 1, 1)
        weight = kernel.reshape(C_out, C_in, 1, 1)

        out_flat = F.conv2d(input_flat, weight, bias)

        # Reshape back to (*, C_out, H, W)
        return out_flat.reshape(B + (C_out, H, W))


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    image_compute = image if image.is_floating_point() else image.float()

    # NOTE: Coefficients are derived from the existing logic to ensure test parity:
    # Y  = 0.299R + 0.587G + 0.114B
    # Cb = (B - Y) * 0.564 + 0.5
    # Cr = (R - Y) * 0.713 + 0.5
    kernel = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.168636, -0.331068, 0.499704],
            [0.499813, -0.418531, -0.081282],
        ],
        device=image_compute.device,
        dtype=image_compute.dtype,
    )

    bias = torch.tensor([0.0, 0.5, 0.5], device=image_compute.device, dtype=image_compute.dtype)

    return _apply_linear_transformation(image_compute, kernel, bias)


def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    image_compute = image if image.is_floating_point() else image.float()

    kernel = torch.tensor([[0.299, 0.587, 0.114]], device=image_compute.device, dtype=image_compute.dtype)

    return _apply_linear_transformation(image_compute, kernel)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5

    """
    KORNIA_CHECK_SHAPE(image, ["*", "3", "H", "W"])

    image_compute = image if image.is_floating_point() else image.float()

    # Coefficients for YCbCr to RGB
    # R = Y + 1.403 * (Cr - 0.5)
    # G = Y - 0.714 * (Cr - 0.5) - 0.344 * (Cb - 0.5)
    # B = Y + 1.773 * (Cb - 0.5)
    #
    # We can fold the -0.5 subtraction into the bias term to avoid creating 'image_shifted':
    # Bias_R = 1.403 * (-0.5) = -0.7015
    # Bias_G = (-0.714 * -0.5) + (-0.344 * -0.5) = 0.357 + 0.172 = 0.529
    # Bias_B = 1.773 * (-0.5) = -0.8865

    # Optimized CPU Path: Explicit AXPY Unrolling
    # 1. The YCbCr->RGB matrix has several zero coefficients (e.g., R does not depend on Cb and B does not depend on Cr,
    #    with rows [1.0, 0.0, 1.403] and [1.0, 1.773, 0.0] respectively). A generic matmul multiplies by these zeros,
    #    wasting cycles.
    # 2. We unroll the math to skip zero-ops entirely.
    if image.device.type == "cpu":
        y, cb, cr = image_compute.unbind(-3)

        # R = Y + 1.403 * Cr - 0.7015
        r = torch.add(y, cr, alpha=1.403).add_(-0.7015)

        # G = Y - 0.714 * Cr - 0.344 * Cb + 0.529
        g = torch.add(y, cr, alpha=-0.714).add_(cb, alpha=-0.344).add_(0.529)

        # B = Y + 1.773 * Cb - 0.8865
        b = torch.add(y, cb, alpha=1.773).add_(-0.8865)

        return torch.stack([r, g, b], -3).clamp(0, 1)

    kernel = torch.tensor(
        [
            [1.0, 0.0, 1.403],
            [1.0, -0.344, -0.714],
            [1.0, 1.773, 0.0],
        ],
        device=image_compute.device,
        dtype=image_compute.dtype,
    )

    # Pre-computed bias to avoid allocating 'image_shifted'
    bias = torch.tensor([-0.7015, 0.529, -0.8865], device=image_compute.device, dtype=image_compute.dtype)

    out = _apply_linear_transformation(image_compute, kernel, bias)

    return out.clamp(0, 1)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)
