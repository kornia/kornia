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
from torch import nn


def _rgb_to_y(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to full-range YCbCr.

    ``ycbcr = rgb_to_ycbcr(image)`` converts a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` to Y, Cb and Cr with the
    BT.601 luma weights, all channels in :math:`[0, 1]` and Cb and Cr centred at 0.5.

    .. image:: _static/img/rgb_to_ycbcr.png

    Convention:
        Channels are Y, Cb, Cr at axis -3. The values equal OpenCV's float ``COLOR_RGB2YCrCb``, which stores Cr before
        Cb. skimage ``rgb2ycbcr`` differs: it uses studio range, with Y in 16..235.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        YCbCr image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`ycbcr_to_rgb`: the inverse.
        - :func:`rgb_to_y`: keeps only the Y channel.
        - :class:`RgbToYcbcr`: the module form.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = _rgb_to_y(r, g, b)
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to the Y channel of YCbCr.

    ``y = rgb_to_y(image)`` returns the BT.601 luma, ``0.299 R + 0.587 G + 0.114 B``, of a ``(*, 3, H, W)`` RGB image
    in :math:`[0, 1]` as one channel.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.

    Returns:
        Y image with shape ``(*, 1, H, W)``.

    See Also:
        - :func:`rgb_to_ycbcr`: returns Cb and Cr as well.
        - :func:`rgb_to_grayscale`: the same weights by default, with custom ``rgb_weights`` allowed.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    y: torch.Tensor = _rgb_to_y(r, g, b)
    return y


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a full-range YCbCr image to RGB.

    ``rgb = ycbcr_to_rgb(image)`` converts a ``(*, 3, H, W)`` YCbCr image in :math:`[0, 1]`, with Cb and Cr centred
    at 0.5 as :func:`rgb_to_ycbcr` returns them, to RGB.

    Convention:
        Channels are Y, Cb, Cr at axis -3. The result is clamped to :math:`[0, 1]`.

    Args:
        image: YCbCr image with shape ``(*, 3, H, W)``.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_ycbcr`: the inverse.
        - :class:`YcbcrToRgb`: the module form.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    # Exact inverse of the forward coefficients (0.564, 0.713) above: 1/0.713,
    # 1/0.564, and the derived green terms. The previously rounded values
    # (1.403, 0.714, 0.344, 1.773) were not a true inverse, so a
    # rgb->ycbcr->rgb round-trip drifted by ~2.7e-4 instead of being lossless.
    r: torch.Tensor = y + 1.4025245442 * cr_shifted
    g: torch.Tensor = y - 0.7144034731 * cr_shifted - 0.3443401356 * cb_shifted
    b: torch.Tensor = y + 1.7730496454 * cb_shifted
    return torch.stack([r, g, b], -3).clamp(0, 1)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr (module form).

    ``out = RgbToYcbcr()(image)`` is :func:`rgb_to_ycbcr`.

    Convention:
        - See the convention block of :func:`rgb_to_ycbcr`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_ycbcr`: the one-call form.
        - :class:`YcbcrToRgb`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to YCbCr.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to RGB channels, and ``H``/``W`` are height and width.

        Returns:
            YCbCr tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to RGB (module form).

    ``out = YcbcrToRgb()(image)`` is :func:`ycbcr_to_rgb`.

    Convention:
        - See the convention block of :func:`ycbcr_to_rgb`.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`ycbcr_to_rgb`: the one-call form.
        - :class:`RgbToYcbcr`: the inverse.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a YCbCr tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` corresponds to YCbCr channels, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return ycbcr_to_rgb(image)
