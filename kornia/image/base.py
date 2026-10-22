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

from dataclasses import dataclass
from enum import Enum

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE, CheckMessage
from kornia.core.exceptions import ShapeError


@dataclass(frozen=True)
class ImageSize:
    r"""Data class to represent the size of an image, without its channels.

    Convention:
        See the Convention block of :class:`~kornia.image.Image`. The positional order is ``(height, width)``,
        kornia's order for sizes and the reverse of :math:`(x, y)` points.

    Args:
        height: image height.
        width: image width.

    Example:
        >>> size = ImageSize(3, 4)
        >>> size.height
        3
        >>> size.width
        4

    """

    height: int | torch.Tensor
    width: int | torch.Tensor


class ColorSpace(Enum):
    r"""Enum that represents the color space of an image.

    Convention:
        See the Convention block of :class:`~kornia.image.Image`, which checks the color space against the channel
        count. There is no color space with alpha: a 4-channel image is ``UNKNOWN``.
    """

    UNKNOWN = 0  # for now, in case of multi band images
    GRAY = 1
    RGB = 2
    BGR = 3


@dataclass(frozen=True)
class PixelFormat:
    r"""Data class to represent the pixel format of an image.

    Convention:
        See the Convention block of :class:`~kornia.image.Image`, which checks a pixel format against the data; the
        data class itself checks nothing.

    Args:
        color_space: color space.
        bit_depth: the storage size of one channel value in bits, ``8 * element_size()`` of the data: 8 for
            ``uint8``, 32 for ``float32``.

    Example:
        >>> pixel_format = PixelFormat(ColorSpace.RGB, 8)
        >>> pixel_format.color_space
        <ColorSpace.RGB: 2>
        >>> pixel_format.bit_depth
        8

    """

    color_space: ColorSpace
    bit_depth: int


class ChannelsOrder(Enum):
    r"""Enum that represents the channels order of an image.

    Convention:
        See the Convention block of :class:`~kornia.image.Image`. ``CHANNELS_FIRST`` stores :math:`(C, H, W)` and
        ``CHANNELS_LAST`` stores :math:`(H, W, C)`.
    """

    CHANNELS_FIRST = 0
    CHANNELS_LAST = 1


@dataclass(frozen=True)
class ImageLayout:
    """Data class to represent the layout of an image.

    Convention:
        See the Convention block of :class:`~kornia.image.Image`, which checks a layout against the data; the data
        class itself checks nothing.

    Args:
        image_size: image size.
        channels: number of channels.
        channels_order: channels order.

    Example:
        >>> layout = ImageLayout(ImageSize(3, 4), 3, ChannelsOrder.CHANNELS_LAST)
        >>> layout.image_size
        ImageSize(height=3, width=4)
        >>> layout.channels
        3
        >>> layout.channels_order
        <ChannelsOrder.CHANNELS_LAST: 1>

    """

    image_size: ImageSize
    channels: int
    channels_order: ChannelsOrder


def _layout_size_to_str(value: float | torch.Tensor | str) -> str:
    """Normalize an ``ImageLayout`` size entry for shape comparison.

    ``ImageSize`` may hold ints, floats, or zero-dim tensors (#4263). ``str()`` of a float or tensor is not
    all digits, so :func:`~kornia.core.check.KORNIA_CHECK_SHAPE` would skip the comparison and accept a
    mismatching image (#6244). Integral numbers therefore compare by their integer value; anything else
    keeps the historical string form.
    """
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, torch.Tensor) and value.numel() == 1:
        number = float(value.item())
    else:
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def KORNIA_CHECK_IMAGE_LAYOUT(
    x: torch.Tensor,
    layout: ImageLayout,
    msg: CheckMessage | None = None,
    raises: bool = True,
) -> bool:
    """Check tensor shape matches the expected ImageLayout.

    Args:
        x: tensor to validate.
        layout: expected image layout.
        msg: custom error message, or a function that returns it; see :func:`~kornia.core.check.KORNIA_CHECK`.
        raises: if True, raise ShapeError on mismatch.

    Returns:
        True if shape matches, False otherwise (when raises=False).

    """
    # A fractional size can never equal a tensor dimension, so it always
    # mismatches; reject it instead of letting the comparison skip it.
    for entry in (layout.channels, layout.image_size.height, layout.image_size.width):
        if isinstance(entry, (int, float)):
            number = float(entry)
        elif isinstance(entry, torch.Tensor) and entry.numel() == 1:
            number = float(entry.item())
        else:
            continue
        if not number.is_integer():
            if raises:
                raise ShapeError(f"Layout size entry {entry} is not an integer.")
            return False

    if layout.channels_order == ChannelsOrder.CHANNELS_FIRST:
        shape = [
            _layout_size_to_str(layout.channels),
            _layout_size_to_str(layout.image_size.height),
            _layout_size_to_str(layout.image_size.width),
        ]
    elif layout.channels_order == ChannelsOrder.CHANNELS_LAST:
        shape = [
            _layout_size_to_str(layout.image_size.height),
            _layout_size_to_str(layout.image_size.width),
            _layout_size_to_str(layout.channels),
        ]
    else:
        if raises:
            raise NotImplementedError(f"Layout {layout.channels_order} not implemented.")
        return False

    return KORNIA_CHECK_SHAPE(x, shape, msg, raises)


# TODO: define CompressedImage
