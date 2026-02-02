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


@dataclass(frozen=True)
class ImageSize:
    r"""Data class to represent image shape.

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
    r"""Enum that represents the color space of an image."""

    UNKNOWN = 0  # for now, in case of multi band images
    GRAY = 1
    RGB = 2
    BGR = 3


@dataclass(frozen=True)
class PixelFormat:
    r"""Data class to represent the pixel format of an image.

    Args:
        color_space: color space.
        bit_depth: the number of bits per channel.

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
    r"""Enum that represents the channels order of an image."""

    CHANNELS_FIRST = 0
    CHANNELS_LAST = 1


@dataclass(frozen=True)
class ImageLayout:
    """Data class to represent the layout of an image.

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


def KORNIA_CHECK_IMAGE_LAYOUT(
    x: torch.Tensor,
    layout: ImageLayout,
    msg: str | None = None,
    raises: bool = True,
) -> bool:
    """Check tensor shape matches the expected ImageLayout.

    Args:
        x: tensor to validate.
        layout: expected image layout.
        msg: custom error message.
        raises: if True, raise ShapeError on mismatch.

    Returns:
        True if shape matches, False otherwise (when raises=False).

    """
    from kornia.core.check import KORNIA_CHECK_SHAPE

    if layout.channels_order == ChannelsOrder.CHANNELS_FIRST:
        shape = [str(layout.channels), str(layout.image_size.height), str(layout.image_size.width)]
    elif layout.channels_order == ChannelsOrder.CHANNELS_LAST:
        shape = [str(layout.image_size.height), str(layout.image_size.width), str(layout.channels)]
    else:
        raise NotImplementedError(f"Layout {layout.channels_order} not implemented.")

    return KORNIA_CHECK_SHAPE(x, shape, msg, raises)


# TODO: define CompressedImage
