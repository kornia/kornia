from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kornia.core import Tensor


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

    height: int | Tensor
    width: int | Tensor


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


# TODO: define CompressedImage
