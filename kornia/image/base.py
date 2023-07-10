from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass
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
    height: int
    width: int


class PixelFormat(Enum):
    r"""Enum that represents the pixel format of an image."""
    GRAY = 0
    RGB = 1
    BGR = 2


class ChannelsOrder(Enum):
    r"""Enum that represents the channels order of an image."""
    CHANNEL_FIRST = 0
    CHANNEL_LAST = 1


@dataclass
class ImageLayout:
    """Data class to represent the layout of an image.

    Args:
        image_size: image size.
        channels: number of channels.
        pixel_format: pixel format.
        channels_order: channels order.

    Example:
        >>> layout = ImageLayout(ImageSize(3, 4), 3, PixelFormat.RGB, ChannelsOrder.CHANNEL_LAST)
        >>> layout.image_size
        ImageSize(height=3, width=4)
        >>> layout.channels
        3
        >>> layout.pixel_format
        <PixelFormat.RGB: 1>
        >>> layout.channels_order
        <ChannelsOrder.CHANNEL_LAST: 1>
    """

    image_size: ImageSize
    channels: int
    pixel_format: PixelFormat
    channels_order: ChannelsOrder


# TODO: define CompressedImage
